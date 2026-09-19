from __future__ import print_function

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from residual_audit_utils import capture_residual_outputs, residual_diagnostics
from gate_audit_utils import capture_gate_outputs, gate_diagnostics
from datasets import find_dataset_def
from gradient_audit_utils import build_audit_cases, run_audit_suite, _write_json_atomic
from models.sger_refinement import ResidualDepthHead
from models import CascadeMVSNet, cas_mvsnet_loss
from gate_bce_audit_utils import bce_forward_tensors, compare_forward


def build_parser():
    parser = argparse.ArgumentParser(
        description="Experiment 15-C single-batch SGER gradient audit")
    parser.add_argument("--trainpath", required=True)
    parser.add_argument("--trainlist", required=True)
    parser.add_argument("--loadckpt", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--audit_epoch", type=int, required=True,
                        help="zero-based epoch index used by SGER schedules")
    parser.add_argument("--dataset", default="dtu_yao")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gate_bce_mode", choices=("legacy_clamp", "stable_logprob"),
                        default="legacy_clamp")
    parser.add_argument("--gate_bce_ab", action="store_true",
                        help="pair legacy/stable BCE with a fixed activation")
    parser.add_argument("--activation_ab", action="store_true",
                        help="audit ReLU and LeakyReLU on identical weights and batches")
    parser.add_argument("--verbose_activation_progress", action="store_true",
                        help="print timing for each residual activation statistic")
    parser.add_argument("--batch_indices", default="0",
                        help="comma-separated zero-based fixed batch indices, e.g. 0,1,2,3,4")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--nviews", type=int, default=3)
    parser.add_argument("--numdepth", type=int, default=192)
    parser.add_argument("--interval_scale", type=float, default=1.06)
    parser.add_argument("--ndepths", default="48,32,8")
    parser.add_argument("--depth_inter_r", default="4,2,1")
    parser.add_argument("--dlossw", default="0.5,1.0,2.0")
    parser.add_argument("--cr_base_chs", default="8,8,8")
    parser.add_argument("--share_cr", action="store_true")
    parser.add_argument("--grad_method", default="detach",
                        choices=("detach", "undetach"))

    parser.set_defaults(use_sger_lite=True)
    parser.add_argument("--use_sger_lite", dest="use_sger_lite",
                        action="store_true")
    parser.add_argument("--no_sger_lite", dest="use_sger_lite",
                        action="store_false")
    parser.add_argument("--sger_share", action="store_true")
    parser.add_argument("--sger_feature_channels", type=int, default=8)
    parser.add_argument("--sger_hidden_channels", type=int, default=32)
    parser.add_argument("--sger_gate_channels", type=int, default=16)
    parser.add_argument("--sger_max_residual_ratio", type=float, default=0.25)
    parser.add_argument("--sger_residual_final_activation", choices=("relu", "leaky_relu"), default="relu")
    parser.add_argument("--sger_residual_negative_slope", type=float, default=0.01)
    parser.add_argument("--sger_warmup_start_epoch", type=int, default=7)
    parser.add_argument("--sger_warmup_end_epoch", type=int, default=10)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--raw_depth_loss_weight", type=float, default=1.0)
    parser.add_argument("--refined_depth_loss_weight", type=float, default=1.0)
    parser.add_argument("--residual_loss_weight", type=float, default=0.01)
    parser.add_argument("--gate_loss_weight", type=float, default=0.0)
    parser.add_argument("--safe_refine_loss_weight", type=float, default=0.1)
    parser.add_argument("--safe_refine_margin", type=float, default=0.0)
    parser.add_argument("--residual_target_loss_weight", type=float,
                        default=0.05)
    parser.add_argument("--gate_benefit_loss_weight", type=float, default=0.05)
    parser.add_argument("--residual_target_ratio", type=float, default=0.25)
    parser.add_argument("--benefit_margin_ratio", type=float, default=0.05)
    parser.add_argument("--normal_smooth_loss_weight", type=float, default=0.02)
    parser.add_argument("--curv_loss_weight", type=float, default=0.005)
    parser.add_argument("--edge_smooth_loss_weight", type=float, default=0.005)
    parser.add_argument("--depth_normal_loss_weight", type=float, default=0.03)
    parser.add_argument("--depth_normal_conf_threshold", type=float, default=0.8)
    parser.add_argument("--edge_grad_threshold", type=float, default=0.05)
    parser.add_argument("--geometry_conf_mid", type=float, default=0.65)
    parser.add_argument("--geometry_k_conf", type=float, default=10.0)
    parser.add_argument("--geometry_edge_mid", type=float, default=0.25)
    parser.add_argument("--geometry_k_edge", type=float, default=10.0)
    parser.add_argument("--geometry_w_min", type=float, default=0.05)
    parser.add_argument("--disable_dual_region_curvature",
                        dest="use_dual_region_curvature", action="store_false")
    parser.set_defaults(use_dual_region_curvature=True)
    parser.add_argument("--region_lambda_a", type=float, default=1.5)
    parser.add_argument("--region_lambda_b", type=float, default=1.0)
    parser.add_argument("--region_edge_threshold", type=float, default=0.25)
    parser.add_argument("--region_depth_threshold", type=float, default=0.02)
    parser.add_argument("--region_curv_threshold", type=float, default=0.02)
    parser.add_argument("--region_smooth_k", type=float, default=2.0)
    return parser


def _parse_numbers(value, conversion):
    return [conversion(item) for item in value.split(",") if item]


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _is_sger_lite_parameter(module, name):
    prefixes = (
        "normal_head.{}.".format(module.num_stage - 1),
        "sger_blocks.",
        "shared_sger.",
        "sger_feature_adapters.{}.".format(module.num_stage - 1),
    )
    return name.startswith(prefixes)


def _build_optimizer(model, args):
    sger_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if _is_sger_lite_parameter(model, name):
            sger_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    groups = [
        {"params": sger_parameters, "lr": args.lr, "name": "sger_lite"},
        {
            "params": backbone_parameters,
            "lr": args.lr * args.backbone_lr_scale,
            "name": "backbone",
        },
    ]
    return optim.Adam(groups, lr=args.lr, betas=(0.9, 0.999),
                      weight_decay=args.wd)


def _load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = (
        "sger_blocks.", "shared_sger.", "sger_feature_adapters.")
    invalid_missing = [key for key in missing
                       if not key.startswith(allowed_missing)]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "incompatible checkpoint; missing={}, unexpected={}".format(
                invalid_missing, list(unexpected)))
    return {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_missing_keys": list(missing),
        "checkpoint_unexpected_keys": list(unexpected),
    }


def _base_loss_weights(args):
    keys = (
        "raw_depth_loss_weight", "refined_depth_loss_weight",
        "residual_loss_weight", "gate_loss_weight",
        "safe_refine_loss_weight", "residual_target_loss_weight",
        "gate_benefit_loss_weight", "normal_smooth_loss_weight",
        "curv_loss_weight", "edge_smooth_loss_weight",
        "depth_normal_loss_weight",
    )
    return {key: getattr(args, key) for key in keys}


def _loss_kwargs(args, sample, case):
    values = dict(case.loss_weights)
    values.update({
        "gate_bce_mode": args.gate_bce_mode,
        "imgs": sample["imgs"],
        "proj_matrices": sample["proj_matrices"],
        "dlossw": _parse_numbers(args.dlossw, float),
        "safe_refine_margin": args.safe_refine_margin,
        "residual_target_ratio": args.residual_target_ratio,
        "benefit_margin_ratio": args.benefit_margin_ratio,
        "residual_target_loss_scale": case.residual_target_scale,
        "gate_benefit_loss_scale": case.gate_benefit_scale,
        "sger_loss_scale": case.sger_scale,
        "depth_normal_conf_threshold": args.depth_normal_conf_threshold,
        "edge_grad_threshold": args.edge_grad_threshold,
        "geometry_conf_mid": args.geometry_conf_mid,
        "geometry_k_conf": args.geometry_k_conf,
        "geometry_edge_mid": args.geometry_edge_mid,
        "geometry_k_edge": args.geometry_k_edge,
        "geometry_w_min": args.geometry_w_min,
        "use_dual_region_curvature": args.use_dual_region_curvature,
        "region_lambda_a": args.region_lambda_a,
        "region_lambda_b": args.region_lambda_b,
        "region_edge_threshold": args.region_edge_threshold,
        "region_depth_threshold": args.region_depth_threshold,
        "region_curv_threshold": args.region_curv_threshold,
        "region_smooth_k": args.region_smooth_k,
        "return_extra": True,
    })
    return values


def audit_loss_inputs(outputs, case):
    """Use shallow stage copies to suppress unconditional coarse depth supervision.

    Adding a depth_raw alias makes the production loss respect the zero raw
    weight on stages without SGER. Forward tensors and gradient paths are intact.
    """
    if case.name.startswith("full_"):
        return outputs
    stages = {}
    for key, value in outputs.items():
        if key.startswith("stage") and key[5:].isdigit():
            stage = dict(value)
            stage.setdefault("depth_raw", stage["depth"])
            stages[key] = stage
    if case.name in ("refined_only", "residual_target_only",
                     "gate_benefit_only", "safe_refine_only"):
        stages = {key: value for key, value in stages.items()
                  if "depth_refined" in value}
    if case.name.startswith("stage3_"):
        if "stage3" not in stages or "depth_refined" not in stages["stage3"]:
            raise ValueError("Stage3 geometry audit requires Stage3 SGER outputs")
        stages = {"stage3": stages["stage3"]}
    return stages


def _print_report(report, output_json):
    print("\nExperiment 15-C single-batch gradient audit")
    print("JSON report: {}".format(os.path.abspath(output_json)))
    for case in report["cases"]:
        metrics = case["forward_metrics"]
        print("\n[{name}] loss={loss} raw_err={raw} refined_err={refined} "
              "delta={delta}".format(
                  name=case["name"], loss=case["loss"],
                  raw=metrics.get("raw_abs_error"),
                  refined=metrics.get("refined_abs_error"),
                  delta=metrics.get("raw_to_refined_error_delta")))
        print("  gradient_summary: {}".format(case["gradient_summary"]))
        for group_name in (
                "normal_head", "sger_feature_projection",
                "sger_residual_body", "sger_residual_output",
                "sger_gate_a", "sger_gate_b", "backbone_feature",
                "cost_regularization"):
            group = case["parameter_gradients"].get(group_name)
            if group is not None:
                print("  {:24s} l2={:.6e} rms={:.6e} no_grad={}".format(
                    group_name, group["l2"], group["rms"],
                    group["missing_grad_tensors"]))
        for warning in case["warnings"]:
            print("  WARNING: {}".format(warning))
        for error in case["fatal_errors"]:
            print("  FATAL: {}".format(error))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.activation_ab and args.gate_bce_ab:
        raise ValueError("activation_ab and gate_bce_ab must be separate experiments")
    if not args.use_sger_lite:
        raise ValueError("Experiment 15-C audit requires --use_sger_lite")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.num_workers != 0:
        raise ValueError("reproducible audit requires --num_workers 0")
    if not 0.0 < args.backbone_lr_scale <= 1.0:
        raise ValueError("backbone_lr_scale must be in (0, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    ndepths = _parse_numbers(args.ndepths, int)
    depth_ratios = _parse_numbers(args.depth_inter_r, float)
    model = CascadeMVSNet(
        refine=False,
        ndepths=ndepths,
        depth_interals_ratio=depth_ratios,
        share_cr=args.share_cr,
        cr_base_chs=_parse_numbers(args.cr_base_chs, int),
        grad_method=args.grad_method,
        use_sger_lite=True,
        sger_share=args.sger_share,
        sger_feature_channels=args.sger_feature_channels,
        sger_hidden_channels=args.sger_hidden_channels,
        sger_gate_channels=args.sger_gate_channels,
        sger_max_residual_ratio=args.sger_max_residual_ratio,
        sger_residual_final_activation=args.sger_residual_final_activation,
        sger_residual_negative_slope=args.sger_residual_negative_slope,
        detach_refined_feedback=True,
        sger_gate_kwargs={
            "threshold_edge": args.region_edge_threshold,
            "threshold_depth": args.region_depth_threshold,
            "threshold_curv": args.region_curv_threshold,
            "conf_mid": args.geometry_conf_mid,
            "k_conf": args.geometry_k_conf,
            "smooth_k": args.region_smooth_k,
        },
    )
    checkpoint_metadata = (_load_checkpoint(model, args.loadckpt)
                           if args.loadckpt else {"initialization": "seeded_random"})
    model.to(device)
    model.train()
    optimizer = _build_optimizer(model, args)
    freeze_backbone = args.audit_epoch < args.freeze_backbone_epochs
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            not freeze_backbone or _is_sger_lite_parameter(model, name))

    dataset_class = find_dataset_def(args.dataset)
    dataset = dataset_class(
        args.trainpath, args.trainlist, "train", args.nviews,
        args.numdepth, args.interval_scale)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, pin_memory=False)
    batch_indices = _parse_numbers(args.batch_indices, int)
    if (not batch_indices or len(set(batch_indices)) != len(batch_indices)
            or any(index < 0 or index >= len(loader) for index in batch_indices)):
        raise ValueError("batch_indices must be unique indices within the dataset batches")
    final_stage_key = "stage{}".format(len(ndepths))

    cases = build_audit_cases(
        args.audit_epoch, _base_loss_weights(args),
        args.sger_warmup_start_epoch, args.sger_warmup_end_epoch)

    bce_reference = {}
    bce_comparisons = {}
    active_bce_mode = args.gate_bce_mode
    ab_raw = {}
    ab_comparison = {}

    def evaluate(case):
        with capture_gate_outputs(model) as captured, capture_residual_outputs(
                model, progress=args.verbose_activation_progress) as residual_captured:
            outputs = model(
                sample["imgs"], sample["proj_matrices"],
                sample["depth_values"])
        if args.activation_ab and case.name == "full_schedule":
            for key in ("depth_raw", "photometric_confidence"):
                value = outputs[final_stage_key][key].detach().cpu()
                if activation == "relu":
                    ab_raw[key] = value.clone()
                else:
                    ab_comparison[key + "_max_abs_difference"] = float(
                        (value - ab_raw[key]).abs().max().item())
        if args.gate_bce_ab:
            current = bce_forward_tensors(outputs[final_stage_key],
                sample["depth"][final_stage_key], sample["mask"][final_stage_key],
                args.benefit_margin_ratio)
            if active_bce_mode == "legacy_clamp":
                bce_reference[case.name] = current
            else:
                bce_comparisons[case.name] = compare_forward(bce_reference.pop(case.name), current)
                if any(v != 0 for v in bce_comparisons[case.name].values()):
                    raise RuntimeError("BCE A/B changed forward tensors: " + case.name)
        diagnostics = {}
        residual_stats = {}
        for key, stage in outputs.items():
            if key.startswith("stage") and key[5:].isdigit() and "benefit_gate" in stage:
                residual_stats[key] = residual_diagnostics(
                    stage, residual_captured.get(id(stage["raw_depth_residual"]), {}),
                    sample["mask"][key])
                diagnostics[key] = gate_diagnostics(
                    stage, captured.get(id(stage["benefit_gate"]), {}),
                    sample["depth"][key], sample["mask"][key], args.benefit_margin_ratio)
        captured.clear()
        residual_captured.clear()
        loss_options = _loss_kwargs(args, sample, case)
        loss_options["gate_bce_mode"] = active_bce_mode
        loss_result = cas_mvsnet_loss(
            audit_loss_inputs(outputs, case), sample["depth"], sample["mask"],
            **loss_options)
        loss = loss_result[0]
        extras = dict(loss_result[-1])
        extras["actual_gate_bce_mode"] = active_bce_mode
        extras["benefit_margin_ratio"] = args.benefit_margin_ratio
        extras["gate_diagnostics"] = diagnostics
        extras["residual_diagnostics"] = residual_stats
        stage = outputs[final_stage_key]
        maximum = args.sger_max_residual_ratio * float(stage["depth_interval"])
        extras["residual_bound_checks"] = {
            "max_abs_proposal": stage["raw_depth_residual"].detach().abs().max(),
            "max_abs_effective_residual": stage["depth_residual"].detach().abs().max(),
            "proposal_bound": maximum,
            "effective_bound": maximum * case.sger_scale,
            "proposal_all_finite": bool(torch.isfinite(stage["raw_depth_residual"]).all()),
            "effective_all_finite": bool(torch.isfinite(stage["depth_residual"]).all()),
        }
        return (
            outputs, loss, extras,
            sample["depth"][final_stage_key],
            sample["mask"][final_stage_key] > 0.5,
        )

    metadata = {
        "audit_epoch": args.audit_epoch,
        "checkpoint": os.path.abspath(args.loadckpt) if args.loadckpt else None,
        "arguments": vars(args),
        "torch_version": torch.__version__,
        "batch_selection": "first batch, shuffle=False",
        "dataset": args.dataset,
        "trainpath": os.path.abspath(args.trainpath),
        "trainlist": os.path.abspath(args.trainlist),
        "seed": args.seed,
        "device": str(device),
        "freeze_backbone": freeze_backbone,
    }
    metadata.update(checkpoint_metadata)
    failed = False
    # Direct batch sampler records exact dataset indices and avoids loading
    # skipped batches. Each suite restores weights/BN before the next batch.
    selections = [list(range(index * args.batch_size,
                             min((index + 1) * args.batch_size, len(dataset))))
                  for index in batch_indices]
    selected_loader = DataLoader(dataset, batch_sampler=selections,
                                 num_workers=0, pin_memory=False)
    for batch_index, indices, cpu_sample in zip(batch_indices, selections, selected_loader):
        sample = _move_to_device(cpu_sample, device)
        batch_metadata = dict(metadata)
        batch_metadata.update(batch_selection="fixed indices, shuffle=False",
                              batch_index=batch_index, dataset_indices=indices,
                              model_mode="train")
        if hasattr(dataset, "metas"):
            batch_metadata["sample_metas"] = [dataset.metas[i] for i in indices]
        path = args.output_json
        if len(batch_indices) > 1 or args.gate_bce_ab:
            stem, extension = os.path.splitext(path)
            path = "{}_batch{:04d}{}".format(stem, batch_index, extension or ".json")
        print("Auditing batch {} (dataset indices {})".format(batch_index, indices), flush=True)
        variants = ("relu", "leaky_relu") if args.activation_ab else (args.sger_residual_final_activation,)
        if args.gate_bce_ab:
            variants = ("legacy_clamp", "stable_logprob")
        heads = [m for m in model.modules() if isinstance(m, ResidualDepthHead)]
        original_activations = [head.body[5] for head in heads]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        variant_reports = {}
        bce_reference.clear()
        bce_comparisons.clear()
        ab_raw.clear()
        ab_comparison.clear()
        try:
            for variant in variants:
                activation = args.sger_residual_final_activation if args.gate_bce_ab else variant
                active_bce_mode = variant if args.gate_bce_ab else args.gate_bce_mode
                for head in heads:
                    head.set_final_activation(activation, args.sger_residual_negative_slope)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                variant_metadata = dict(batch_metadata)
                variant_metadata.update(actual_gate_bce_mode=active_bce_mode,
                                        actual_residual_final_activation=activation,
                                        actual_residual_negative_slope=(
                                            args.sger_residual_negative_slope if activation == "leaky_relu" else 0.0))
                variant_path = path
                if args.activation_ab:
                    stem, extension = os.path.splitext(path)
                    variant_path = stem + "_" + activation + (extension or ".json")
                if args.gate_bce_ab:
                    stem, extension = os.path.splitext(path)
                    variant_path = stem + "_" + activation + "_" + active_bce_mode + (extension or ".json")
                report = run_audit_suite(
                    model, optimizer, cases, evaluate,
                    final_stage_key=final_stage_key,
                    output_json=variant_path, metadata=variant_metadata)
                _print_report(report, variant_path)
                failed = failed or bool(report["fatal_errors"])
                variant_reports[variant] = {
                    "report": os.path.abspath(variant_path),
                    "fatal_errors": report["fatal_errors"],
                    "full_schedule": report["cases"][0]["forward_metrics"],
                    "residual_target_gradient_summary": next(
                        c["gradient_summary"] for c in report["cases"]
                        if c["name"] == "residual_target_only"),
                }
        finally:
            for head, original in zip(heads, original_activations):
                head.body[5] = original
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        if args.activation_ab:
            stem, extension = os.path.splitext(path)
            _write_json_atomic(stem + "_ab_comparison" + (extension or ".json"), {
                "metadata": batch_metadata, "variants": variant_reports,
                "full_schedule_raw_invariance": ab_comparison,
                "note": "Same batch, weights and BN state; no optimizer step. Nonzero gradients do not prove quality gains."})
        if args.gate_bce_ab:
            stem, extension = os.path.splitext(path)
            _write_json_atomic(stem + "_" + activation + "_bce_comparison" + (extension or ".json"), {
                "metadata": batch_metadata, "variants": variant_reports,
                "case_forward_invariance": bce_comparisons,
                "note": "Fixed activation, same weights/BN/RNG; loss-only A/B, no optimizer step."})
        del sample
    return 1 if failed else 0



if __name__ == "__main__":
    sys.exit(main())
