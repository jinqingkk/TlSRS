from __future__ import print_function

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import find_dataset_def
from gradient_audit_utils import build_audit_cases, run_audit_suite
from models import CascadeMVSNet, cas_mvsnet_loss


def build_parser():
    parser = argparse.ArgumentParser(
        description="Experiment 15-C single-batch SGER gradient audit")
    parser.add_argument("--trainpath", required=True)
    parser.add_argument("--trainlist", required=True)
    parser.add_argument("--loadckpt", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--audit_epoch", type=int, required=True,
                        help="zero-based epoch index used by SGER schedules")
    parser.add_argument("--dataset", default="dtu_yao")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
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
    parser.add_argument("--sger_warmup_start_epoch", type=int, default=7)
    parser.add_argument("--sger_warmup_end_epoch", type=int, default=10)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=8)
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
    if not args.use_sger_lite:
        raise ValueError("Experiment 15-C audit requires --use_sger_lite")
    if args.batch_size != 1:
        raise ValueError("single-batch audit requires --batch_size 1")
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
    checkpoint_metadata = _load_checkpoint(model, args.loadckpt)
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
        dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=False, pin_memory=False)
    try:
        sample = next(iter(loader))
    except StopIteration:
        raise RuntimeError("training dataset is empty")
    sample = _move_to_device(sample, device)
    final_stage_key = "stage{}".format(len(ndepths))

    cases = build_audit_cases(
        args.audit_epoch, _base_loss_weights(args),
        args.sger_warmup_start_epoch, args.sger_warmup_end_epoch)

    def evaluate(case):
        outputs = model(
            sample["imgs"], sample["proj_matrices"],
            sample["depth_values"])
        loss_result = cas_mvsnet_loss(
            outputs, sample["depth"], sample["mask"],
            **_loss_kwargs(args, sample, case))
        loss = loss_result[0]
        extras = loss_result[-1]
        return (
            outputs, loss, extras,
            sample["depth"][final_stage_key],
            sample["mask"][final_stage_key] > 0.5,
        )

    metadata = {
        "audit_epoch": args.audit_epoch,
        "checkpoint": os.path.abspath(args.loadckpt),
        "dataset": args.dataset,
        "trainpath": os.path.abspath(args.trainpath),
        "trainlist": os.path.abspath(args.trainlist),
        "seed": args.seed,
        "device": str(device),
        "freeze_backbone": freeze_backbone,
    }
    metadata.update(checkpoint_metadata)
    report = run_audit_suite(
        model, optimizer, cases, evaluate,
        final_stage_key=final_stage_key,
        output_json=args.output_json,
        metadata=metadata)
    _print_report(report, args.output_json)
    return 1 if report["fatal_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
