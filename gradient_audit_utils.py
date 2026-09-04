from __future__ import print_function

import copy
import json
import math
import os
import tempfile
from collections import namedtuple

import torch


AUDIT_TENSOR_KEYS = (
    "depth_raw",
    "raw_depth_residual",
    "benefit_gate",
    "geometry_gate",
    "depth_residual",
    "depth_refined",
    "normal",
    "photometric_confidence",
)

LOSS_WEIGHT_KEYS = (
    "raw_depth_loss_weight",
    "refined_depth_loss_weight",
    "residual_loss_weight",
    "gate_loss_weight",
    "safe_refine_loss_weight",
    "residual_target_loss_weight",
    "gate_benefit_loss_weight",
    "normal_smooth_loss_weight",
    "curv_loss_weight",
    "edge_smooth_loss_weight",
    "depth_normal_loss_weight",
)

AuditCase = namedtuple(
    "AuditCase",
    [
        "name",
        "loss_weights",
        "sger_scale",
        "residual_target_scale",
        "gate_benefit_scale",
    ],
)


def _sger_warmup_scale(epoch_idx, start_epoch, end_epoch):
    if start_epoch < 0 or end_epoch < start_epoch:
        raise ValueError("invalid SGER warm-up range")
    if epoch_idx < start_epoch:
        return 0.0
    if epoch_idx >= end_epoch:
        return 1.0
    return float(epoch_idx - start_epoch + 1) / float(
        end_epoch - start_epoch + 1)


def _experiment14_scales(epoch_idx):
    residual_scale = 1.0 if epoch_idx >= 4 else 0.0
    if epoch_idx < 7:
        gate_scale = 0.0
    elif epoch_idx >= 10:
        gate_scale = 1.0
    else:
        gate_scale = float(epoch_idx - 7 + 1) / float(10 - 7 + 1)
    return residual_scale, gate_scale


def build_audit_cases(epoch_idx, base_weights,
                      warmup_start_epoch=7, warmup_end_epoch=10):
    base = {key: float(base_weights.get(key, 0.0))
            for key in LOSS_WEIGHT_KEYS}
    scheduled_sger = _sger_warmup_scale(
        epoch_idx, warmup_start_epoch, warmup_end_epoch)
    scheduled_residual, scheduled_gate = _experiment14_scales(epoch_idx)

    cases = [
        AuditCase(
            "full_schedule", dict(base), scheduled_sger,
            scheduled_residual, scheduled_gate),
        AuditCase("full_forced", dict(base), 1.0, 1.0, 1.0),
    ]
    isolated = (
        ("refined_only", "refined_depth_loss_weight"),
        ("residual_target_only", "residual_target_loss_weight"),
        ("gate_benefit_only", "gate_benefit_loss_weight"),
        ("safe_refine_only", "safe_refine_loss_weight"),
    )
    for name, active_key in isolated:
        weights = {key: 0.0 for key in LOSS_WEIGHT_KEYS}
        weights[active_key] = 1.0
        cases.append(AuditCase(name, weights, 1.0, 1.0, 1.0))
    return cases


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def gradient_stats(gradient):
    if gradient is None:
        return {
            "present": False,
            "all_finite": True,
            "numel": 0,
            "finite_count": 0,
            "zero_ratio": None,
            "abs_mean": None,
            "abs_max": None,
            "rms": None,
            "l2": None,
        }
    grad = gradient.detach().float().reshape(-1)
    finite_mask = torch.isfinite(grad)
    finite = grad[finite_mask]
    numel = int(grad.numel())
    finite_count = int(finite_mask.sum().item())
    result = {
        "present": True,
        "all_finite": finite_count == numel,
        "numel": numel,
        "finite_count": finite_count,
        "zero_ratio": float((grad == 0).float().mean().item())
        if numel else None,
        "abs_mean": None,
        "abs_max": None,
        "rms": None,
        "l2": None,
    }
    if finite_count:
        result.update({
            "abs_mean": _safe_float(finite.abs().mean().item()),
            "abs_max": _safe_float(finite.abs().max().item()),
            "rms": _safe_float(finite.pow(2).mean().sqrt().item()),
            "l2": _safe_float(finite.norm().item()),
        })
    return result


def classify_parameter(name, final_stage_index):
    if name.startswith("normal_head.{}".format(final_stage_index)):
        return "normal_head"
    if "sger_feature_adapters" in name or ".feature_projection." in name:
        return "sger_feature_projection"
    if ".residual.output." in name:
        return "sger_residual_output"
    if ".residual.body." in name:
        return "sger_residual_body"
    if ".gate.gate_a." in name:
        return "sger_gate_a"
    if ".gate.gate_b." in name:
        return "sger_gate_b"
    if name.startswith("feature."):
        return "backbone_feature"
    if name.startswith("cost_regularization."):
        return "cost_regularization"
    if name.startswith("sger_blocks.") or name.startswith("shared_sger."):
        return "sger_other"
    return "other"


def parameter_group_stats(model, optimizer, final_stage_index):
    module = model.module if hasattr(model, "module") else model
    optimizer_ids = set(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"])
    groups = {}
    for name, parameter in module.named_parameters():
        group_name = classify_parameter(name, final_stage_index)
        group = groups.setdefault(group_name, {
            "parameter_tensors": 0,
            "parameter_values": 0,
            "missing_grad_tensors": 0,
            "missing_optimizer_tensors": 0,
            "nonfinite_values": 0,
            "grad_values": 0,
            "l2": 0.0,
            "rms": 0.0,
            "abs_max": 0.0,
            "missing_grad_names": [],
            "missing_optimizer_names": [],
        })
        group["parameter_tensors"] += 1
        group["parameter_values"] += int(parameter.numel())
        if id(parameter) not in optimizer_ids:
            group["missing_optimizer_tensors"] += 1
            group["missing_optimizer_names"].append(name)
        if parameter.grad is None:
            group["missing_grad_tensors"] += 1
            group["missing_grad_names"].append(name)
            continue
        grad = parameter.grad.detach().float()
        finite_mask = torch.isfinite(grad)
        finite = grad[finite_mask]
        group["nonfinite_values"] += int((~finite_mask).sum().item())
        group["grad_values"] += int(finite.numel())
        if finite.numel():
            group["l2"] += float(finite.pow(2).sum().item())
            group["abs_max"] = max(
                group["abs_max"], float(finite.abs().max().item()))
    for group in groups.values():
        squared_sum = group["l2"]
        group["l2"] = math.sqrt(squared_sum)
        group["rms"] = (
            math.sqrt(squared_sum / group["grad_values"])
            if group["grad_values"] else 0.0)
        group["all_finite"] = group["nonfinite_values"] == 0
    return groups


def register_output_gradients(outputs):
    registration = {}
    for stage_key in sorted(key for key in outputs
                            if key.startswith("stage")):
        stage = outputs[stage_key]
        seen = {}
        stage_registration = {}
        for key in AUDIT_TENSOR_KEYS:
            tensor = stage.get(key)
            if not torch.is_tensor(tensor):
                continue
            identity = id(tensor)
            alias_of = seen.get(identity)
            if alias_of is None:
                seen[identity] = key
                if tensor.requires_grad:
                    tensor.retain_grad()
            stage_registration[key] = {
                "alias_of": alias_of,
                "requires_grad": bool(tensor.requires_grad),
            }
        registration[stage_key] = stage_registration
    return registration


def output_gradient_stats(outputs, registration):
    report = {}
    for stage_key, registered in registration.items():
        report[stage_key] = {}
        for key, metadata in registered.items():
            tensor = outputs[stage_key][key]
            stats = gradient_stats(tensor.grad)
            stats.update(metadata)
            report[stage_key][key] = stats
    return report


def _masked_mean_or_none(value, mask):
    if value is None or not bool(mask.any().item()):
        return None
    return _safe_float(value[mask].float().mean().item())


def refined_direction_stats(depth_refined, depth_raw, raw_residual,
                            benefit_gate, depth_gt, valid_mask):
    mask = valid_mask > 0.5
    result = {
        "refined_descent_direction_accuracy": None,
        "gate_descent_beneficial_mean": None,
        "gate_descent_harmful_mean": None,
        "beneficial_pixel_count": 0,
        "harmful_pixel_count": 0,
    }
    if depth_refined.grad is not None:
        target = depth_gt - depth_refined.detach()
        descent = -depth_refined.grad.detach()
        direction_mask = mask & (target.abs() > 1e-6)
        if bool(direction_mask.any().item()):
            correct = torch.sign(target) == torch.sign(descent)
            result["refined_descent_direction_accuracy"] = _safe_float(
                correct[direction_mask].float().mean().item())
    if (benefit_gate is None or benefit_gate.grad is None
            or depth_raw is None or raw_residual is None):
        return result
    raw_error = (depth_raw.detach() - depth_gt).abs()
    proposal_error = (
        depth_raw.detach() + raw_residual.detach() - depth_gt).abs()
    beneficial = mask & (proposal_error < raw_error)
    harmful = mask & (proposal_error > raw_error)
    gate_descent = -benefit_gate.grad.detach()
    result["beneficial_pixel_count"] = int(beneficial.sum().item())
    result["harmful_pixel_count"] = int(harmful.sum().item())
    result["gate_descent_beneficial_mean"] = _masked_mean_or_none(
        gate_descent, beneficial)
    result["gate_descent_harmful_mean"] = _masked_mean_or_none(
        gate_descent, harmful)
    return result


def _json_value(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _safe_float(value.detach().float().item())
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _forward_metrics(stage, depth_gt, mask):
    valid = mask > 0.5
    metrics = {"valid_pixel_count": int(valid.sum().item())}
    if not bool(valid.any().item()):
        return metrics
    depth_raw = stage.get("depth_raw")
    depth_refined = stage.get("depth_refined", stage.get("depth"))
    if depth_raw is not None:
        raw_error = (depth_raw.detach() - depth_gt).abs()
        metrics["raw_abs_error"] = _masked_mean_or_none(raw_error, valid)
    else:
        raw_error = None
    if depth_refined is not None:
        refined_error = (depth_refined.detach() - depth_gt).abs()
        metrics["refined_abs_error"] = _masked_mean_or_none(
            refined_error, valid)
    else:
        refined_error = None
    if raw_error is not None and refined_error is not None:
        metrics["raw_to_refined_error_delta"] = _masked_mean_or_none(
            raw_error - refined_error, valid)
        metrics["refined_improved_pixel_ratio"] = _masked_mean_or_none(
            (refined_error < raw_error).float(), valid)
        metrics["refined_worsened_pixel_ratio"] = _masked_mean_or_none(
            (refined_error > raw_error).float(), valid)
    for key in ("raw_depth_residual", "depth_residual", "benefit_gate"):
        if key in stage:
            value = stage[key].detach()
            metric_name = key + "_abs_mean" if "residual" in key else key + "_mean"
            metric_value = value.abs() if "residual" in key else value
            metrics[metric_name] = _masked_mean_or_none(metric_value, valid)
    return metrics


def _optimizer_report(optimizer):
    return [
        {
            "index": index,
            "name": group.get("name", "unnamed"),
            "lr": float(group["lr"]),
            "parameter_values": int(sum(
                parameter.numel() for parameter in group["params"])),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]


def _critical_gradient_errors(case_name, parameter_gradients, loss_value):
    errors = []

    def group_l2(name):
        return parameter_gradients.get(name, {}).get("l2", 0.0)

    residual_l2 = group_l2("sger_residual_output")
    gate_l2 = group_l2("sger_gate_a") + group_l2("sger_gate_b")
    if case_name in ("full_forced", "refined_only",
                     "residual_target_only") and residual_l2 <= 0.0:
        errors.append("missing critical SGER residual output gradient")
    if case_name in ("full_forced", "gate_benefit_only") and gate_l2 <= 0.0:
        errors.append("missing critical SGER benefit gate gradient")
    if (case_name == "safe_refine_only" and loss_value > 0.0
            and residual_l2 <= 0.0):
        errors.append("active safe-refine loss has no residual output gradient")
    for name, stats in parameter_gradients.items():
        if name.startswith("sger_") or name == "normal_head":
            if stats["missing_optimizer_tensors"]:
                errors.append(
                    "{} has {} parameter tensors outside optimizer".format(
                        name, stats["missing_optimizer_tensors"]))
    return errors


def _write_json_atomic(path, report):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".gradient-audit-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def run_audit_suite(model, optimizer, cases, evaluate,
                    final_stage_key="stage3", output_json=None,
                    metadata=None):
    module = model.module if hasattr(model, "module") else model
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    final_stage_index = int(final_stage_key.replace("stage", "")) - 1
    report = {
        "metadata": _json_value(metadata or {}),
        "optimizer_groups": _optimizer_report(optimizer),
        "cases": [],
        "fatal_errors": [],
    }
    try:
        for case in cases:
            module.load_state_dict(copy.deepcopy(initial_state))
            optimizer.zero_grad()
            module.set_sger_residual_scale(case.sger_scale)
            outputs, loss, extras, depth_gt, valid_mask = evaluate(case)
            registration = register_output_gradients(outputs)
            case_errors = []
            case_warnings = []
            stage = outputs.get(final_stage_key)
            if stage is None:
                case_errors.append("missing {} outputs".format(final_stage_key))
            elif "depth_raw" not in stage or "depth_refined" not in stage:
                case_errors.append("missing critical SGER outputs")
            if not bool((valid_mask > 0.5).any().item()):
                case_errors.append("no valid depth pixels")
            if not bool(torch.isfinite(loss.detach()).all().item()):
                case_errors.append("non-finite total loss")
            if not case_errors:
                loss.backward()
            tensor_gradients = output_gradient_stats(outputs, registration)
            parameter_gradients = parameter_group_stats(
                module, optimizer, final_stage_index)
            nonfinite_groups = [
                name for name, stats in parameter_gradients.items()
                if not stats["all_finite"]]
            if nonfinite_groups:
                case_errors.append(
                    "non-finite parameter gradients: {}".format(
                        ", ".join(nonfinite_groups)))
            loss_value = float(loss.detach().item())
            case_errors.extend(_critical_gradient_errors(
                case.name, parameter_gradients, loss_value))
            directions = {}
            forward_metrics = {}
            if stage is not None:
                forward_metrics = _forward_metrics(stage, depth_gt, valid_mask)
                depth_refined = stage.get("depth_refined", stage.get("depth"))
                directions = refined_direction_stats(
                    depth_refined,
                    stage.get("depth_raw"),
                    stage.get("raw_depth_residual"),
                    stage.get("benefit_gate", stage.get("geometry_gate")),
                    depth_gt,
                    valid_mask,
                )
                if (case.name == "safe_refine_only"
                        and loss_value == 0.0):
                    case_warnings.append("safe-refine hinge is inactive")
            case_report = {
                "name": case.name,
                "loss": _safe_float(loss.detach().float().item()),
                "loss_weights": dict(case.loss_weights),
                "scales": {
                    "sger": case.sger_scale,
                    "residual_target": case.residual_target_scale,
                    "gate_benefit": case.gate_benefit_scale,
                },
                "loss_extras": _json_value(extras),
                "forward_metrics": forward_metrics,
                "tensor_gradients": tensor_gradients,
                "parameter_gradients": parameter_gradients,
                "direction_checks": directions,
                "warnings": case_warnings,
                "fatal_errors": case_errors,
            }
            report["cases"].append(case_report)
            report["fatal_errors"].extend(
                "{}: {}".format(case.name, message)
                for message in case_errors)
    finally:
        module.load_state_dict(initial_state)
        optimizer.zero_grad()
    if output_json is not None:
        _write_json_atomic(output_json, report)
    return report
