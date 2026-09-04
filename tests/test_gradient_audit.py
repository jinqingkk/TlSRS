import json
import math
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim

from gradient_audit_utils import (
    AUDIT_TENSOR_KEYS,
    build_audit_cases,
    classify_parameter,
    gradient_stats,
    parameter_group_stats,
    refined_direction_stats,
    register_output_gradients,
    run_audit_suite,
)
from gradient_audit import build_parser


def test_builds_six_cases_with_expected_isolation():
    base = {
        "raw_depth_loss_weight": 1.0,
        "refined_depth_loss_weight": 1.0,
        "residual_loss_weight": 0.01,
        "gate_loss_weight": 0.0,
        "safe_refine_loss_weight": 0.1,
        "residual_target_loss_weight": 0.05,
        "gate_benefit_loss_weight": 0.05,
        "normal_smooth_loss_weight": 0.02,
        "curv_loss_weight": 0.005,
        "edge_smooth_loss_weight": 0.005,
        "depth_normal_loss_weight": 0.03,
    }
    cases = build_audit_cases(
        epoch_idx=8,
        base_weights=base,
        warmup_start_epoch=7,
        warmup_end_epoch=10,
    )

    assert [case.name for case in cases] == [
        "full_schedule",
        "full_forced",
        "refined_only",
        "residual_target_only",
        "gate_benefit_only",
        "safe_refine_only",
    ]
    assert math.isclose(cases[0].sger_scale, 0.5)
    assert cases[0].residual_target_scale == 1.0
    assert math.isclose(cases[0].gate_benefit_scale, 0.5)
    assert cases[1].sger_scale == 1.0
    assert cases[1].residual_target_scale == 1.0
    assert cases[1].gate_benefit_scale == 1.0

    isolated = {case.name: case for case in cases[2:]}
    expected_nonzero = {
        "refined_only": "refined_depth_loss_weight",
        "residual_target_only": "residual_target_loss_weight",
        "gate_benefit_only": "gate_benefit_loss_weight",
        "safe_refine_only": "safe_refine_loss_weight",
    }
    for name, case in isolated.items():
        for key, value in case.loss_weights.items():
            if key == expected_nonzero[name]:
                assert value == 1.0
            elif key != "raw_depth_loss_weight":
                assert value == 0.0
        assert case.loss_weights["raw_depth_loss_weight"] == 0.0


def test_gradient_stats_handle_missing_zero_and_nonfinite():
    missing = gradient_stats(None)
    assert missing["present"] is False
    assert missing["numel"] == 0

    zero = gradient_stats(torch.zeros(4))
    assert zero["present"] is True
    assert zero["all_finite"] is True
    assert zero["zero_ratio"] == 1.0
    assert zero["l2"] == 0.0

    mixed = gradient_stats(torch.tensor([-2.0, 0.0, 1.0, 2.0]))
    assert mixed["finite_count"] == 4
    assert mixed["zero_ratio"] == 0.25
    assert math.isclose(mixed["abs_mean"], 1.25)
    assert math.isclose(mixed["abs_max"], 2.0)
    assert math.isclose(mixed["l2"], 3.0)
    assert math.isclose(mixed["rms"], 1.5)

    nonfinite = gradient_stats(torch.tensor([1.0, float("nan")]))
    assert nonfinite["present"] is True
    assert nonfinite["all_finite"] is False
    assert nonfinite["finite_count"] == 1
    json.dumps(nonfinite, allow_nan=False)


def test_classifies_real_sger_parameter_paths():
    expected = {
        "normal_head.2.conv1.conv.weight": "normal_head",
        "sger_blocks.0.feature_projection.weight": "sger_feature_projection",
        "sger_blocks.0.residual.body.0.weight": "sger_residual_body",
        "sger_blocks.0.residual.output.weight": "sger_residual_output",
        "sger_blocks.0.gate.gate_a.0.weight": "sger_gate_a",
        "sger_blocks.0.gate.gate_b.0.weight": "sger_gate_b",
        "feature.conv0.0.conv.weight": "backbone_feature",
        "cost_regularization.0.conv0.conv.weight": "cost_regularization",
    }
    for name, group in expected.items():
        assert classify_parameter(name, final_stage_index=2) == group


def test_parameter_stats_report_optimizer_coverage():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    optimizer = optim.SGD(model[0].parameters(), lr=0.1)
    model(torch.ones(1, 2)).sum().backward()

    stats = parameter_group_stats(model, optimizer, final_stage_index=2)

    assert stats["other"]["parameter_tensors"] == 4
    assert stats["other"]["missing_optimizer_tensors"] == 2
    assert stats["other"]["missing_grad_tensors"] == 0
    json.dumps(stats, allow_nan=False)


def test_retains_output_gradients_once_and_records_aliases():
    source = torch.ones(1, 2, 2, requires_grad=True)
    gate = source * 2.0
    stage = {
        "depth_refined": source * 3.0,
        "benefit_gate": gate,
        "geometry_gate": gate,
    }
    outputs = {"stage3": stage}

    registration = register_output_gradients(outputs)
    stage["depth_refined"].sum().backward(retain_graph=True)

    assert set(AUDIT_TENSOR_KEYS).issuperset(stage.keys())
    assert registration["stage3"]["geometry_gate"]["alias_of"] == (
        "benefit_gate")
    assert registration["stage3"]["benefit_gate"]["alias_of"] is None
    assert stage["depth_refined"].grad is not None


def test_direction_stats_use_negative_gradient():
    depth_gt = torch.tensor([[[2.0, 0.0]]])
    depth_refined = torch.tensor([[[1.0, 1.0]]], requires_grad=True)
    depth_refined.grad = torch.tensor([[[-1.0, 1.0]]])
    depth_raw = torch.tensor([[[1.0, 1.0]]])
    raw_residual = torch.tensor([[[0.5, 1.0]]])
    gate = torch.tensor([[[0.5, 0.5]]], requires_grad=True)
    gate.grad = torch.tensor([[[-0.2, 0.3]]])
    mask = torch.ones_like(depth_gt, dtype=torch.bool)

    stats = refined_direction_stats(
        depth_refined, depth_raw, raw_residual, gate, depth_gt, mask)

    assert stats["refined_descent_direction_accuracy"] == 1.0
    assert math.isclose(
        stats["gate_descent_beneficial_mean"], 0.2, abs_tol=1e-6)
    assert math.isclose(
        stats["gate_descent_harmful_mean"], -0.3, abs_tol=1e-6)


class TinyAuditModel(nn.Module):
    def __init__(self):
        super(TinyAuditModel, self).__init__()
        self.sger_residual_scale = 1.0
        self.sger_blocks = nn.ModuleList([nn.Linear(1, 1, bias=False)])
        self.register_buffer("forward_count", torch.zeros(1))

    def set_sger_residual_scale(self, scale):
        self.sger_residual_scale = float(scale)

    def forward(self, value):
        self.forward_count += 1
        raw = value.detach()
        proposal = self.sger_blocks[0](value.unsqueeze(-1)).squeeze(-1)
        gate = torch.sigmoid(proposal)
        residual = self.sger_residual_scale * gate * proposal
        refined = raw + residual
        return {
            "stage3": {
                "depth_raw": raw,
                "raw_depth_residual": proposal,
                "benefit_gate": gate,
                "geometry_gate": gate,
                "depth_residual": residual,
                "depth_refined": refined,
                "depth": refined,
            }
        }


def test_suite_restores_state_and_never_updates_parameters():
    model = TinyAuditModel()
    with torch.no_grad():
        model.sger_blocks[0].weight.fill_(0.25)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    initial_weight = model.sger_blocks[0].weight.detach().clone()
    initial_buffer = model.forward_count.detach().clone()
    batch = {
        "value": torch.ones(1, 1),
        "depth_gt": torch.full((1, 1), 2.0),
        "mask": torch.ones(1, 1, dtype=torch.bool),
    }
    cases = build_audit_cases(8, {
        "raw_depth_loss_weight": 1.0,
        "refined_depth_loss_weight": 1.0,
        "residual_loss_weight": 0.01,
        "gate_loss_weight": 0.0,
        "safe_refine_loss_weight": 0.1,
        "residual_target_loss_weight": 0.05,
        "gate_benefit_loss_weight": 0.05,
        "normal_smooth_loss_weight": 0.02,
        "curv_loss_weight": 0.005,
        "edge_smooth_loss_weight": 0.005,
        "depth_normal_loss_weight": 0.03,
    }, 7, 10)[:2]

    starts = []

    def evaluate(case):
        starts.append(float(model.forward_count.item()))
        outputs = model(batch["value"])
        stage = outputs["stage3"]
        loss = (stage["depth_refined"] - batch["depth_gt"]).pow(2).mean()
        return outputs, loss, {}, batch["depth_gt"], batch["mask"]

    with tempfile.NamedTemporaryFile(suffix=".json") as report_file:
        report = run_audit_suite(
            model, optimizer, cases, evaluate,
            final_stage_key="stage3",
            output_json=report_file.name,
        )
        with open(report_file.name, "r") as handle:
            disk_report = json.load(handle)

    assert starts == [0.0, 0.0]
    assert torch.equal(model.sger_blocks[0].weight, initial_weight)
    assert torch.equal(model.forward_count, initial_buffer)
    assert len(report["cases"]) == 2
    assert len(disk_report["cases"]) == 2


def test_isolated_case_reports_missing_critical_gradient():
    model = TinyAuditModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    case = build_audit_cases(8, {
        "refined_depth_loss_weight": 1.0,
    }, 7, 10)[2]
    batch = {
        "value": torch.zeros(1, 1),
        "depth_gt": torch.ones(1, 1),
        "mask": torch.ones(1, 1, dtype=torch.bool),
    }

    def evaluate(audit_case):
        outputs = model(batch["value"])
        loss = outputs["stage3"]["depth_refined"].pow(2).mean()
        return outputs, loss, {}, batch["depth_gt"], batch["mask"]

    report = run_audit_suite(model, optimizer, [case], evaluate)

    assert any("residual output" in error
               for error in report["fatal_errors"])


def test_cli_parser_requires_audit_inputs_and_defaults_to_sger_lite():
    parser = build_parser()
    args = parser.parse_args([
        "--trainpath", "/data/dtu",
        "--trainlist", "lists/dtu/train.txt",
        "--loadckpt", "checkpoints/model.ckpt",
        "--audit_epoch", "8",
        "--output_json", "outputs/audit.json",
    ])

    assert args.batch_size == 1
    assert args.num_workers == 0
    assert args.use_sger_lite is True
    assert args.audit_epoch == 8
    assert args.output_json == "outputs/audit.json"


def run_all_tests():
    test_builds_six_cases_with_expected_isolation()
    test_gradient_stats_handle_missing_zero_and_nonfinite()
    test_classifies_real_sger_parameter_paths()
    test_parameter_stats_report_optimizer_coverage()
    test_retains_output_gradients_once_and_records_aliases()
    test_direction_stats_use_negative_gradient()
    test_suite_restores_state_and_never_updates_parameters()
    test_isolated_case_reports_missing_critical_gradient()
    test_cli_parser_requires_audit_inputs_and_defaults_to_sger_lite()
    print("Gradient audit unit tests: PASS")


if __name__ == "__main__":
    run_all_tests()
