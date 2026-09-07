"""Verify Fig. 3 against the registered Full model and actual AMS source.

Run from the repository root:
  python output/pdf/ICSPS2026_Fig3/verify_workflow.py
Append --dependency-dir tmp/fig1_runtime when using the isolated runtime.

Two CPU forwards use random weights and synthetic inputs only. No checkpoint,
dataset, training loop, backward call, or optimizer is loaded/executed. To avoid
loading unrelated dataset dependencies or the AMS entry point, the four pure
AMS helper functions and its encoder-freeze loop are compiled directly from
their original AST nodes without changing their bodies. The freeze loop affects
only this disposable random model's requires_grad flags; no weights are updated.
Reported values are structural diagnostics, not experimental results.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-dir", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("workflow_trace.json"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root = next(p for p in Path(__file__).resolve().parents
                if (p / "model_registry.py").is_file())
    sys.path.insert(0, str(root))
    if args.dependency_dir is not None:
        sys.path.insert(0, str(args.dependency_dir.resolve()))

    import torch
    from model_registry import build_model
    from numeric_domain import (INTENSITY_DOMAIN, LOG_ALPHA,
                                log_transform_01_torch,
                                inverse_log_transform_01_torch)

    source_names = ["transform_main.py", "numeric_domain.py", "model_registry.py",
                    "ablation_config.py", "train_icsps2026_ams.py",
                    "evaluate_icsps2026_real.py", "configs/icsps2026_frozen_v2.json"]
    source_text = {name: (root / name).read_text(encoding="utf-8")
                   for name in source_names}
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in source_names}
    tree = ast.parse(source_text["train_icsps2026_ams.py"])
    helpers = ("stable_seed", "make_mask", "masked_l1", "total_variation")
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in helpers]
    assert len(selected) == len(helpers)
    scope = {"torch": torch, "hashlib": hashlib}
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 str(root / "train_icsps2026_ams.py"), "exec"), scope)
    stable_seed = scope["stable_seed"]
    make_mask = scope["make_mask"]
    masked_l1 = scope["masked_l1"]
    total_variation = scope["total_variation"]
    helper_sources = {node.name: {
        "file": "train_icsps2026_ams.py", "start_line": node.lineno,
        "end_line": node.end_lineno,
        "body_sha256": hashlib.sha256(ast.get_source_segment(
            source_text["train_icsps2026_ams.py"], node).encode("utf-8")).hexdigest(),
        "execution": "original unmodified AST node; full module/entry point not imported",
    } for node in selected}

    def normalized(node):
        return ast.dump(node, include_attributes=False)

    def statements_matching(tree_node, statement):
        target = normalized(ast.parse(statement).body[0])
        return [node for node in ast.walk(tree_node)
                if isinstance(node, ast.stmt) and normalized(node) == target]

    def record_source_check(name, tree_node, statement, expected_count=1):
        matches = statements_matching(tree_node, statement)
        source_checks[name] = len(matches) == expected_count
        source_evidence[name] = {
            "expected_occurrences": expected_count, "actual_occurrences": len(matches),
            "lines": [node.lineno for node in matches], "statement": statement,
        }

    source_checks = {}
    source_evidence = {}
    train_main = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "main")
    validation_node = next(node for node in tree.body
                           if isinstance(node, ast.FunctionDef)
                           and node.name == "fixed_mask_validation")
    record_source_check("AMS_training_input_is_zero_masked_before_entire_model", train_main,
                        "prediction = model(noisy * (1.0 - mask))")
    record_source_check("AMS_validation_input_uses_same_masking_rule", validation_node,
                        "prediction = model(noisy * (1.0 - mask))")
    loss_statement = "loss = masked_l1(prediction, noisy, mask) + lambda_tv * total_variation(prediction)"
    record_source_check("AMS_training_loss_uses_original_noisy_target_and_whole_prediction_TV",
                        train_main, loss_statement)
    record_source_check("AMS_validation_uses_same_objective", validation_node, loss_statement)
    record_source_check("training_mask_seed_contains_epoch_and_sample_id", train_main,
                        'mask_seed = stable_seed(PROTOCOL_ID, "ams-train-mask", args.seed, epoch, sample_id)')
    record_source_check("validation_mask_seed_excludes_epoch_and_is_per_sample", train_main,
                        'mask_seed = stable_seed(PROTOCOL_ID, "ams-validation-mask", args.seed, sample_id)')
    record_source_check("training_sets_model_to_train_mode", train_main, "model.train()")
    record_source_check("training_keeps_Tenc_in_eval_mode", train_main, "model.log_branch.Tenc.eval()")
    record_source_check("only_encoder_parameters_explicitly_frozen", train_main,
                        "for parameter in model.log_branch.Tenc.parameters():\n    parameter.requires_grad = False")
    grad_assignments = [node for node in ast.walk(train_main) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Attribute) and target.attr == "requires_grad"
                                for target in node.targets)]
    source_checks["no_other_requires_grad_assignment_in_AMS_main"] = len(grad_assignments) == 1
    record_source_check("optimizer_selects_only_trainable_parameters", train_main,
                        "optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate, weight_decay=weight_decay)")
    record_source_check("checkpoint_selection_minimizes_fixed_mask_validation_loss", train_main,
                        'is_best = fixed_val_loss < float(best["fixed_val_masked_loss"])')
    record_source_check("selection_initially_has_no_adapted_best_epoch", train_main,
                        'best = {"epoch": None, "fixed_val_masked_loss": math.inf}')
    # Compare the loop header directly: its body necessarily contains training
    # and must not be executed by this diagnostic.
    epoch_loops = [node for node in ast.walk(train_main) if isinstance(node, ast.For)
                   and isinstance(node.target, ast.Name) and node.target.id == "epoch"]
    source_checks["adapted_epochs_start_at_one_and_end_at_configured_epoch"] = (
        len(epoch_loops) == 1 and normalized(epoch_loops[0].iter)
        == normalized(ast.parse("range(start_epoch + 1, epochs + 1)", mode="eval").body))
    source_evidence["adapted_epochs_start_at_one_and_end_at_configured_epoch"] = {
        "lines": [node.lineno for node in epoch_loops],
        "expression": "range(start_epoch + 1, epochs + 1)",
        "loop_executed": False,
    }
    eval_tree = ast.parse(source_text["evaluate_icsps2026_real.py"])
    record_source_check("real_evaluation_calls_model_on_unmasked_tensor", eval_tree,
                        "prediction_tensor = model(tensor)")
    record_source_check("evaluation_tensor_is_directly_made_from_noisy_image", eval_tree,
                        "tensor = torch.from_numpy(noisy)[None, None].to(device)")
    for name in ("real_evaluation_calls_model_on_unmasked_tensor",
                 "evaluation_tensor_is_directly_made_from_noisy_image"):
        source_evidence[name]["file"] = "evaluate_icsps2026_real.py"
    for evidence in source_evidence.values():
        evidence.setdefault("file", "train_icsps2026_ams.py")

    config = json.loads(source_text["configs/icsps2026_frozen_v2.json"])
    config = config.get("effective_protocol", config)
    ams = config["ams"]
    config_checks = {
        "eight_adaptation_epochs": ams["epochs"] == 8,
        "AMS_learning_rate_1e_minus_6": ams["optimizer"]["learning_rate"] == 1e-6,
        "AMS_mask_is_Bernoulli_probability_0_2": ams["mask"]["distribution"] == "Bernoulli" and ams["mask"]["ratio"] == 0.2,
        "AMS_TV_weight_1e_minus_3": ams["loss"]["lambda_tv"] == 1e-3,
        "AMS_batch_one_and_256_square": ams["batch_size"] == 1 and ams["crop_height"] == ams["crop_width"] == 256,
        "config_encoder_is_only_frozen_module": ams["frozen_modules"] == ["model.log_branch.Tenc"],
        "config_adapted_checkpoint_candidates_1_to_8": ams["checkpoint_selection"]["candidate_epochs"] == "adapted_epochs_1_to_8",
        "config_inference_has_no_mask": ams["inference_mask"] == "none",
    }

    torch.manual_seed(2026)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model = build_model("ours", variant="full", numeric_domain=INTENSITY_DOMAIN).cpu().eval()
    freeze_loop = statements_matching(train_main,
        "for parameter in model.log_branch.Tenc.parameters():\n    parameter.requires_grad = False")
    assert len(freeze_loop) == 1
    exec(compile(ast.Module(body=freeze_loop, type_ignores=[]),
                 str(root / "train_icsps2026_ams.py"), "exec"), {"model": model})
    parameter_groups = {}
    for group_name, group in {
        "encoder": model.log_branch.Tenc,
        "guidance_estimator": model.log_branch.noise_estimator,
        "bottleneck": model.log_branch.bottleneck_refine,
        "decoder_including_guidance_gates": model.log_branch.convproj,
        "reconstruction_head": model.log_branch.clean,
        "intensity_compensator": model.dual_fusion,
    }.items():
        params = list(group.parameters())
        parameter_groups[group_name] = {
            "parameter_tensors": len(params), "parameter_scalars": sum(p.numel() for p in params),
            "trainable_tensors": sum(p.requires_grad for p in params),
            "trainable_scalars": sum(p.numel() for p in params if p.requires_grad),
        }
    freeze_checks = {
        "encoder_parameters_are_all_frozen": all(not p.requires_grad for p in model.log_branch.Tenc.parameters()),
        "every_parameter_outside_encoder_remains_trainable": all(
            p.requires_grad for name, p in model.named_parameters() if not name.startswith("log_branch.Tenc.")),
        "compensator_gamma_is_trainable": model.dual_fusion.gamma.requires_grad,
        "compensator_gamma_initial_value_is_0_1": bool(torch.allclose(model.dual_fusion.gamma.detach(), torch.tensor([0.1]))),
        "verification_keeps_entire_model_in_eval_mode": all(not module.training for module in model.modules()),
    }

    def close(a, b):
        return bool(torch.allclose(a, b, rtol=1e-5, atol=1e-6))

    def desc(tensor):
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "finite": bool(torch.isfinite(tensor).all()),
                "min": float(tensor.min()), "max": float(tensor.max())}

    linear = torch.linspace(0, 1, 4097)
    with torch.inference_mode():
        logged = log_transform_01_torch(linear)
        roundtrip = inverse_log_transform_01_torch(logged)
        clipped_test = torch.tensor([-0.2, 0.0, 0.5, 1.0, 1.2])
        clipping_roundtrip = inverse_log_transform_01_torch(log_transform_01_torch(clipped_test))
    representation_checks = {
        "formal_domain_is_intensity_v1": model.numeric_domain == "intensity_v1",
        "formal_full_model_uses_log_representation": model.representation == "log",
        "log_alpha_is_ten": LOG_ALPHA == 10.0,
        "normalized_log_and_inverse_roundtrip": close(roundtrip, linear),
        "log_maps_zero_and_one_to_zero_and_one": close(logged[[0, -1]], torch.tensor([0.0, 1.0])),
        "log_transform_is_monotone_on_normalized_input": bool((logged.diff() >= 0).all()),
        "transform_clamps_out_of_range_input": close(clipping_roundtrip, clipped_test.clamp(0, 1)),
        "branch_reconstruction_activation_is_sigmoid": isinstance(model.log_branch.active, torch.nn.Sigmoid),
    }

    shape = (1, 1, 256, 256)
    synthetic = torch.linspace(0.001, 0.999, 256 * 256).reshape(shape)
    val_seed = stable_seed(config["protocol_id"], "ams-validation-mask", 42, "diagnostic-only")
    epoch_one_seed = stable_seed(config["protocol_id"], "ams-train-mask", 42, 1, "diagnostic-only")
    epoch_two_seed = stable_seed(config["protocol_id"], "ams-train-mask", 42, 2, "diagnostic-only")
    val_mask = make_mask(shape, ams["mask"]["ratio"], val_seed)
    repeated_val_mask = make_mask(shape, ams["mask"]["ratio"], val_seed)
    train_mask = make_mask(shape, ams["mask"]["ratio"], epoch_one_seed)
    next_train_mask = make_mask(shape, ams["mask"]["ratio"], epoch_two_seed)
    masked_input = synthetic * (1 - train_mask)
    mask_checks = {
        "mask_matches_input_shape": tuple(train_mask.shape) == shape,
        "mask_is_binary": bool(((train_mask == 0) | (train_mask == 1)).all()),
        "fixed_validation_seed_reproduces_identical_mask": bool(torch.equal(val_mask, repeated_val_mask)),
        "training_seed_changes_with_epoch": epoch_one_seed != epoch_two_seed,
        "training_mask_changes_with_epoch": not bool(torch.equal(train_mask, next_train_mask)),
        "masked_positions_are_zero_at_entire_model_input": bool((masked_input[train_mask == 1] == 0).all()),
        "unmasked_positions_retain_original_noisy_values": bool(torch.equal(masked_input[train_mask == 0], synthetic[train_mask == 0])),
    }

    cases = []
    for case_name, model_input in (("unmasked_inference", synthetic), ("masked_AMS_input", masked_input)):
        captured = {}
        handles = []

        def hook(name):
            def capture(_module, inputs, output):
                captured[name] = {"inputs": tuple(t.detach() for t in inputs), "output": output.detach()}
            return capture

        for name in ("log_branch", "log_branch.clean", "log_branch.active", "dual_fusion", "dual_fusion.fusion"):
            handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(model_input)
        elapsed = time.perf_counter() - started
        for handle in handles:
            handle.remove()
        branch_out = captured["log_branch"]["output"]
        base = inverse_log_transform_01_torch(branch_out)
        residual = captured["dual_fusion.fusion"]["output"]
        preclip = captured["dual_fusion"]["output"]
        fusion_in = captured["dual_fusion.fusion"]["inputs"][0]
        expected_preclip = base + model.dual_fusion.gamma.detach() * residual
        case_checks = {
            "output_shape_is_B_1_H_W": tuple(output.shape) == shape,
            "model_output_is_finite": bool(torch.isfinite(output).all()),
            "final_output_in_unit_interval": bool(((output >= 0) & (output <= 1)).all()),
            "branch_receives_log_of_bounded_entire_input": close(captured["log_branch"]["inputs"][0], log_transform_01_torch(model_input.clamp(0, 1))),
            "head_input_has_eight_channels": list(captured["log_branch.clean"]["inputs"][0].shape) == [1, 8, 256, 256],
            "head_output_has_one_channel": tuple(captured["log_branch.clean"]["output"].shape) == shape,
            "head_sigmoid_receives_reconstruction_logits": close(captured["log_branch.active"]["inputs"][0], captured["log_branch.clean"]["output"]),
            "head_sigmoid_matches_actual_branch_output": close(branch_out, captured["log_branch.clean"]["output"].sigmoid()),
            "normalized_log_estimate_is_in_unit_interval": bool(((branch_out >= 0) & (branch_out <= 1)).all()),
            "compensator_first_argument_is_bounded_actual_input": close(captured["dual_fusion"]["inputs"][0], model_input.clamp(0, 1)),
            "compensator_second_argument_is_inverse_log_intensity": close(captured["dual_fusion"]["inputs"][1], base),
            "compensator_concatenates_actual_input_then_inverse_log_result": close(fusion_in, torch.cat([model_input.clamp(0, 1), base], dim=1)),
            "compensator_input_has_two_intensity_channels": list(fusion_in.shape) == [1, 2, 256, 256],
            "compensator_residual_has_one_channel": tuple(residual.shape) == shape,
            "compensator_adds_gamma_scaled_residual_to_inverse_log_base": close(preclip, expected_preclip),
            "final_model_applies_clip_after_residual_addition": close(output, preclip.clamp(0, 1)),
        }
        if case_name == "masked_AMS_input":
            case_checks["compensator_bypass_also_zero_at_masked_pixels"] = bool((fusion_in[:, :1][train_mask == 1] == 0).all())
            case_checks["original_noisy_target_is_not_fed_to_compensator"] = not close(fusion_in[:, :1], synthetic)
        cases.append({"name": case_name, "input": desc(model_input), "output": desc(output),
                      "observed_module_io": {name: {"inputs": [desc(t) for t in pair["inputs"]], "output": desc(pair["output"])} for name, pair in captured.items()},
                      "checks": case_checks, "forward_seconds": elapsed,
                      "preclip_reconstruction_max_abs_error": float((preclip - expected_preclip).abs().max()),
                      "clip_reconstruction_max_abs_error": float((output - preclip.clamp(0, 1)).abs().max())})

    # Tiny hand-checkable tensors demonstrate the support of the original loss
    # functions. Their numeric outputs are diagnostics, never paper results.
    target = torch.zeros((1, 1, 2, 2))
    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    support = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    changed_off_support = prediction + (1 - support) * 100
    masked_value = masked_l1(prediction, target, support)
    tv_value = total_variation(prediction)
    loss_checks = {
        "masked_L1_uses_mask_one_pixels_and_normalizes_by_mask_count": close(masked_value, torch.tensor(2.5)),
        "changing_prediction_only_outside_mask_does_not_change_masked_L1": close(masked_value, masked_l1(changed_off_support, target, support)),
        "empty_mask_yields_zero_finite_reconstruction_loss": close(masked_l1(prediction, target, torch.zeros_like(support)), torch.tensor(0.0)),
        "TV_is_sum_of_vertical_and_horizontal_absolute_difference_means": close(tv_value, torch.tensor(3.0)),
        "TV_depends_on_pixels_outside_reconstruction_mask": not close(tv_value, total_variation(changed_off_support)),
    }
    fusion_layers = list(model.dual_fusion.fusion)
    compensation_checks = {
        "compensator_is_three_3x3_convolutions_with_two_GELUs": len(fusion_layers) == 5 and all(isinstance(fusion_layers[i], torch.nn.Conv2d) and fusion_layers[i].kernel_size == (3, 3) and fusion_layers[i].stride == (1, 1) and fusion_layers[i].padding == (1, 1) for i in (0, 2, 4)) and all(isinstance(fusion_layers[i], torch.nn.GELU) for i in (1, 3)),
        "compensator_channel_sequence_is_2_32_32_1": [(fusion_layers[i].in_channels, fusion_layers[i].out_channels) for i in (0, 2, 4)] == [(2, 32), (32, 32), (32, 1)],
        "compensator_gamma_is_one_shared_scalar": model.dual_fusion.gamma.numel() == 1,
        "no_parameter_gradients_computed": all(p.grad is None for p in model.parameters()),
        "all_source_files_remain_unchanged": all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in source_hashes.items()),
    }
    groups = {"source_static_checks": source_checks, "frozen_config_checks": config_checks,
              "parameter_freeze_checks": freeze_checks, "representation_checks": representation_checks,
              "mask_checks": mask_checks, "loss_checks": loss_checks,
              "compensator_and_read_only_checks": compensation_checks}
    all_checks = [value for group in groups.values() for value in group.values()]
    all_checks += [value for case in cases for value in case["checks"].values()]
    result = {
        "purpose": "Fig. 3 structural and workflow diagnostics only; random weights and synthetic inputs; not experimental results.",
        "checkpoint_loaded": False, "dataset_loaded": False, "training_performed": False,
        "backward_performed": False, "optimizer_created": False, "optimizer_step_performed": False,
        "model_source_modified": False, "device": "cpu", "inference_mode": True,
        "entry_point": 'model_registry.build_model("ours", variant="full", numeric_domain="intensity_v1")',
        "model_class": type(model).__name__, "registry_metadata": model.registry_metadata,
        "variant_config": model.variant_config, "synthetic_seed": 2026,
        "python_version": platform.python_version(),
        "package_versions": {package: importlib.metadata.version(package) for package in ("torch", "torchvision", "timm", "numpy")},
        "threads": args.threads, "model_inference_count": 2,
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "sources_sha256": source_hashes,
        "AMS_function_extraction": helper_sources,
        "freeze_execution": {"method": "original AST loop on disposable random model; requires_grad flags only", "line": freeze_loop[0].lineno},
        "source_evidence": source_evidence, "ams_configuration": ams,
        "parameter_groups_after_source_freeze_loop": parameter_groups,
        "log_inverse_roundtrip_max_abs_error": float((roundtrip - linear).abs().max()),
        "compensator_gamma_initial_value": float(model.dual_fusion.gamma.detach().item()),
        "diagnostic_mask_realized_fraction_not_experiment": float(train_mask.mean()),
        "diagnostic_loss_values_not_experiment": {"masked_L1": float(masked_value), "whole_image_TV": float(tv_value)},
        "check_groups": groups, "forward_cases": cases,
        "check_count": len(all_checks), "passed_count": sum(all_checks),
        "all_checks_passed": all(all_checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "check_count": len(all_checks),
                      "passed_count": sum(all_checks), "all_checks_passed": result["all_checks_passed"],
                      "model_forward_seconds": [round(case["forward_seconds"], 3) for case in cases],
                      "failed_checks": [name for group in groups.values() for name, ok in group.items() if not ok]
                      + [case["name"] + ": " + name for case in cases for name, ok in case["checks"].items() if not ok]}, indent=2))
    if not result["all_checks_passed"]:
        raise AssertionError("Fig. 3 workflow verification failed; inspect workflow_trace.json")


if __name__ == "__main__":
    main()
