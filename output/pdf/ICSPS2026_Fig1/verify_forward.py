"""Verify Fig. 1 against the registered Full model using one CPU forward pass.

This is a shape/connectivity check with random weights and synthetic input. It
does not load checkpoints, train the model, or produce an experimental result.

Example (from the repository root)::

    python output/pdf/ICSPS2026_Fig1/verify_forward.py

If dependencies were installed in an isolated directory, add
``--dependency-dir tmp/fig1_runtime``. Only the requested JSON is written.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().with_name("forward_trace.json"),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root = next(p for p in Path(__file__).resolve().parents
                if (p / "model_registry.py").is_file())
    sys.path.insert(0, str(root))
    if args.dependency_dir is not None:
        sys.path.insert(0, str(args.dependency_dir.resolve()))

    import torch
    from model_registry import build_model
    from numeric_domain import (
        INTENSITY_DOMAIN, LOG_ALPHA, inverse_log_transform_01_torch,
        log_transform_01_torch,
    )
    from transform_main import FFTRefineBlock

    torch.manual_seed(2026)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model = build_model("ours", variant="full", numeric_domain=INTENSITY_DOMAIN)
    model.cpu().eval()
    branch = model.log_branch
    decoder = branch.convproj
    records: list[dict] = []
    snapshots: dict = {}
    assertions: dict[str, bool] = {}
    handles = []

    def describe(value):
        if isinstance(value, torch.Tensor):
            return {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "finite": bool(torch.isfinite(value).all()),
                "min": float(value.min()), "max": float(value.max()),
            }
        if isinstance(value, (list, tuple)):
            return [describe(v) for v in value]
        return value

    def register(name, module):
        def hook(_module, inputs, output):
            entry = {
                "event": len(records) + 1, "module": name,
                "class": type(_module).__name__,
                "input": describe(inputs), "output": describe(output),
            }
            records.append(entry)
            snapshots[name] = entry
        handles.append(module.register_forward_hook(hook))

    for i in range(1, 6):
        register(f"log_branch.Tenc.norm{i}", getattr(branch.Tenc, f"norm{i}"))
    register("log_branch.Tenc", branch.Tenc)
    register("log_branch.noise_estimator", branch.noise_estimator)
    register("log_branch.bottleneck_refine.local", branch.bottleneck_refine.local)
    register("log_branch.bottleneck_refine.freq.freq", branch.bottleneck_refine.freq.freq)
    register("log_branch.bottleneck_refine.freq", branch.bottleneck_refine.freq)
    register("log_branch.bottleneck_refine.fuse", branch.bottleneck_refine.fuse)
    register("log_branch.bottleneck_refine", branch.bottleneck_refine)
    for scale, level in [(32, 4), (16, 3), (8, 2), (4, 1), (2, 0)]:
        register(f"log_branch.convproj.convd{scale}x", getattr(decoder, f"convd{scale}x"))
        if level:
            register(f"log_branch.convproj.fuse{level}", getattr(decoder, f"fuse{level}"))
        register(f"log_branch.convproj.freq{level}.freq", getattr(decoder, f"freq{level}").freq)
        for prefix in ("freq", "refine", "ng"):
            register(f"log_branch.convproj.{prefix}{level}", getattr(decoder, f"{prefix}{level}"))
        register(f"log_branch.convproj.ng{level}.gate", getattr(decoder, f"ng{level}").gate)
    register("log_branch.convproj.convd1x", decoder.convd1x)
    register("log_branch.convproj", decoder)
    register("log_branch.clean", branch.clean)
    register("log_branch.active", branch.active)
    register("log_branch", branch)
    register("dual_fusion.fusion", model.dual_fusion.fusion)
    register("dual_fusion", model.dual_fusion)
    register("model", model)

    x = torch.linspace(0.0, 1.0, 256 * 256, dtype=torch.float32).reshape(1, 1, 256, 256)
    expected_log_input = log_transform_01_torch(x, alpha=LOG_ALPHA)

    def check_log_input(name):
        def hook(_module, inputs):
            assertions[name] = bool(torch.equal(inputs[0], expected_log_input))
        return hook

    handles.append(branch.Tenc.register_forward_pre_hook(check_log_input("encoder_receives_log_input")))
    handles.append(branch.noise_estimator.register_forward_pre_hook(check_log_input("guidance_receives_log_input")))
    branch_output = {}

    def capture_branch(_module, _inputs, output):
        branch_output["value"] = output.detach()

    def check_compensation_inputs(_module, inputs):
        assertions["compensator_receives_original_intensity"] = bool(torch.equal(inputs[0], x))
        expected = inverse_log_transform_01_torch(branch_output["value"], alpha=LOG_ALPHA)
        assertions["compensator_receives_inverse_log_reconstruction"] = bool(torch.equal(inputs[1], expected))

    handles.append(branch.register_forward_hook(capture_branch))
    handles.append(model.dual_fusion.register_forward_pre_hook(check_compensation_inputs))
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(x)
    elapsed = time.perf_counter() - start
    for handle in handles:
        handle.remove()

    def shape(name):
        return snapshots[name]["output"]["shape"]

    enc_shapes = [[1, c, s, s] for c, s in [(32, 128), (64, 64), (128, 32), (320, 16), (512, 8)]]
    assertions["encoder_five_stage_shapes"] = [d["shape"] for d in snapshots["log_branch.Tenc"]["output"]] == enc_shapes
    assertions["bottleneck_shape"] = shape("log_branch.bottleneck_refine") == [1, 512, 8, 8]
    assertions["guidance_shape"] = shape("log_branch.noise_estimator") == [1, 1, 256, 256]
    for scale, level, channels, side in [(32, 4, 320, 16), (16, 3, 128, 32), (8, 2, 64, 64), (4, 1, 32, 128), (2, 0, 16, 256)]:
        expected = [1, channels, side, side]
        names = [f"convd{scale}x", f"freq{level}", f"refine{level}", f"ng{level}"]
        if level:
            names.append(f"fuse{level}")
            assertions[f"decoder_stage_{level}_concatenation"] = snapshots[f"log_branch.convproj.fuse{level}"]["input"][0]["shape"] == [1, 2 * channels, side, side]
        assertions[f"decoder_stage_{level}_shapes"] = all(shape(f"log_branch.convproj.{name}") == expected for name in names)
        assertions[f"decoder_stage_{level}_resized_guidance"] = snapshots[f"log_branch.convproj.ng{level}.gate"]["input"][0]["shape"] == [1, channels + 1, side, side]
    assertions["tail_shape"] = shape("log_branch.convproj.convd1x") == [1, 8, 256, 256]
    assertions["reconstruction_head_shape"] = shape("log_branch.clean") == [1, 1, 256, 256]
    assertions["reconstruction_head_sigmoid"] = isinstance(branch.active, torch.nn.Sigmoid)
    assertions["compensator_concatenation_shape"] = snapshots["dual_fusion.fusion"]["input"][0]["shape"] == [1, 2, 256, 256]
    assertions["output_shape"] = list(output.shape) == [1, 1, 256, 256]
    assertions["output_finite_and_in_unit_interval"] = bool(torch.isfinite(output).all() and (output >= 0).all() and (output <= 1).all())
    fdr_names = [name for name, module in model.named_modules() if isinstance(module, FFTRefineBlock)]
    assertions["six_fdr_instances"] = len(fdr_names) == 6
    assertions["each_fdr_executed_once"] = all(sum(r["module"] == name for r in records) == 1 for name in fdr_names)

    # Compare definitions only. No baseline inference and no claim that newly
    # initialized weights equal a historical or trained baseline checkpoint.
    baseline = build_model("transsar_v2").cpu().eval()
    inherited = {"encoder_same_module_definition": repr(branch.Tenc) == repr(baseline.Tenc)}
    for scale in (32, 16, 8, 4, 2, 1):
        inherited[f"convd{scale}x_same_module_definition"] = repr(getattr(decoder, f"convd{scale}x")) == repr(getattr(baseline.convproj, f"convd{scale}x"))
    for level in range(5):
        inherited[f"refine{level}_same_residual_definition_as_dense_{level + 1}"] = repr(getattr(decoder, f"refine{level}")) == repr(getattr(baseline.convproj, f"dense_{level + 1}")[0])
    inherited["clean_same_module_definition"] = repr(branch.clean) == repr(baseline.clean)
    inherited["note"] = "Definition equivalence only: residual blocks are repositioned; additive skip merging is replaced by concatenation/fusion, and the formal branch uses sigmoid instead of baseline tanh."
    assertions["inherited_operator_definition_comparisons"] = all(value for value in inherited.values() if isinstance(value, bool))

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    versions = {}
    for name in ("torch", "torchvision", "timm", "numpy", "huggingface-hub", "safetensors", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    result = {
        "purpose": "Fig. 1 structural verification only; random weights and synthetic input; not an experiment or restoration result.",
        "checkpoint_loaded": False, "training_performed": False,
        "entry_point": 'model_registry.build_model("ours", variant="full", numeric_domain="intensity_v1")',
        "model_class": type(model).__name__, "registry_metadata": model.registry_metadata,
        "variant_config": model.variant_config, "log_alpha": LOG_ALPHA,
        "device": "cpu", "eval_mode": not model.training, "inference_mode": True,
        "seed_for_structure_check_only": 2026, "threads": args.threads,
        "python": platform.python_version(), "package_versions": versions,
        "input_definition": "torch.linspace(0, 1, 256 * 256, dtype=float32).reshape(1, 1, 256, 256)",
        "input": describe(x), "output": describe(output),
        "forward_seconds_this_machine": elapsed,
        "fdr_modules": fdr_names, "checks": assertions,
        "inherited_definition_comparisons": inherited,
        "sources_sha256": {str(p.relative_to(root)).replace('\\', '/'): sha256(p) for p in [root / "model_registry.py", root / "ablation_config.py", root / "numeric_domain.py", root / "transform_main.py", root / "arch/trans_basenetworks.py"]},
        "forward_completion_events": records,
        "all_checks_passed": all(assertions.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "all_checks_passed": result["all_checks_passed"], "checks": len(assertions), "fdr_instances": len(fdr_names), "encoder_shapes": enc_shapes, "model_output": describe(output), "forward_seconds": round(elapsed, 3)}, indent=2))
    if not result["all_checks_passed"]:
        raise AssertionError([name for name, passed in assertions.items() if not passed])


if __name__ == "__main__":
    main()
