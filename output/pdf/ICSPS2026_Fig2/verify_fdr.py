"""Read-only verification of the FDR diagram against the registered Full model.

Run from the repository root:
    python output/pdf/ICSPS2026_Fig2/verify_fdr.py
For a separate dependency directory, append --dependency-dir tmp/fig1_runtime.

One CPU inference of the real registered model is observed with module hooks.
Its six FFTRefineBlock paths are then independently reconstructed and compared
to those observations. One small odd-width block checks inverse-FFT sizing.
Random weights and synthetic inputs serve ONLY structural checks: no checkpoint
is loaded, no training occurs, and these numbers are not experimental results.
Only the output JSON is written; no project model source is modified.
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
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("fdr_forward_trace.json"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root = next(p for p in Path(__file__).resolve().parents
                if (p / "model_registry.py").is_file())
    sys.path.insert(0, str(root))
    if args.dependency_dir is not None:
        sys.path.insert(0, str(args.dependency_dir.resolve()))

    import torch
    from model_registry import build_model
    from numeric_domain import INTENSITY_DOMAIN
    from transform_main import FFTRefineBlock

    torch.manual_seed(2026)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model = build_model("ours", variant="full", numeric_domain=INTENSITY_DOMAIN)
    model.cpu().eval()
    expected = {
        "log_branch.bottleneck_refine.freq": [1, 512, 8, 8],
        "log_branch.convproj.freq4": [1, 320, 16, 16],
        "log_branch.convproj.freq3": [1, 128, 32, 32],
        "log_branch.convproj.freq2": [1, 64, 64, 64],
        "log_branch.convproj.freq1": [1, 32, 128, 128],
        "log_branch.convproj.freq0": [1, 16, 256, 256],
    }
    blocks = {n: m for n, m in model.named_modules()
              if isinstance(m, FFTRefineBlock)}
    assert set(blocks) == set(expected), "Registered Full FDR locations changed"

    def shape(tensor):
        return list(tensor.shape)

    def describe(tensor):
        return {"shape": shape(tensor), "dtype": str(tensor.dtype),
                "finite": bool(torch.isfinite(tensor).all())}

    def close(left, right):
        return bool(torch.allclose(left, right, rtol=1e-5, atol=1e-6))

    def max_error(left, right):
        return float((left - right).abs().max())

    def conv_spec(layer):
        return {"in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": list(layer.kernel_size),
                "stride": list(layer.stride), "padding": list(layer.padding),
                "groups": layer.groups, "bias": layer.bias is not None}

    observed = {}
    counts = {}
    handles = []

    def attach(name, block):
        observed[name] = {}
        counts[name] = 0

        def hook(label):
            def capture(_module, inputs, output):
                observed[name][label] = (inputs[0].detach(), output.detach())
                if label == "block":
                    counts[name] += 1
            return capture

        handles.append(block.register_forward_hook(hook("block")))
        for branch in ("spatial", "freq", "fuse"):
            handles.append(getattr(block, branch).register_forward_hook(hook(branch)))
            for index, layer in enumerate(getattr(block, branch)):
                handles.append(layer.register_forward_hook(hook(f"{branch}.{index}")))

    for name, block in blocks.items():
        attach(name, block)
    synthetic = torch.linspace(0, 1, 256 * 256, dtype=torch.float32).reshape(1, 1, 256, 256)
    start = time.perf_counter()
    with torch.inference_mode():
        model_output = model(synthetic)
    model_seconds = time.perf_counter() - start
    for handle in handles:
        handle.remove()
    handles.clear()

    # This shape is a diagnostic, not one of the six manuscript feature sizes.
    odd_name = "diagnostic_odd_width_only"
    odd_block = FFTRefineBlock(4).cpu().eval()
    attach(odd_name, odd_block)
    with torch.inference_mode():
        odd_block(torch.linspace(-1, 1, 4 * 7 * 9).reshape(1, 4, 7, 9))
    for handle in handles:
        handle.remove()

    def verify(name, block, expected_shape, is_model_instance):
        data = observed[name]
        x, actual_output = data["block"]
        batch, channels, height, width = x.shape
        spectral_shape = [batch, channels, height, width // 2 + 1]
        packed_shape = [batch, 2 * channels, height, width // 2 + 1]
        # Independent explicit reconstruction of the equations displayed in Fig. 2.
        with torch.inference_mode():
            spectrum = torch.fft.rfft2(x, norm="ortho")
            packed = torch.cat([spectrum.real, spectrum.imag], dim=1)
            refined = block.freq(packed)
            real, imag = torch.chunk(refined, 2, dim=1)
            complex_refined = torch.complex(real, imag)
            reconstructed = torch.fft.irfft2(complex_refined, s=(height, width), norm="ortho")
            spatial = block.spatial(x)
            joined = torch.cat([spatial, reconstructed], dim=1)
            residual = block.fuse(joined)
            reference_output = x + residual
        checks = {
            "actual_block_executed_once": counts[name] == 1,
            "actual_input_expected_shape": shape(x) == expected_shape,
            "actual_output_same_shape": shape(actual_output) == expected_shape,
            "actual_output_finite": bool(torch.isfinite(actual_output).all()),
            "real_fft_half_spectrum_shape": shape(spectrum) == spectral_shape,
            "frequency_packed_2C_shape": shape(data["freq"][0]) == packed_shape,
            "actual_frequency_input_is_full_real_imag_concat": close(data["freq"][0], packed),
            "frequency_mlp_preserves_packed_shape": shape(data["freq"][1]) == packed_shape,
            "actual_frequency_output_matches_reconstruction": close(data["freq"][1], refined),
            "split_real_and_imag_have_C_channels": shape(real) == spectral_shape and shape(imag) == spectral_shape,
            "explicit_irfft_restores_original_HW": shape(reconstructed) == expected_shape,
            "spatial_branch_matches_reconstruction": close(data["spatial"][1], spatial),
            "fuse_input_is_spatial_then_inverse_fft_concat": close(data["fuse"][0], joined),
            "fuse_input_has_2C_channels_at_original_HW": shape(data["fuse"][0]) == [batch, 2 * channels, height, width],
            "fuse_output_has_C_channels_at_original_HW": shape(data["fuse"][1]) == expected_shape,
            "actual_fuse_residual_matches_reconstruction": close(data["fuse"][1], residual),
            "actual_output_equals_input_plus_fuse_output": close(actual_output, x + data["fuse"][1]),
            "actual_output_matches_independent_reconstruction": close(actual_output, reference_output),
            "spatial_is_depthwise3x3_then_pointwise1x1_then_GELU": (
                block.spatial[0].groups == channels
                and block.spatial[0].kernel_size == (3, 3)
                and block.spatial[1].kernel_size == (1, 1)
                and isinstance(block.spatial[2], torch.nn.GELU)),
            "freq_is_pointwise1x1_then_GELU_then_pointwise1x1": (
                block.freq[0].kernel_size == block.freq[2].kernel_size == (1, 1)
                and isinstance(block.freq[1], torch.nn.GELU)),
            "fuse_is_pointwise1x1_then_GELU_then_conv3x3": (
                block.fuse[0].kernel_size == (1, 1)
                and isinstance(block.fuse[1], torch.nn.GELU)
                and block.fuse[2].kernel_size == (3, 3)
                and block.fuse[2].groups == 1),
            "no_activation_after_residual_sum": close(actual_output, x + data["fuse.2"][1]),
        }
        record = {
            "module": name, "is_registered_model_instance": is_model_instance,
            "class": type(block).__name__, "input": describe(x),
            "rfft2_output": describe(spectrum), "packed_real_imag": describe(packed),
            "frequency_mlp_output": describe(refined), "split_real": describe(real),
            "split_imag": describe(imag), "irfft2_output": describe(reconstructed),
            "spatial_output": describe(spatial), "fusion_input": describe(joined),
            "fused_residual": describe(residual), "output": describe(actual_output),
            "reconstruction_max_abs_error": max_error(actual_output, reference_output),
            "observed_module_io": {key: {"input": describe(pair[0]), "output": describe(pair[1])}
                                   for key, pair in data.items()},
            "convolution_specs": {key: conv_spec(layer) for key, layer in block.named_modules()
                                  if isinstance(layer, torch.nn.Conv2d)},
            "checks": checks,
        }
        if width % 2:
            with torch.inference_mode():
                inferred = torch.fft.irfft2(complex_refined, norm="ortho")
            record["irfft2_without_explicit_size_shape"] = shape(inferred)
            checks["odd_width_requires_explicit_size_to_recover_original_width"] = (
                inferred.shape[-1] == width - 1 and reconstructed.shape[-1] == width)
        record["all_checks_passed"] = all(checks.values())
        return record

    cases = [verify(name, block, expected[name], True) for name, block in blocks.items()]
    cases.append(verify(odd_name, odd_block, [1, 4, 7, 9], False))
    sources = ["transform_main.py", "model_registry.py", "ablation_config.py", "numeric_domain.py"]
    result = {
        "purpose": "Fig. 2 structural verification only; random weights and synthetic inputs; not an experiment or restoration result.",
        "checkpoint_loaded": False, "training_performed": False,
        "model_source_modified": False, "device": "cpu", "inference_mode": True,
        "entry_point": 'model_registry.build_model("ours", variant="full", numeric_domain="intensity_v1")',
        "model_class": type(model).__name__, "registry_metadata": model.registry_metadata,
        "variant_config": model.variant_config, "seed_for_structure_check_only": 2026,
        "python_version": platform.python_version(),
        "package_versions": {package: importlib.metadata.version(package)
                             for package in ("torch", "torchvision", "timm", "numpy")},
        "threads": args.threads, "model_inference_count": 1,
        "model_input": describe(synthetic), "model_output": describe(model_output),
        "model_forward_seconds_this_machine": model_seconds,
        "fft_api": "torch.fft.rfft2 / torch.fft.irfft2", "fft_norm": "ortho",
        "retained_frequency_bins": "all bins returned by rfft2; no mask, truncation, or fftshift in FFTRefineBlock.forward",
        "spectral_width_formula": "W_f = floor(W / 2) + 1",
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "sources_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                           for name in sources},
        "cases": cases, "registered_FDR_instance_count": len(blocks),
        "diagnostic_case_count": 1,
        "check_count": sum(len(case["checks"]) for case in cases),
        "all_checks_passed": all(case["all_checks_passed"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "all_checks_passed": result["all_checks_passed"],
                      "check_count": result["check_count"],
                      "registered_FDR_instances": len(blocks),
                      "model_forward_seconds": round(model_seconds, 3),
                      "input_shapes": [case["input"]["shape"] for case in cases],
                      "max_reconstruction_error": max(case["reconstruction_max_abs_error"] for case in cases)}, indent=2))
    if not result["all_checks_passed"]:
        raise AssertionError({case["module"]: [name for name, ok in case["checks"].items() if not ok]
                              for case in cases if not case["all_checks_passed"]})


if __name__ == "__main__":
    main()
