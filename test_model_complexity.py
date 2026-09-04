# -*- coding: utf-8 -*-
# Test Params(M), FLOPs(G), and inference Time(ms) for SAR denoising models

import argparse
import os
import csv
import time
import math
import torch
from torch import nn

from transform_main import *


try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


def parse_args():
    parser = argparse.ArgumentParser(description="Test model complexity")

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model class names, e.g. TransSARV2 TransSARV2_Freq"
    )

    parser.add_argument(
        "--loadmodels",
        nargs="+",
        required=True,
        help="Checkpoint paths. Number must match --models"
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=256,
        help="Input image size, default 256"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup iterations"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=200,
        help="Timing iterations"
    )

    parser.add_argument(
        "--save_csv",
        type=str,
        default="./model_complexity.csv",
        help="CSV save path"
    )

    return parser.parse_args()


def strip_module_prefix(state_dict):
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    state_dict = strip_module_prefix(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")


def build_model(model_name, checkpoint_path, device):
    if model_name not in globals():
        raise ValueError(f"transform_main.py 中找不到模型类：{model_name}")

    model_class = globals()[model_name]
    model = model_class()

    if checkpoint_path.lower() not in ["none", "null", ""]:
        load_checkpoint(model, checkpoint_path)

    model.to(device)
    model.eval()

    return model


def count_params(model):
    params = sum(p.numel() for p in model.parameters())
    return params / 1e6


def estimate_fft_flops(model, dummy_input, device):
    """
    粗略估计 FFTRefineBlock 中 rfft2 + irfft2 的额外计算量。
    注意：
        thop 通常不会统计 torch.fft 的 FLOPs。
        这里用近似公式：
        FFT2D cost ≈ 5 * H * W * log2(H * W)
        rfft2 + irfft2 约按 2 次 FFT 估计。
    """

    fft_flops = {"value": 0.0}
    hooks = []

    def hook_fn(module, inputs, output):
        x = inputs[0]

        if not torch.is_tensor(x):
            return

        if x.dim() != 4:
            return

        B, C, H, W = x.shape

        n = H * W
        if n <= 1:
            return

        # rfft2 + irfft2
        cost = 2.0 * 5.0 * B * C * H * W * math.log2(n)

        fft_flops["value"] += cost

    for m in model.modules():
        if m.__class__.__name__ == "FFTRefineBlock":
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()

    return fft_flops["value"] / 1e9


def count_flops(model, dummy_input, device):
    if not HAS_THOP:
        return None, None, None

    with torch.no_grad():
        flops, params = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )

    thop_flops_g = flops / 1e9

    fft_flops_g = estimate_fft_flops(model, dummy_input, device)

    total_flops_g = thop_flops_g + fft_flops_g

    return thop_flops_g, fft_flops_g, total_flops_g


def measure_time(model, dummy_input, device, warmup=50, repeat=200):
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        times = []

        with torch.no_grad():
            for _ in range(repeat):
                starter.record()
                _ = model(dummy_input)
                ender.record()

                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                times.append(curr_time)

        times = torch.tensor(times)

        return {
            "mean_ms": float(times.mean().item()),
            "std_ms": float(times.std().item()),
            "min_ms": float(times.min().item()),
            "max_ms": float(times.max().item())
        }

    else:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        times = []

        with torch.no_grad():
            for _ in range(repeat):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()

                times.append((end - start) * 1000.0)

        times = torch.tensor(times)

        return {
            "mean_ms": float(times.mean().item()),
            "std_ms": float(times.std().item()),
            "min_ms": float(times.min().item()),
            "max_ms": float(times.max().item())
        }


def write_csv(save_path, rows):
    save_dir = os.path.dirname(save_path)

    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    fieldnames = [
        "model",
        "checkpoint",
        "input_shape",
        "params_M",
        "thop_flops_G",
        "estimated_fft_flops_G",
        "total_flops_G",
        "time_mean_ms",
        "time_std_ms",
        "time_min_ms",
        "time_max_ms"
    ]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    if len(args.models) != len(args.loadmodels):
        raise ValueError("--models 和 --loadmodels 数量必须一致")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if not HAS_THOP:
        print("[警告] 当前环境未安装 thop，将无法统计 FLOPs。")
        print("请运行：pip install thop")

    rows = []

    for model_name, checkpoint_path in zip(args.models, args.loadmodels):
        print("\n" + "=" * 80)
        print("Testing model:", model_name)

        model = build_model(model_name, checkpoint_path, device)

        dummy_input = torch.randn(
            1,
            1,
            args.input_size,
            args.input_size
        ).to(device)

        # Params
        params_M = count_params(model)

        # FLOPs
        thop_flops_G = None
        estimated_fft_flops_G = None
        total_flops_G = None

        if HAS_THOP:
            thop_flops_G, estimated_fft_flops_G, total_flops_G = count_flops(
                model,
                dummy_input,
                device
            )

        # Time
        time_info = measure_time(
            model,
            dummy_input,
            device,
            warmup=args.warmup,
            repeat=args.repeat
        )

        row = {
            "model": model_name,
            "checkpoint": checkpoint_path,
            "input_shape": f"[1, 1, {args.input_size}, {args.input_size}]",
            "params_M": params_M,
            "thop_flops_G": thop_flops_G,
            "estimated_fft_flops_G": estimated_fft_flops_G,
            "total_flops_G": total_flops_G,
            "time_mean_ms": time_info["mean_ms"],
            "time_std_ms": time_info["std_ms"],
            "time_min_ms": time_info["min_ms"],
            "time_max_ms": time_info["max_ms"]
        }

        rows.append(row)

        print(f"Params(M): {params_M:.4f}")

        if total_flops_G is not None:
            print(f"THOP FLOPs(G): {thop_flops_G:.4f}")
            print(f"Estimated FFT FLOPs(G): {estimated_fft_flops_G:.4f}")
            print(f"Total FLOPs(G): {total_flops_G:.4f}")
        else:
            print("FLOPs(G): None")

        print(f"Time mean(ms): {time_info['mean_ms']:.4f}")
        print(f"Time std(ms):  {time_info['std_ms']:.4f}")
        print(f"Time min(ms):  {time_info['min_ms']:.4f}")
        print(f"Time max(ms):  {time_info['max_ms']:.4f}")

        del model
        torch.cuda.empty_cache()

    write_csv(args.save_csv, rows)

    print("\n" + "=" * 80)
    print("Done.")
    print("CSV saved to:", args.save_csv)


if __name__ == "__main__":
    main()