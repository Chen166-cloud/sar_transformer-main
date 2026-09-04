# -*- coding: utf-8 -*-
# Test all UCMerced synthetic SAR test set
# Metrics: average PSNR / SSIM

import argparse
import os
import csv
import math
import cv2
import numpy as np
import torch
from torch import nn
from scipy.io import loadmat

from transform_main import *


def parse_args():
    parser = argparse.ArgumentParser(description='Test all UCMerced SAR mat test set')

    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to UCMerced SAR mat test dataset')

    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to save metrics csv')

    parser.add_argument('--model', type=str, required=True,
                        help='Model class name')

    parser.add_argument('--loadmodel', type=str, required=True,
                        help='Path to model checkpoint')

    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')

    return parser.parse_args()


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def normalize_01(x):
    x = np.asarray(x).astype(np.float32)
    x = np.squeeze(x)

    if x.ndim == 3:
        if x.shape[-1] == 3:
            x = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        else:
            x = x[..., 0]

    if x.max() > 1.5:
        x = x / 255.0

    x = np.clip(x, 0.0, 1.0)

    return x.astype(np.float32)


def load_mat_pair(mat_path):
    data = loadmat(mat_path)

    if 'noisy' not in data:
        raise KeyError(f"{mat_path} 中找不到字段 noisy")

    if 'clean' not in data:
        raise KeyError(f"{mat_path} 中找不到字段 clean")

    noisy = normalize_01(data['noisy'])
    clean = normalize_01(data['clean'])

    return noisy, clean


def calc_psnr(pred, target, eps=1e-10):
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    mse = np.mean((pred - target) ** 2)

    if mse < eps:
        return 100.0

    return 10.0 * math.log10(1.0 / mse)


def calc_ssim(img1, img2):
    img1 = np.clip(img1, 0.0, 1.0).astype(np.float32)
    img2 = np.clip(img2, 0.0, 1.0).astype(np.float32)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    kernel_size = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(img1, kernel_size, sigma)
    mu2 = cv2.GaussianBlur(img2, kernel_size, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, kernel_size, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, kernel_size, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, kernel_size, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def get_all_mat_files(dataset_dir):
    mat_files = []

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith('.mat'):
                mat_files.append(os.path.join(root, f))

    mat_files = sorted(mat_files)

    return mat_files


def build_model(model_name, checkpoint_path, device):
    if model_name not in globals():
        raise ValueError(f"transform_main.py 中找不到模型类：{model_name}")

    model_class = globals()[model_name]
    model = model_class()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    return model


def main():
    args = parse_args()

    ensure_dir(args.save_path)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = build_model(args.model, args.loadmodel, device)

    mat_files = get_all_mat_files(args.dataset)

    if len(mat_files) == 0:
        raise RuntimeError(f"没有在 {args.dataset} 中找到 .mat 文件")

    print(f"Found {len(mat_files)} test files.")

    rows = []

    for idx, mat_path in enumerate(mat_files):
        noisy, clean = load_mat_pair(mat_path)

        x = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)

        pred = pred.detach().cpu().numpy()
        pred = np.squeeze(pred)
        pred = np.clip(pred, 0.0, 1.0)

        noisy_psnr = calc_psnr(noisy, clean)
        noisy_ssim = calc_ssim(noisy, clean)

        pred_psnr = calc_psnr(pred, clean)
        pred_ssim = calc_ssim(pred, clean)

        rows.append({
            'index': idx + 1,
            'file': mat_path,
            'model': args.model,
            'noisy_psnr': noisy_psnr,
            'noisy_ssim': noisy_ssim,
            'pred_psnr': pred_psnr,
            'pred_ssim': pred_ssim
        })

        print(
            f"[{idx + 1}/{len(mat_files)}] "
            f"{os.path.basename(mat_path)} | "
            f"PSNR: {pred_psnr:.4f}, SSIM: {pred_ssim:.4f}"
        )

    csv_path = os.path.join(args.save_path, f'{args.model}_all_test_metrics.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'index',
            'file',
            'model',
            'noisy_psnr',
            'noisy_ssim',
            'pred_psnr',
            'pred_ssim'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    avg_noisy_psnr = np.mean([r['noisy_psnr'] for r in rows])
    avg_noisy_ssim = np.mean([r['noisy_ssim'] for r in rows])

    avg_pred_psnr = np.mean([r['pred_psnr'] for r in rows])
    avg_pred_ssim = np.mean([r['pred_ssim'] for r in rows])

    print("\n========== Average Test Results ==========")
    print(f"Dataset: {args.dataset}")
    print(f"Number of test images: {len(rows)}")
    print(f"Noisy Avg PSNR: {avg_noisy_psnr:.4f}")
    print(f"Noisy Avg SSIM: {avg_noisy_ssim:.4f}")
    print(f"{args.model} Avg PSNR: {avg_pred_psnr:.4f}")
    print(f"{args.model} Avg SSIM: {avg_pred_ssim:.4f}")

    print("\nCSV saved to:", csv_path)


if __name__ == '__main__':
    main()