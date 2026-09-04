# -*- coding: utf-8 -*-
# Test all UCMerced synthetic SAR images from 3 classes:
# agricultural / buildings / residential
# Metrics: PSNR / SSIM

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
    parser = argparse.ArgumentParser(description='Test UCMerced 3 classes')

    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to UCMerced SAR mat test dataset')

    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to save test results')

    parser.add_argument('--model', type=str, required=True,
                        help='Model class name')

    parser.add_argument('--loadmodel', type=str, required=True,
                        help='Path to model checkpoint')

    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')

    parser.add_argument('--classes', nargs='+',
                        default=['agricultural', 'buildings', 'residential'],
                        help='Selected class keywords')

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


def save_gray(path, img):
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)
    cv2.imwrite(path, img)


def normalize_name(name):
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def find_class_files(dataset_dir, class_names):
    """
    根据文件名筛选类别。
    支持：
        tst_agricultural00.mat
        test_agricultural00.mat
        agricultural00.mat
        tst_buildings00.mat
        tst_residential00.mat
        tst_denseresidential00.mat
        tst_mediumresidential00.mat
        tst_sparseresidential00.mat
    """

    class_aliases = {
        "agricultural": [
            "agricultural",
            "agriculture"
        ],
        "buildings": [
            "buildings",
            "building"
        ],
        "residential": [
            "residential",
            "denseresidential",
            "mediumresidential",
            "sparseresidential",
            "dense_residential",
            "medium_residential",
            "sparse_residential"
        ]
    }

    all_mat_files = []

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(".mat"):
                all_mat_files.append(os.path.join(root, f))

    selected = {}

    for cls in class_names:
        aliases = class_aliases.get(cls, [cls])
        aliases_norm = [normalize_name(a) for a in aliases]

        candidates = []

        for path in all_mat_files:
            filename = normalize_name(os.path.basename(path))

            for alias in aliases_norm:
                if alias in filename:
                    candidates.append(path)
                    break

        candidates = sorted(candidates)
        selected[cls] = candidates

        print(f"\n[{cls}] 找到 {len(candidates)} 个文件")
        for p in candidates[:5]:
            print("  ", p)
        if len(candidates) > 5:
            print("  ...")

    return selected


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


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ensure_dir(args.save_path)

    model = build_model(args.model, args.loadmodel, device)

    selected_files = find_class_files(args.dataset, args.classes)

    all_rows = []
    class_avg_rows = []

    for class_name, file_list in selected_files.items():
        class_save_dir = os.path.join(args.save_path, class_name)
        ensure_dir(class_save_dir)

        class_rows = []

        if len(file_list) == 0:
            print(f"[警告] 类别 {class_name} 没有找到测试文件，跳过")
            continue

        for idx, mat_path in enumerate(file_list):
            print(f"\nTesting [{class_name}] {idx + 1}/{len(file_list)}: {mat_path}")

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

            base_name = os.path.splitext(os.path.basename(mat_path))[0]

            save_gray(os.path.join(class_save_dir, f"{base_name}_clean.png"), clean)
            save_gray(os.path.join(class_save_dir, f"{base_name}_noisy.png"), noisy)
            save_gray(os.path.join(class_save_dir, f"{base_name}_{args.model}.png"), pred)

            residual = np.abs(clean - pred)
            save_gray(os.path.join(class_save_dir, f"{base_name}_residual.png"), residual)

            row = {
                'class': class_name,
                'file': mat_path,
                'model': args.model,
                'noisy_psnr': noisy_psnr,
                'noisy_ssim': noisy_ssim,
                'pred_psnr': pred_psnr,
                'pred_ssim': pred_ssim
            }

            class_rows.append(row)
            all_rows.append(row)

            print(f"  Noisy PSNR: {noisy_psnr:.4f}, Noisy SSIM: {noisy_ssim:.4f}")
            print(f"  Pred  PSNR: {pred_psnr:.4f}, Pred  SSIM: {pred_ssim:.4f}")

        class_csv_path = os.path.join(class_save_dir, f"{class_name}_metrics.csv")
        write_csv(
            class_csv_path,
            class_rows,
            [
                'class',
                'file',
                'model',
                'noisy_psnr',
                'noisy_ssim',
                'pred_psnr',
                'pred_ssim'
            ]
        )

        class_avg = {
            'class': class_name,
            'num_images': len(class_rows),
            'model': args.model,
            'avg_noisy_psnr': float(np.mean([r['noisy_psnr'] for r in class_rows])),
            'avg_noisy_ssim': float(np.mean([r['noisy_ssim'] for r in class_rows])),
            'avg_pred_psnr': float(np.mean([r['pred_psnr'] for r in class_rows])),
            'avg_pred_ssim': float(np.mean([r['pred_ssim'] for r in class_rows]))
        }

        class_avg_rows.append(class_avg)

        print(f"\n========== {class_name} Average ==========")
        print(f"Num images: {class_avg['num_images']}")
        print(f"Noisy Avg PSNR: {class_avg['avg_noisy_psnr']:.4f}")
        print(f"Noisy Avg SSIM: {class_avg['avg_noisy_ssim']:.4f}")
        print(f"Pred  Avg PSNR: {class_avg['avg_pred_psnr']:.4f}")
        print(f"Pred  Avg SSIM: {class_avg['avg_pred_ssim']:.4f}")

    metrics_all_path = os.path.join(args.save_path, "metrics_all.csv")
    write_csv(
        metrics_all_path,
        all_rows,
        [
            'class',
            'file',
            'model',
            'noisy_psnr',
            'noisy_ssim',
            'pred_psnr',
            'pred_ssim'
        ]
    )

    class_avg_path = os.path.join(args.save_path, "metrics_class_average.csv")
    write_csv(
        class_avg_path,
        class_avg_rows,
        [
            'class',
            'num_images',
            'model',
            'avg_noisy_psnr',
            'avg_noisy_ssim',
            'avg_pred_psnr',
            'avg_pred_ssim'
        ]
    )

    if len(all_rows) > 0:
        total_avg = [{
            'num_images': len(all_rows),
            'model': args.model,
            'total_noisy_psnr': float(np.mean([r['noisy_psnr'] for r in all_rows])),
            'total_noisy_ssim': float(np.mean([r['noisy_ssim'] for r in all_rows])),
            'total_pred_psnr': float(np.mean([r['pred_psnr'] for r in all_rows])),
            'total_pred_ssim': float(np.mean([r['pred_ssim'] for r in all_rows]))
        }]

        total_avg_path = os.path.join(args.save_path, "metrics_total_average.csv")
        write_csv(
            total_avg_path,
            total_avg,
            [
                'num_images',
                'model',
                'total_noisy_psnr',
                'total_noisy_ssim',
                'total_pred_psnr',
                'total_pred_ssim'
            ]
        )

        print("\n========== Total Average ==========")
        print(f"Total images: {total_avg[0]['num_images']}")
        print(f"Noisy Avg PSNR: {total_avg[0]['total_noisy_psnr']:.4f}")
        print(f"Noisy Avg SSIM: {total_avg[0]['total_noisy_ssim']:.4f}")
        print(f"Pred  Avg PSNR: {total_avg[0]['total_pred_psnr']:.4f}")
        print(f"Pred  Avg SSIM: {total_avg[0]['total_pred_ssim']:.4f}")

    print("\nDone.")
    print("All metrics saved to:", metrics_all_path)
    print("Class average saved to:", class_avg_path)
    print("Results saved to:", args.save_path)


if __name__ == '__main__':
    main()