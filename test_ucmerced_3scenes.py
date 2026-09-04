# -*- coding: utf-8 -*-
# Test 3 typical UCMerced remote-sensing scenes for synthetic SAR denoising
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

# 这里用 *，确保你新增的模型类也能被找到
from transform_main import *


def parse_args():
    parser = argparse.ArgumentParser(description='Test UCMerced 3 typical scenes')

    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to UCMerced SAR mat val dataset')

    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to save test results')

    parser.add_argument('--model', type=str, required=True,
                        help='Model class name, e.g. TransSARV2_DualFreqNG_Bottle')

    parser.add_argument('--loadmodel', type=str, required=True,
                        help='Path to model checkpoint')

    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')

    parser.add_argument('--num_per_scene', type=int, default=1,
                        help='Number of images selected from each scene')

    parser.add_argument('--scenes', nargs='+',
                        default=['agricultural', 'buildings', 'dense_residential'],
                        help='Selected UCMerced scene names')

    return parser.parse_args()


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def normalize_01(x):
    x = np.asarray(x).astype(np.float32)

    x = np.squeeze(x)

    if x.ndim == 3:
        # 如果是 HWC，只取灰度
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
    """
    简单版 SSIM，避免依赖 skimage。
    输入范围：[0, 1]
    """
    img1 = np.clip(img1, 0.0, 1.0).astype(np.float32)
    img2 = np.clip(img2, 0.0, 1.0).astype(np.float32)

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

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


def save_compare_image(path, noisy, pred, clean, titles=None):
    """
    保存横向对比图：
    Noisy | Model Output | Clean

    输入均为 [0, 1] 灰度图
    """
    if titles is None:
        titles = ["Noisy", "Output", "Clean"]

    imgs = [noisy, pred, clean]
    vis_imgs = []

    for img in imgs:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).round().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis_imgs.append(img)

    h, w = vis_imgs[0].shape[:2]

    # 顶部标题区域
    title_h = 35
    panels = []

    for img, title in zip(vis_imgs, titles):
        canvas = np.ones((h + title_h, w, 3), dtype=np.uint8) * 255
        canvas[title_h:, :, :] = img

        cv2.putText(
            canvas,
            title,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        panels.append(canvas)

    compare = np.concatenate(panels, axis=1)

    cv2.imwrite(path, compare)


def find_scene_files(dataset_dir, scenes, num_per_scene=1):
    """
    根据文件名筛选 UCMerced 场景。
    适用于这种结构：

    UCMerced_LandUse_SAR_mat/
    ├── train/
    │   ├── trn_agricultural00.mat
    │   ├── trn_buildings01.mat
    │   └── ...
    └── val/
        ├── val_agricultural03.mat
        ├── val_buildings05.mat
        └── ...

    不要求有 agricultural / buildings / dense_residential 子文件夹。
    """
    # 不同写法的别名，防止 dense_residential 和 denseresidential 不一致
    scene_aliases = {
        "agricultural": ["agricultural", "agriculture"],
        "buildings": ["buildings", "building"],
        "dense_residential": [
            "dense_residential",
            "denseresidential",
            "dense-residential",
            "dense residential"
        ]
    }

    all_mat_files = []

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(".mat"):
                all_mat_files.append(os.path.join(root, f))

    selected = {}

    for scene in scenes:
        aliases = scene_aliases.get(scene, [scene])

        candidates = []

        for path in all_mat_files:
            filename = os.path.basename(path).lower()

            for alias in aliases:
                alias = alias.lower().replace(" ", "")
                filename_norm = filename.replace("_", "").replace("-", "").replace(" ", "")

                if alias.replace("_", "").replace("-", "").replace(" ", "") in filename_norm:
                    candidates.append(path)
                    break

        candidates = sorted(candidates)

        if len(candidates) == 0:
            print(f"[警告] 没有找到场景 {scene} 的 .mat 文件")
            selected[scene] = []
        else:
            selected[scene] = candidates[:num_per_scene]
            print(f"\n[{scene}] 选中的文件：")
            for p in selected[scene]:
                print("  ", p)

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

    # 去掉 DataParallel 的 module. 前缀
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

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ensure_dir(args.save_path)

    model = build_model(args.model, args.loadmodel, device)

    selected_files = find_scene_files(
        dataset_dir=args.dataset,
        scenes=args.scenes,
        num_per_scene=args.num_per_scene
    )

    csv_path = os.path.join(args.save_path, 'metrics.csv')

    rows = []

    for scene_name, file_list in selected_files.items():
        scene_save_dir = os.path.join(args.save_path, scene_name)
        ensure_dir(scene_save_dir)

        for idx, mat_path in enumerate(file_list):
            print(f"Testing scene: {scene_name}, file: {mat_path}")

            noisy, clean = load_mat_pair(mat_path)

            x = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(x)

            pred = pred.detach().cpu().numpy()
            pred = np.squeeze(pred)
            pred = np.clip(pred, 0.0, 1.0)

            psnr = calc_psnr(pred, clean)
            ssim = calc_ssim(pred, clean)

            noisy_psnr = calc_psnr(noisy, clean)
            noisy_ssim = calc_ssim(noisy, clean)

            base_name = os.path.splitext(os.path.basename(mat_path))[0]
            prefix = f"{idx + 1:02d}_{base_name}"

            save_gray(os.path.join(scene_save_dir, f"{prefix}_clean.png"), clean)
            save_gray(os.path.join(scene_save_dir, f"{prefix}_noisy.png"), noisy)
            save_gray(os.path.join(scene_save_dir, f"{prefix}_{args.model}.png"), pred)

            save_compare_image(
                os.path.join(scene_save_dir, f"{prefix}_compare.png"),
                noisy,
                pred,
                clean,
                titles=["Noisy", args.model, "Clean"]
            )

            # 残差图：模型输出和 clean 的绝对误差
            residual = np.abs(clean - pred)
            save_gray(os.path.join(scene_save_dir, f"{prefix}_residual.png"), residual)

            rows.append({
                'scene': scene_name,
                'file': mat_path,
                'model': args.model,
                'noisy_psnr': noisy_psnr,
                'noisy_ssim': noisy_ssim,
                'psnr': psnr,
                'ssim': ssim
            })

            print(f"  Noisy PSNR: {noisy_psnr:.4f}, Noisy SSIM: {noisy_ssim:.4f}")
            print(f"  Pred  PSNR: {psnr:.4f}, Pred  SSIM: {ssim:.4f}")

    # 写 CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'scene',
            'file',
            'model',
            'noisy_psnr',
            'noisy_ssim',
            'psnr',
            'ssim'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    # 计算平均
    if len(rows) > 0:
        avg_psnr = np.mean([r['psnr'] for r in rows])
        avg_ssim = np.mean([r['ssim'] for r in rows])

        avg_noisy_psnr = np.mean([r['noisy_psnr'] for r in rows])
        avg_noisy_ssim = np.mean([r['noisy_ssim'] for r in rows])

        print("\n========== Average Results ==========")
        print(f"Noisy Avg PSNR: {avg_noisy_psnr:.4f}")
        print(f"Noisy Avg SSIM: {avg_noisy_ssim:.4f}")
        print(f"{args.model} Avg PSNR: {avg_psnr:.4f}")
        print(f"{args.model} Avg SSIM: {avg_ssim:.4f}")

    print("\nDone.")
    print("Results saved to:", args.save_path)
    print("CSV saved to:", csv_path)


if __name__ == '__main__':
    main()