# -*- coding: utf-8 -*-
# Select 60 representative real SAR images:
# 20 homogeneous + 20 structural edge + 20 complex texture

import os
import csv
import shutil
import argparse
import cv2
import numpy as np
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True,
                        help='Original real SAR test folder, e.g. ./real_sar_dataset/test')

    parser.add_argument('--output', type=str, required=True,
                        help='Output selected folder, e.g. ./real_sar_dataset/test_selected')

    parser.add_argument('--n_each', type=int, default=20,
                        help='Number of images selected for each category')

    parser.add_argument('--score_size', type=int, default=256,
                        help='Resize image to this size for scoring only')

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

    x = x.astype(np.float32)

    if x.max() > 1.5:
        x = x / 255.0

    x = x - x.min()
    if x.max() > 1e-8:
        x = x / x.max()

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def load_sar_image(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == '.mat':
        data = loadmat(path)

        candidate_keys = [
            'noisy',
            'sar',
            'image',
            'img',
            'data',
            'input'
        ]

        for key in candidate_keys:
            if key in data:
                return normalize_01(data[key])

        for key, value in data.items():
            if key.startswith('__'):
                continue

            arr = np.asarray(value)
            if arr.ndim == 2 or arr.ndim == 3:
                print(f"[提示] {path} 使用字段: {key}")
                return normalize_01(arr)

        raise KeyError(f"{path} 中没有找到可用图像字段")

    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"无法读取图像: {path}")

        return normalize_01(img)


def collect_files(input_dir):
    exts = ['.mat', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

    files = []

    for root, _, names in os.walk(input_dir):
        for name in names:
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                files.append(os.path.join(root, name))

    return sorted(files)


def compute_orientation_entropy(gray, grad, edge_mask, num_bins=8):
    if np.sum(edge_mask) < 10:
        return 1.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    angle = np.arctan2(gy, gx)
    angle = (angle + np.pi) / (2 * np.pi)

    vals = angle[edge_mask]
    hist, _ = np.histogram(vals, bins=num_bins, range=(0, 1), density=False)

    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-12)

    entropy = -np.sum(hist * np.log(hist + 1e-12))
    entropy = entropy / np.log(num_bins)

    return float(entropy)


def compute_patch_variance_ratio(gray, patch_size=32, var_thr=0.002):
    H, W = gray.shape

    vars_ = []

    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch = gray[y:y + patch_size, x:x + patch_size]
            vars_.append(np.var(patch))

    if len(vars_) == 0:
        return 0.0

    vars_ = np.asarray(vars_)

    uniform_ratio = np.mean(vars_ < var_thr)

    return float(uniform_ratio)


def compute_scores(img, score_size=256):
    gray = normalize_01(img)

    gray = cv2.resize(gray, (score_size, score_size), interpolation=cv2.INTER_AREA)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.sqrt(gx ** 2 + gy ** 2)

    grad_mean = float(np.mean(grad))
    grad_std = float(np.std(grad))

    edge_thr = np.percentile(grad, 85)
    edge_mask = grad > edge_thr
    edge_density = float(np.mean(edge_mask))

    # 局部方差，用于纹理复杂度
    mean = cv2.blur(gray, (9, 9))
    mean_sq = cv2.blur(gray * gray, (9, 9))
    local_var = np.maximum(mean_sq - mean * mean, 0.0)

    texture_score = float(np.mean(local_var))
    texture_std = float(np.std(local_var))

    gray_std = float(np.std(gray))

    uniform_ratio = compute_patch_variance_ratio(
        gray,
        patch_size=32,
        var_thr=0.002
    )

    orientation_entropy = compute_orientation_entropy(
        gray,
        grad,
        edge_mask,
        num_bins=8
    )

    # 三类分数
    # 1. 均匀区域：均匀 patch 比例高、边缘少、纹理弱
    homogeneous_score = (
        2.0 * uniform_ratio
        - 1.0 * edge_density
        - 50.0 * texture_score
        - 0.5 * gray_std
    )

    # 2. 建筑/道路边缘：边缘密度高，且方向性强
    # orientation_entropy 越低，说明边缘方向越集中，更像建筑/道路
    structural_score = (
        2.0 * edge_density
        + 1.0 * grad_mean
        + 1.0 * (1.0 - orientation_entropy)
        - 20.0 * texture_score
    )

    # 3. 山地/林地/复杂纹理：纹理强、梯度强、方向杂
    texture_complex_score = (
        80.0 * texture_score
        + 1.5 * grad_mean
        + 1.0 * orientation_entropy
        + 0.5 * texture_std
    )

    return {
        'edge_density': edge_density,
        'texture_score': texture_score,
        'texture_std': texture_std,
        'gray_std': gray_std,
        'grad_mean': grad_mean,
        'grad_std': grad_std,
        'uniform_ratio': uniform_ratio,
        'orientation_entropy': orientation_entropy,
        'homogeneous_score': float(homogeneous_score),
        'structural_score': float(structural_score),
        'texture_complex_score': float(texture_complex_score)
    }


def save_preview(path, img, text):
    img = normalize_01(img)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    title_h = 35
    canvas = np.ones((256 + title_h, 256, 3), dtype=np.uint8) * 255
    canvas[title_h:, :, :] = img

    cv2.putText(
        canvas,
        text[:28],
        (5, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    cv2.imwrite(path, canvas)


def copy_selected(records, category, selected_records, output_dir):
    cat_dir = os.path.join(output_dir, category)
    preview_dir = os.path.join(output_dir, category + '_preview')

    ensure_dir(cat_dir)
    ensure_dir(preview_dir)

    for i, r in enumerate(selected_records):
        src = r['path']
        ext = os.path.splitext(src)[1]
        dst_name = f"{i + 1:02d}_{category}_{os.path.basename(src)}"
        dst = os.path.join(cat_dir, dst_name)

        shutil.copy(src, dst)

        try:
            img = load_sar_image(src)
            preview_path = os.path.join(preview_dir, f"{i + 1:02d}_{category}.png")
            save_preview(preview_path, img, os.path.basename(src))
        except Exception as e:
            print(f"[警告] 保存预览失败: {src}, {e}")


def write_csv(path, records):
    if len(records) == 0:
        return

    fieldnames = [
        'path',
        'category',
        'edge_density',
        'texture_score',
        'texture_std',
        'gray_std',
        'grad_mean',
        'grad_std',
        'uniform_ratio',
        'orientation_entropy',
        'homogeneous_score',
        'structural_score',
        'texture_complex_score'
    ]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def main():
    args = parse_args()

    ensure_dir(args.output)

    files = collect_files(args.input)

    print(f"Found {len(files)} files.")

    records = []

    for i, path in enumerate(files):
        print(f"[{i + 1}/{len(files)}] Scoring: {path}")

        try:
            img = load_sar_image(path)
            scores = compute_scores(img, score_size=args.score_size)

            r = {
                'path': path,
                'category': '',
                **scores
            }

            records.append(r)

        except Exception as e:
            print(f"[跳过] {path}: {e}")

    if len(records) == 0:
        raise RuntimeError("没有成功读取任何图像。")

    # 1. 选 20 张均匀区域
    homogeneous_sorted = sorted(
        records,
        key=lambda x: x['homogeneous_score'],
        reverse=True
    )

    selected_homogeneous = homogeneous_sorted[:args.n_each]

    used_paths = set([r['path'] for r in selected_homogeneous])

    # 2. 从剩余图中选 20 张结构边缘
    remaining_1 = [r for r in records if r['path'] not in used_paths]

    structural_sorted = sorted(
        remaining_1,
        key=lambda x: x['structural_score'],
        reverse=True
    )

    selected_structural = structural_sorted[:args.n_each]

    used_paths.update([r['path'] for r in selected_structural])

    # 3. 从剩余图中选 20 张复杂纹理
    remaining_2 = [r for r in records if r['path'] not in used_paths]

    texture_sorted = sorted(
        remaining_2,
        key=lambda x: x['texture_complex_score'],
        reverse=True
    )

    selected_texture = texture_sorted[:args.n_each]

    for r in selected_homogeneous:
        r['category'] = 'homogeneous'

    for r in selected_structural:
        r['category'] = 'structural'

    for r in selected_texture:
        r['category'] = 'texture'

    selected_all = selected_homogeneous + selected_structural + selected_texture

    # 保存所有评分
    write_csv(os.path.join(args.output, 'all_scores.csv'), records)

    # 保存选中的 60 张评分
    write_csv(os.path.join(args.output, 'selected_60_scores.csv'), selected_all)

    # 复制文件
    copy_selected(records, 'homogeneous', selected_homogeneous, args.output)
    copy_selected(records, 'structural', selected_structural, args.output)
    copy_selected(records, 'texture', selected_texture, args.output)

    # 额外复制到 all 文件夹，方便 test_real_sar.py 一次性测试
    all_dir = os.path.join(args.output, 'all')
    ensure_dir(all_dir)

    for i, r in enumerate(selected_all):
        src = r['path']
        category = r['category']
        dst_name = f"{i + 1:02d}_{category}_{os.path.basename(src)}"
        dst = os.path.join(all_dir, dst_name)
        shutil.copy(src, dst)

    print("\nDone.")
    print("Selected files saved to:", args.output)
    print("All selected 60 files:", all_dir)
    print("CSV:", os.path.join(args.output, 'selected_60_scores.csv'))


if __name__ == '__main__':
    main()