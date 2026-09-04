# -*- coding: utf-8 -*-
# Test real SAR images
# Metrics: ENL / M / EPI

import argparse
import os
import csv
import cv2
import numpy as np
import torch
from torch import nn
from scipy.io import loadmat

from transform_main import *


def parse_args():
    parser = argparse.ArgumentParser(description='Test real SAR images')

    parser.add_argument('--dataset', type=str, default='./datasets/real_sar_dataset/test',
                        help='Path to real SAR test dataset')

    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to save results')

    parser.add_argument('--model', type=str, required=True,
                        help='Model class name')

    parser.add_argument('--loadmodel', type=str, required=True,
                        help='Path to model checkpoint')

    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')

    parser.add_argument('--roi_size', type=int, default=32,
                        help='ROI size for ENL calculation')

    parser.add_argument('--num_rois', type=int, default=5,
                        help='Number of homogeneous ROIs for ENL')

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

    x = x - x.min()
    if x.max() > 1e-8:
        x = x / x.max()

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def load_real_sar(path):
    """
    支持：
    1. .mat 文件
    2. png/jpg/tif 图像
    """

    ext = os.path.splitext(path)[1].lower()

    if ext == '.mat':
        data = loadmat(path)

        # 优先读取这些字段
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

        # 如果字段名不固定，就自动找第一个二维数组
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


def collect_files(dataset_dir):
    exts = ['.mat', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

    files = []

    for root, _, filenames in os.walk(dataset_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))

    return sorted(files)


def save_gray(path, img):
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)
    cv2.imwrite(path, img)


def save_compare_image(path, noisy, pred, residual, titles=None):
    """
    保存：
    Noisy | Output | Residual
    """

    if titles is None:
        titles = ["Noisy", "Output", "Residual"]

    imgs = [noisy, pred, residual]
    vis_imgs = []

    for img in imgs:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).round().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis_imgs.append(img)

    h, w = vis_imgs[0].shape[:2]
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


def calc_enl(img, roi_size=32, num_rois=5, eps=1e-8):
    """
    自动选择方差最低的若干个 ROI 计算 ENL。
    ENL = mean^2 / variance

    注意：
    这个是自动 ROI，适合批量测试。
    论文最终图表最好人工确认 ROI 是否位于均匀区域。
    """

    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    H, W = img.shape

    rois = []

    step = roi_size

    for y in range(0, H - roi_size + 1, step):
        for x in range(0, W - roi_size + 1, step):
            roi = img[y:y + roi_size, x:x + roi_size]

            mean = np.mean(roi)
            var = np.var(roi)

            if mean > 0.03:
                rois.append({
                    'x': x,
                    'y': y,
                    'mean': mean,
                    'var': var,
                    'score': var
                })

    if len(rois) == 0:
        return 0.0

    rois = sorted(rois, key=lambda r: r['score'])
    selected = rois[:num_rois]

    enl_values = []

    for r in selected:
        enl = (r['mean'] ** 2) / (r['var'] + eps)
        enl_values.append(enl)

    return float(np.mean(enl_values))


def integral_image(arr):
    h, w = arr.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    return ii


def box_sum_valid(arr, kh, kw):
    ii = integral_image(arr)
    return ii[kh:, kw:] - ii[:-kh, kw:] - ii[kh:, :-kw] + ii[:-kh, :-kw]


def local_mean_var_valid(arr, win):
    area = win * win

    s1 = box_sum_valid(arr, win, win)
    s2 = box_sum_valid(arr * arr, win, win)

    mean = s1 / area
    var = s2 / area - mean * mean
    var = np.maximum(var, 0.0)

    return mean, var


def enl_from_mean_var(mean, var, eps=1e-12):
    return (mean * mean) / np.maximum(var, eps)


def quantize_linear(arr, levels=8, vmin=None, vmax=None):
    if vmin is None:
        vmin = float(np.min(arr))
    if vmax is None:
        vmax = float(np.max(arr))

    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.int32)

    norm = (arr - vmin) / (vmax - vmin)
    q = np.floor(norm * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1)

    return q


def local_homogeneity_mean_from_quantized(q, win, direction="h"):
    if direction == "h":
        a = q[:, :-1]
        b = q[:, 1:]
        kh, kw = win, win - 1
    elif direction == "v":
        a = q[:-1, :]
        b = q[1:, :]
        kh, kw = win - 1, win
    else:
        raise ValueError("direction 只能是 'h' 或 'v'")

    weight_map = 1.0 / (1.0 + (a - b) ** 2)
    local_sum = box_sum_valid(weight_map.astype(np.float64), kh, kw)
    local_mean = local_sum / (kh * kw)

    return float(np.mean(local_mean))


def compute_delta_h_from_array(
    ratio_img,
    win=25,
    levels=8,
    p_shuffle=100,
    seed=42,
    direction="h"
):
    ratio_img = np.asarray(ratio_img, dtype=np.float64)

    vmin = float(np.min(ratio_img))
    vmax = float(np.max(ratio_img))

    q = quantize_linear(ratio_img, levels=levels, vmin=vmin, vmax=vmax)

    ho = local_homogeneity_mean_from_quantized(q, win=win, direction=direction)

    rng = np.random.default_rng(seed)
    flat = q.ravel()

    hg_list = []

    for _ in range(p_shuffle):
        q_shuffle = rng.permutation(flat).reshape(q.shape)
        hg_i = local_homogeneity_mean_from_quantized(
            q_shuffle,
            win=win,
            direction=direction
        )
        hg_list.append(hg_i)

    hg = float(np.mean(hg_list))

    delta_h = 100.0 * abs(ho - hg) / max(abs(ho), 1e-12)

    return delta_h, ho, hg


def compute_first_order_residual_from_array(
    noisy_img,
    ratio_img,
    win=25,
    tol=0.03,
    strict=False,
    fallback_topk=100
):
    noisy_img = np.asarray(noisy_img, dtype=np.float64)
    ratio_img = np.asarray(ratio_img, dtype=np.float64)

    noisy_mean, noisy_var = local_mean_var_valid(noisy_img, win)
    ratio_mean, ratio_var = local_mean_var_valid(ratio_img, win)

    noisy_enl = enl_from_mean_var(noisy_mean, noisy_var)
    ratio_enl = enl_from_mean_var(ratio_mean, ratio_var)

    r_enl = np.abs(noisy_enl - ratio_enl) / np.maximum(noisy_enl, 1e-12)
    r_mu = np.abs(1.0 - ratio_mean)

    finite_mask = np.isfinite(r_enl) & np.isfinite(r_mu) & (noisy_enl > 0)
    sel_mask = finite_mask & (r_enl <= tol) & (r_mu <= tol)

    n_selected = int(np.sum(sel_mask))

    if n_selected == 0:
        if strict:
            raise RuntimeError(
                "没有找到满足论文容差条件的均匀窗口。"
                "你可以调大 tol，或设置 strict=False 使用回退策略。"
            )

        score = np.where(finite_mask, r_enl + r_mu, np.inf)
        flat_idx = np.argsort(score.ravel())
        flat_idx = flat_idx[np.isfinite(score.ravel()[flat_idx])]

        if len(flat_idx) == 0:
            raise RuntimeError("无法找到有效窗口，请检查输入图像。")

        topk = min(fallback_topk, len(flat_idx))
        sel_mask = np.zeros_like(score, dtype=bool)
        sel_mask.ravel()[flat_idx[:topk]] = True
        n_selected = int(np.sum(sel_mask))

    r_first = 0.5 * float(np.mean((r_enl + r_mu)[sel_mask]))

    return r_first, n_selected


def calc_m_metric(
    noisy,
    pred,
    win=25,
    tol=0.03,
    levels=8,
    p_shuffle=100,
    seed=42,
    direction="h",
    eps=1e-12,
    strict=False
):
    """
    论文定义的 M 指标。

    noisy: 原始含噪 SAR 图 Z
    pred:  去噪图 X_hat

    ratio = Z / X_hat
    M = r_ENL,mu + delta_h

    M 越小越好，理想值为 0。
    """

    noisy = np.asarray(noisy, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    if noisy.shape != pred.shape:
        raise ValueError(f"noisy 和 pred 尺寸不一致: {noisy.shape} vs {pred.shape}")

    if min(noisy.shape) < win:
        raise ValueError(f"图像尺寸 {noisy.shape} 小于窗口大小 {win}")

    ratio = noisy / np.maximum(pred, eps)

    r_first, n_selected = compute_first_order_residual_from_array(
        noisy_img=noisy,
        ratio_img=ratio,
        win=win,
        tol=tol,
        strict=strict
    )

    delta_h, ho, hg = compute_delta_h_from_array(
        ratio_img=ratio,
        win=win,
        levels=levels,
        p_shuffle=p_shuffle,
        seed=seed,
        direction=direction
    )

    m_value = r_first + delta_h

    return float(m_value)


def calc_epi(noisy, pred, eps=1e-8):
    """
    EPI: Edge Preservation Index
    用 Sobel 梯度相关性计算。
    越接近 1，边缘保持越好。
    """

    noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)
    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)

    gx1 = cv2.Sobel(noisy, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(noisy, cv2.CV_32F, 0, 1, ksize=3)
    edge_noisy = np.sqrt(gx1 ** 2 + gy1 ** 2)

    gx2 = cv2.Sobel(pred, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(pred, cv2.CV_32F, 0, 1, ksize=3)
    edge_pred = np.sqrt(gx2 ** 2 + gy2 ** 2)

    a = edge_noisy - np.mean(edge_noisy)
    b = edge_pred - np.mean(edge_pred)

    epi = np.sum(a * b) / (np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + eps)

    return float(epi)


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


def get_category_from_path(path):
    """
    根据文件名或路径判断类别。
    兼容：
    ./test_selected/all/01_homogeneous_xxx.mat
    ./test_selected/homogeneous/xxx.mat
    ./test_selected/structural/xxx.mat
    ./test_selected/texture/xxx.mat
    """
    path_lower = path.lower()
    filename_lower = os.path.basename(path).lower()
    parent_lower = os.path.basename(os.path.dirname(path)).lower()

    text = path_lower + " " + filename_lower + " " + parent_lower

    if "homogeneous" in text:
        return "homogeneous"

    if "structural" in text:
        return "structural"

    if "texture" in text:
        return "texture"

    # 如果没有识别出来，放到 unknown，避免程序崩
    return "unknown"


def write_metrics_csv(csv_path, rows):
    """
    保存逐图指标。
    """
    if len(rows) == 0:
        return

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'category',
            'file',
            'model',
            'noisy_enl',
            'pred_enl',
            'm_paper',
            'epi'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def summarize_rows(rows, name):
    """
    计算某一组 rows 的平均指标。
    """
    if len(rows) == 0:
        return {
            'category': name,
            'num_images': 0,
            'avg_noisy_enl': 0.0,
            'avg_pred_enl': 0.0,
            'avg_m_paper': 0.0,
            'avg_epi': 0.0
        }

    return {
        'category': name,
        'num_images': len(rows),
        'avg_noisy_enl': float(np.mean([r['noisy_enl'] for r in rows])),
        'avg_pred_enl': float(np.mean([r['pred_enl'] for r in rows])),
        'avg_m_paper': float(np.mean([r['m_paper'] for r in rows])),
        'avg_epi': float(np.mean([r['epi'] for r in rows]))
    }


def write_summary_csv(csv_path, summaries):
    """
    保存三类场景 + 总平均指标。
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'category',
            'num_images',
            'avg_noisy_enl',
            'avg_pred_enl',
            'avg_m_paper',
            'avg_epi'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summaries:
            writer.writerow(row)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ensure_dir(args.save_path)

    model = build_model(args.model, args.loadmodel, device)

    files = collect_files(args.dataset)

    if len(files) == 0:
        raise RuntimeError(f"测试路径中没有找到图像或 .mat 文件: {args.dataset}")

    print(f"Found {len(files)} test files.")

    # 三类结果分别存放
    category_names = ["homogeneous", "structural", "texture", "unknown"]

    category_rows = {
        "homogeneous": [],
        "structural": [],
        "texture": [],
        "unknown": []
    }

    all_rows = []

    # 创建类别输出文件夹
    for cat in category_names:
        cat_dir = os.path.join(args.save_path, cat)
        ensure_dir(cat_dir)

    for idx, path in enumerate(files):
        category = get_category_from_path(path)

        print(f"[{idx + 1}/{len(files)}] Testing: {path}")
        print(f"  Category: {category}")

        noisy = load_real_sar(path)

        x = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)

        pred = pred.detach().cpu().numpy()
        pred = np.squeeze(pred)
        pred = np.clip(pred, 0.0, 1.0)

        residual = np.abs(noisy - pred)

        enl_noisy = calc_enl(noisy, roi_size=args.roi_size, num_rois=args.num_rois)
        enl_pred = calc_enl(pred, roi_size=args.roi_size, num_rois=args.num_rois)

        m_value = calc_m_metric(
            noisy,
            pred,
            win=25,
            tol=0.03,
            levels=8,
            p_shuffle=100,
            seed=42,
            direction="h",
            strict=False
        )

        epi_value = calc_epi(noisy, pred)

        base_name = os.path.splitext(os.path.basename(path))[0]

        # 保存到对应类别文件夹
        category_save_dir = os.path.join(args.save_path, category)
        ensure_dir(category_save_dir)

        save_prefix = f"{len(category_rows[category]) + 1:02d}_{base_name}"

        save_gray(
            os.path.join(category_save_dir, f"{save_prefix}_noisy.png"),
            noisy
        )

        save_gray(
            os.path.join(category_save_dir, f"{save_prefix}_{args.model}.png"),
            pred
        )

        save_gray(
            os.path.join(category_save_dir, f"{save_prefix}_residual.png"),
            residual
        )

        save_compare_image(
            os.path.join(category_save_dir, f"{save_prefix}_compare.png"),
            noisy,
            pred,
            residual,
            titles=["Noisy", args.model, "Residual"]
        )

        row = {
            'category': category,
            'file': path,
            'model': args.model,
            'noisy_enl': enl_noisy,
            'pred_enl': enl_pred,
            'm_paper': m_value,
            'epi': epi_value
        }

        category_rows[category].append(row)
        all_rows.append(row)

        print(f"  Noisy ENL: {enl_noisy:.4f}")
        print(f"  Pred  ENL: {enl_pred:.4f}")
        print(f"  M_paper:   {m_value:.6f}")
        print(f"  EPI:       {epi_value:.6f}")

    # 1. 保存总 CSV
    all_csv_path = os.path.join(args.save_path, 'real_sar_metrics_all.csv')
    write_metrics_csv(all_csv_path, all_rows)

    # 2. 保存每类 CSV
    for cat in category_names:
        if len(category_rows[cat]) == 0:
            continue

        cat_csv_path = os.path.join(
            args.save_path,
            cat,
            f"real_sar_metrics_{cat}.csv"
        )

        write_metrics_csv(cat_csv_path, category_rows[cat])

    # 3. 计算每类平均值 + 总平均值
    summaries = []

    for cat in ["homogeneous", "structural", "texture"]:
        summary = summarize_rows(category_rows[cat], cat)
        summaries.append(summary)

    if len(category_rows["unknown"]) > 0:
        summaries.append(summarize_rows(category_rows["unknown"], "unknown"))

    summaries.append(summarize_rows(all_rows, "all"))

    # 4. 保存 summary CSV
    summary_csv_path = os.path.join(args.save_path, 'real_sar_summary_by_category.csv')
    write_summary_csv(summary_csv_path, summaries)

    # 5. 控制台打印结果
    print("\n========== Real SAR Results by Category ==========")

    for s in summaries:
        print(f"\n[{s['category']}]")
        print(f"Num Images:     {s['num_images']}")
        print(f"Noisy Avg ENL:  {s['avg_noisy_enl']:.4f}")
        print(f"Pred  Avg ENL:  {s['avg_pred_enl']:.4f}")
        print(f"Avg M_paper:    {s['avg_m_paper']:.6f}")
        print(f"Avg EPI:        {s['avg_epi']:.6f}")

    print("\nDone.")
    print("Results saved to:", args.save_path)
    print("All CSV saved to:", all_csv_path)
    print("Summary CSV saved to:", summary_csv_path)


if __name__ == '__main__':
    main()