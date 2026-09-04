from scipy.io import loadmat, savemat
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


seed = np.random.RandomState(112311)


def read_image_to_gray(image_path, force_size=(256, 256)):
    """
    读取图片并转为灰度图，归一化到 [0, 1]
    """
    img = Image.open(image_path).convert("RGB")

    if force_size is not None:
        img = img.resize(force_size, Image.BILINEAR)

    arr = np.array(img).astype(np.float32)

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    gray = gray / 255.0
    gray = np.clip(gray, 0.0, 1.0)

    return gray.astype(np.float32)


def plot_clean_noisy_img(filepath):
    """
    查看生成的 .mat 文件
    注意：clean/noisy 保存的是强度域，所以显示时开平方
    """
    dt = loadmat(filepath)

    clean_img = dt["clean"]
    noisy_img = dt["noisy"]

    clean_img = np.sqrt(clean_img)
    noisy_img = np.sqrt(noisy_img)

    plt.figure()
    plt.imshow(clean_img * 255, cmap="gray", vmin=0, vmax=255)
    plt.title("Clean")
    plt.axis("off")

    plt.figure()
    plt.imshow(noisy_img * 255, cmap="gray", vmin=0, vmax=255)
    plt.title("Noisy")
    plt.axis("off")

    plt.show()


def gradient_magnitude(img):
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)

    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    gx[:, 0] = img[:, 1] - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]

    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    gy[0, :] = img[1, :] - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]

    return np.sqrt(gx * gx + gy * gy)


def dilate_mask(mask, iterations=1):
    out = mask.copy()
    h, w = mask.shape

    for _ in range(iterations):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        new_mask = np.zeros_like(out, dtype=bool)

        for dy in range(3):
            for dx in range(3):
                new_mask |= padded[dy:dy + h, dx:dx + w]

        out = new_mask

    return out


def gamma_speckle(shape, L, rng):
    """
    均值为 1 的 Gamma 乘性斑点噪声
    L 越小，斑点噪声越强
    """
    return rng.gamma(
        shape=L,
        scale=1.0 / L,
        size=shape
    ).astype(np.float32)


def build_region_L_map(h, w, grid=(4, 4), L_choices=(1, 2, 4, 8), rng=None):
    """
    构造空间非均匀 L 图，使不同区域拥有不同强度的 speckle
    """
    if rng is None:
        rng = np.random.default_rng()

    L_map = np.zeros((h, w), dtype=np.float32)

    hs = np.linspace(0, h, grid[0] + 1, dtype=int)
    ws = np.linspace(0, w, grid[1] + 1, dtype=int)

    for i in range(grid[0]):
        for j in range(grid[1]):
            L = rng.choice(L_choices)
            L_map[hs[i]:hs[i + 1], ws[j]:ws[j + 1]] = L

    return L_map


def apply_spatially_varying_speckle(img, L_map, rng):
    """
    根据 L_map 对不同区域施加不同强度的乘性斑点噪声
    """
    noisy = np.zeros_like(img, dtype=np.float32)
    unique_L = np.unique(L_map).astype(np.int32)

    for L in unique_L:
        mask = L_map == L
        n = gamma_speckle((mask.sum(),), L=L, rng=rng)
        noisy[mask] = img[mask] * n

    return np.clip(noisy, 0.0, 1.0)


def add_registration_stripe_system_noise(
    img,
    misreg_alpha=0.08,
    shift_xy=(1, 2),
    stripe_amp=0.02,
    row_prob=0.08,
    col_prob=0.08,
    rng=None
):
    """
    添加配准误差、行列条纹、低频系统噪声
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = img.shape
    dy, dx = shift_xy

    shifted = np.roll(img, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)

    noisy = (1.0 - misreg_alpha) * img + misreg_alpha * shifted

    row_offsets = np.zeros((h, 1), dtype=np.float32)
    col_offsets = np.zeros((1, w), dtype=np.float32)

    row_mask = rng.random(h) < row_prob
    col_mask = rng.random(w) < col_prob

    if row_mask.any():
        row_offsets[row_mask, 0] = rng.normal(
            0.0,
            stripe_amp,
            size=row_mask.sum()
        ).astype(np.float32)

    if col_mask.any():
        col_offsets[0, col_mask] = rng.normal(
            0.0,
            stripe_amp,
            size=col_mask.sum()
        ).astype(np.float32)

    rows = np.arange(h, dtype=np.float32)
    cols = np.arange(w, dtype=np.float32)

    row_band = 0.5 * stripe_amp * np.sin(
        2 * np.pi * rows / max(16, h // 4)
    )
    col_band = 0.5 * stripe_amp * np.sin(
        2 * np.pi * cols / max(16, w // 5)
    )

    row_offsets += row_band[:, None]
    col_offsets += col_band[None, :]

    noisy = noisy + row_offsets + col_offsets

    return np.clip(noisy, 0.0, 1.0)


def simulate_complex_sar_like(
    img,
    grid=(4, 4),
    region_L_choices=(1, 2, 4, 8),
    strong_L=1,
    bright_percentile=90,
    edge_percentile=85,
    target_gain=0.20,
    impulse_prob=0.008,
    sigma_add=0.015,
    misreg_alpha=0.08,
    shift_xy=(1, 2),
    stripe_amp=0.02,
    rng=None
):
    """
    综合 SAR 模拟噪声：

    1. 空间非均匀乘性斑点噪声
    2. 强散射目标区域增强
    3. 高亮脉冲点
    4. 加性热噪声
    5. 配准误差、条纹、系统噪声
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = img.shape

    grad = gradient_magnitude(img)

    bright_thr = np.percentile(img, bright_percentile)
    edge_thr = np.percentile(grad, edge_percentile)

    strong_mask = (img >= bright_thr) | (grad >= edge_thr)
    strong_mask = dilate_mask(strong_mask, iterations=1)

    L_map = build_region_L_map(
        h,
        w,
        grid=grid,
        L_choices=region_L_choices,
        rng=rng
    )

    L_map[strong_mask] = strong_L

    noisy = apply_spatially_varying_speckle(img, L_map, rng)

    noisy[strong_mask] = noisy[strong_mask] * (1.0 + target_gain)

    impulse_mask = strong_mask & (rng.random((h, w)) < impulse_prob)

    if impulse_mask.any():
        noisy[impulse_mask] = noisy[impulse_mask] * rng.uniform(
            1.1,
            1.6,
            size=impulse_mask.sum()
        ).astype(np.float32)

    noisy = np.clip(noisy, 0.0, 1.0)

    noisy = noisy + rng.normal(
        0.0,
        sigma_add,
        size=(h, w)
    ).astype(np.float32)

    noisy = np.clip(noisy, 0.0, 1.0)

    noisy = add_registration_stripe_system_noise(
        noisy,
        misreg_alpha=misreg_alpha,
        shift_xy=shift_xy,
        stripe_amp=stripe_amp,
        row_prob=0.08,
        col_prob=0.08,
        rng=rng
    )

    return np.clip(noisy, 0.0, 1.0)


def save_one_mat(
    image_path,
    mat_save_path,
    force_size=(256, 256),
    rng=None,
    grid=(4, 4),
    region_L_choices=(1, 2, 4, 8),
    strong_L=1,
    bright_percentile=90,
    edge_percentile=85,
    target_gain=0.20,
    impulse_prob=0.008,
    sigma_add=0.015,
    misreg_alpha=0.08,
    shift_xy=(1, 2),
    stripe_amp=0.02
):
    """
    读取单张图像，生成 clean/noisy，并保存为 .mat

    注意：
    这里为了和你原来的代码保持一致，
    clean 和 noisy 保存的是强度域数据，即 gray ** 2。
    训练或可视化时如果需要显示图像，可以 np.sqrt()。
    """
    gray = read_image_to_gray(image_path, force_size=force_size)

    clean = np.square(gray).astype(np.float32)

    noisy = simulate_complex_sar_like(
        clean,
        grid=grid,
        region_L_choices=region_L_choices,
        strong_L=strong_L,
        bright_percentile=bright_percentile,
        edge_percentile=edge_percentile,
        target_gain=target_gain,
        impulse_prob=impulse_prob,
        sigma_add=sigma_add,
        misreg_alpha=misreg_alpha,
        shift_xy=shift_xy,
        stripe_amp=stripe_amp,
        rng=rng
    )

    noisy = noisy.astype(np.float32)

    dat = dict()
    dat["clean"] = clean
    dat["noisy"] = noisy

    os.makedirs(os.path.dirname(mat_save_path), exist_ok=True)
    savemat(mat_save_path, dat)


def generate_syn_dataset(
    source_root,
    savepath,
    force_size=(256, 256),
    base_seed=42,
    grid=(4, 4),
    region_L_choices=(1, 2, 4, 8),
    strong_L=1,
    bright_percentile=90,
    edge_percentile=85,
    target_gain=0.20,
    impulse_prob=0.008,
    sigma_add=0.015,
    misreg_alpha=0.08,
    shift_xy=(1, 2),
    stripe_amp=0.02
):
    """
    从 source_root 下的 train/val 图像生成模拟 SAR .mat 数据集
    """
    save_path_train = os.path.join(savepath, "train")
    save_path_val = os.path.join(savepath, "val")

    os.makedirs(save_path_train, exist_ok=True)
    os.makedirs(save_path_val, exist_ok=True)

    supported_exts = (
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff", ".webp"
    )

    path_train = os.path.join(source_root, "train")
    path_val = os.path.join(source_root, "val")

    i = 0

    # =========================
    # 1. train 文件夹 -> train
    # =========================
    if os.path.exists(path_train):
        for file in sorted(os.listdir(path_train)):
            if not file.lower().endswith(supported_exts):
                continue

            i += 1
            print(f"[train] {i}: {file}")

            im_file = os.path.join(path_train, file)
            file_base = os.path.splitext(file)[0]

            data_save_path = os.path.join(
                save_path_train,
                "trn_" + file_base + ".mat"
            )

            rng = np.random.default_rng(base_seed + i)

            save_one_mat(
                image_path=im_file,
                mat_save_path=data_save_path,
                force_size=force_size,
                rng=rng,
                grid=grid,
                region_L_choices=region_L_choices,
                strong_L=strong_L,
                bright_percentile=bright_percentile,
                edge_percentile=edge_percentile,
                target_gain=target_gain,
                impulse_prob=impulse_prob,
                sigma_add=sigma_add,
                misreg_alpha=misreg_alpha,
                shift_xy=shift_xy,
                stripe_amp=stripe_amp
            )

    # =========================
    # 2. val 文件夹 -> val
    # =========================
    if os.path.exists(path_val):
        for file in sorted(os.listdir(path_val)):
            if not file.lower().endswith(supported_exts):
                continue
            im_file = os.path.join(path_val, file)
            file_base = os.path.splitext(file)[0]

            i += 1
            rng = np.random.default_rng(base_seed + i)

            print(f"[val] {i}: {file}")
            data_save_path = os.path.join(
                save_path_val,
                "val_" + file_base + ".mat"
            )

            save_one_mat(
                image_path=im_file,
                mat_save_path=data_save_path,
                force_size=force_size,
                rng=rng,
                grid=grid,
                region_L_choices=region_L_choices,
                strong_L=strong_L,
                bright_percentile=bright_percentile,
                edge_percentile=edge_percentile,
                target_gain=target_gain,
                impulse_prob=impulse_prob,
                sigma_add=sigma_add,
                misreg_alpha=misreg_alpha,
                shift_xy=shift_xy,
                stripe_amp=stripe_amp
            )

    print("\n数据集生成完成")
    print(f"总生成数量: {i}")
    print(f"训练集路径: {save_path_train}")
    print(f"验证集路径: {save_path_val}")


if __name__ == "__main__":

    source_root = "./datasets/UCMerced_LandUse"
    savepath = "./datasets/UCMerced_LandUse_SAR_mat/"

    generate_syn_dataset(
        source_root=source_root,
        savepath=savepath,
        force_size=(256, 256),

        # 综合 SAR 模拟噪声参数
        base_seed=42,
        grid=(4, 4),
        region_L_choices=(1, 2, 4, 8),
        strong_L=1,
        bright_percentile=90,
        edge_percentile=85,
        target_gain=0.20,
        impulse_prob=0.008,
        sigma_add=0.015,
        misreg_alpha=0.08,
        shift_xy=(1, 2),
        stripe_amp=0.02
    )