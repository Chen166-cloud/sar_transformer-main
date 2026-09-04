#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _resize_to_match_height(images: list[Image.Image]) -> list[Image.Image]:
    target_height = min(img.height for img in images)
    resized: list[Image.Image] = []
    for img in images:
        if img.height == target_height:
            resized.append(img)
            continue
        new_width = int(round(img.width * (target_height / img.height)))
        resized.append(img.resize((new_width, target_height), Image.Resampling.LANCZOS))
    return resized


def make_triplet(noisy_path: Path, residual_path: Path, result_path: Path, output_path: Path) -> None:
    labels = ["noisy", "residual", "result"]
    image_paths = [noisy_path, residual_path, result_path]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    images = _resize_to_match_height(images)

    font_size = max(18, min(48, images[0].height // 18))
    font = _get_font(font_size)
    top_bar_height = int(font_size * 1.8)
    gap = 10

    total_width = sum(img.width for img in images) + gap * (len(images) - 1)
    total_height = top_bar_height + images[0].height
    canvas = Image.new("RGB", (total_width, total_height), color="white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for label, img in zip(labels, images):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x + (img.width - text_w) // 2
        text_y = (top_bar_height - text_h) // 2
        draw.text((text_x, text_y), label, fill="black", font=font)
        canvas.paste(img, (x, top_bar_height))
        x += img.width + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose noisy/residual/result in one labeled figure.")
    parser.add_argument(
        "--noisy",
        type=Path,
        default=Path(
            "test_results/real_sar/TransSARV2_DualFreqNG_Bottle_AMS/texture/"
            "05_45_texture_4608_3584_y0_x256_noisy.png"
        ),
    )
    parser.add_argument(
        "--residual",
        type=Path,
        default=Path(
            "test_results/real_sar/TransSARV2_DualFreqNG_Bottle_AMS/texture/"
            "05_45_texture_4608_3584_y0_x256_residual.png"
        ),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(
            "test_results/real_sar/TransSARV2_DualFreqNG_Bottle_AMS/texture/"
            "05_45_texture_4608_3584_y0_x256_TransSARV2_DualFreqNG_Bottle.png"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "examples/realsar/"
            "05_45_texture_4608_3584_y0_x256_TransSARV2_DualFreqNG_Bottle_AMS_triplet_labeled.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_triplet(args.noisy, args.residual, args.result, args.output)


if __name__ == "__main__":
    main()
