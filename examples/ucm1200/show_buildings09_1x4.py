from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    base = root / "test_results" / "ucmerced_3scenes" / "TransSARV2_DualFreqNG_Bottle" / "buildings"
    output_dir = root / "examples" / "ucm1200"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        ("noisy", base / "trn_buildings09_noisy.png"),
        ("clean", base / "trn_buildings09_clean.png"),
        ("residual", base / "trn_buildings09_residual.png"),
        ("result", base / "trn_buildings09_TransSARV2_DualFreqNG_Bottle.png"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (label, path) in zip(axes, images):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")

    fig.subplots_adjust(wspace=0.02, hspace=0.0)
    save_path = output_dir / "trn_buildings09_1x4_labeled.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
