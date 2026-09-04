import os
from datasets import load_dataset
from tqdm import tqdm


def main() -> None:
    save_root = "./datasets/NWPU-RESISC45"

    # 下载 Hugging Face 镜像版 NWPU-RESISC45
    dataset = load_dataset("jonathan-roberts1/NWPU-RESISC45")

    # 兼容可能存在的不同 split 名称
    if "train" in dataset:
        ds = dataset["train"]
    else:
        first_split = list(dataset.keys())[0]
        ds = dataset[first_split]

    label_names = ds.features["label"].names
    os.makedirs(save_root, exist_ok=True)

    for i, item in enumerate(tqdm(ds, desc="Exporting images")):
        img = item["image"]
        label_id = item["label"]
        label_name = label_names[label_id]

        class_dir = os.path.join(save_root, label_name)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, f"{label_name}_{i:05d}.jpg")
        img.convert("RGB").save(save_path)

    print("下载并导出完成:", os.path.abspath(save_root))


if __name__ == "__main__":
    main()
