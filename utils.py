import os
from pathlib import Path
import numpy as np
import torch
import scipy.io as sio

# from skimage import io,color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable
import os
import cv2
import pandas as pd

from numbers import Number
from typing import Container
from collections import defaultdict

from scipy.io import loadmat

from numeric_domain import (
    INTENSITY_DOMAIN,
    LEGACY_AMPLITUDE_DOMAIN,
    normalize_real_intensity,
    prepare_synthetic_pair,
    validate_numeric_domain,
)


def normalize_real_sar(img, low_percentile=1, high_percentile=99, use_sqrt=False):
    """Compatibility wrapper around the canonical real-SAR normalization."""
    img = normalize_real_intensity(img, low_percentile, high_percentile)
    if use_sqrt:
        img = np.sqrt(img)
    return img.astype(np.float32)


class BSD_SAR(Dataset):
    """
    Reads the synthetic images (created useing create_synthetic_data.py) saved as .mat files .
    """

    def __init__(
        self,
        dataset_path,
        crop_size,
        training_set=True,
        numeric_domain=INTENSITY_DOMAIN,
        manifest_path=None,
        split=None,
    ) -> None:
        self.dataset_path = os.path.abspath(dataset_path)
        # self.input_path = os.path.join(dataset_path, 'noisy')
        # self.output_path = os.path.join(dataset_path, 'clean')
        if manifest_path is not None:
            if split not in ("train", "val", "test"):
                raise ValueError("split must be train/val/test when manifest_path is provided")
            import json
            from synthetic_manifest import verify_paired_sar_manifest

            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            verify_paired_sar_manifest(
                Path(self.dataset_path),
                manifest,
                required_splits=(split,),
                verify_hashes=False,
                require_exact_partition=False,
                verify_files=False,
            )
            # Preserve manifest order.  Controlled protocols may intentionally
            # interleave fixed degradation groups for deterministic smoke runs.
            self.images_list = list(manifest["files"][split])
        else:
            self.images_list = sorted(
                name for name in os.listdir(self.dataset_path) if name.lower().endswith(".mat")
            )
        self.training_set = training_set
        self.numeric_domain = validate_numeric_domain(numeric_domain)

        self.crop = crop_size
        


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        

        # read .dat file
        data_SAR = loadmat(os.path.join(self.dataset_path, image_filename))

        # get noisy image and numpy to tensor
        image, mask = prepare_synthetic_pair(
            data_SAR['noisy'], data_SAR['clean'], self.numeric_domain
        )
        # Mode F preserves float32 intensity values.  The default mode converts
        # float input to 8-bit L, which made validation differ from direct MAT
        # evaluation even with the same checkpoint and crop.
        image = F.to_pil_image(image, mode="F")
        
    

        # get clean image and numpy to tensor
        mask = F.to_pil_image(mask, mode="F")
        
        # print(image.shape)
        # print(mask.shape)

        # # read noisy image
        # image = cv2.imread(os.path.join(self.input_path, image_filename),0)
        
        # # read clean image
        # mask = cv2.imread(os.path.join(self.output_path, image_filename),0)
        
        # # transforming to PIL image
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        if self.training_set:
            # # random resized crop
            # i, j, h, w = T.RandomResizedCrop.get_params(image,scale= (0.12, 1.0), ratio=(1, 1))
            # image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

            # random crop
            i, j, h, w = T.RandomCrop.get_params(image, output_size= self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        

            # # resize
            # image, mask = F.resize(image,self.crop), F.resize(mask,self.crop)

            # rotation
            a = T.RandomRotation.get_params((-90, 90))
            image, mask = F.rotate(image, a), F.rotate(mask, a)


            # random horizontal flipping
            if np.random.rand() < 0.5:
                image, mask = F.hflip(image), F.hflip(mask)

            # random affine transform
            if np.random.rand() < 0.5:
                affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
                image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        else:
            # Validation is deterministic: use the centered crop rather than a random crop.
            image_width, image_height = image.size
            crop_height, crop_width = self.crop
            if image_height < crop_height or image_width < crop_width:
                raise ValueError(
                    f"Image {image_filename} is smaller than requested crop {self.crop}"
                )
            i = (image_height - crop_height) // 2
            j = (image_width - crop_width) // 2
            h, w = crop_height, crop_width
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)


        # transforming to tensor
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        

        return image, mask, image_filename


class RealSARDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, manifest_path=None, split=None):
        super(RealSARDataset, self).__init__()
        self.root_dir = root_dir

        if manifest_path is not None:
            if split not in ("train", "val", "test"):
                raise ValueError("split must be train/val/test when manifest_path is provided")
            import json
            from pathlib import Path
            from resplit_real_sar_dataset import verify_manifest

            dataset_root = Path(root_dir).resolve()
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            verify_manifest(dataset_root, manifest)
            all_files = [str(dataset_root / path) for path in manifest["files"][split]]
        else:
            all_files = []
            for name in os.listdir(root_dir):
                if name.endswith('.mat'):
                    all_files.append(os.path.join(root_dir, name))

        all_files = sorted(all_files)

        self.file_list = []
        skipped_black = 0
        skipped_invalid = 0

        for path in all_files:
            try:
                data = sio.loadmat(path)
            except Exception:
                skipped_invalid += 1
                continue

            if 'noisy' in data:
                noisy = data['noisy']
            elif 'sar' in data:
                noisy = data['sar']
            elif 'image' in data:
                noisy = data['image']
            else:
                skipped_invalid += 1
                continue

            noisy = noisy.astype(np.float32)
            if noisy.ndim == 3:
                noisy = np.squeeze(noisy)

            # Skip all-black / constant images (no valid signal for normalization).
            if float(noisy.max()) <= float(noisy.min()):
                skipped_black += 1
                continue

            self.file_list.append(path)

        print(
            "Real SAR dataset size: {} (from {} files, skipped_black={}, skipped_invalid={})".format(
                len(self.file_list),
                len(all_files),
                skipped_black,
                skipped_invalid,
            )
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        data = sio.loadmat(path)

        if 'noisy' in data:
            noisy = data['noisy']
        elif 'sar' in data:
            noisy = data['sar']
        elif 'image' in data:
            noisy = data['image']
        else:
            raise KeyError(
                "真实 SAR mat 文件中没有找到 noisy / sar / image 字段: {}".format(path)
            )

        noisy = noisy.astype(np.float32)

        if noisy.ndim == 3:
            noisy = noisy.squeeze()
            noisy = normalize_real_sar(noisy)
            noisy = noisy[None, :, :]
        else:
            noisy = normalize_real_sar(noisy)
            noisy = noisy[None, :, :]

        noisy = torch.from_numpy(noisy).float()
        return noisy
