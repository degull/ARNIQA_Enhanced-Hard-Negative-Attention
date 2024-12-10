""" import pandas as pd
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import ImageFilter
import random

# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "gaussian_blur",
    2: "lens_blur",
    3: "motion_blur",
    4: "color_diffusion",
    5: "color_shift",
    6: "color_quantization",
    7: "color_saturation_1",
    8: "color_saturation_2",
    9: "jpeg2000",
    10: "jpeg",
    11: "white_noise",
    12: "white_noise_color_component",
    13: "impulse_noise",
    14: "multiplicative_noise",
    15: "denoise",
    16: "brighten",
    17: "darken",
    18: "mean_shift",
    19: "jitter",
    20: "non_eccentricity_patch",
    21: "pixelate",
    22: "quantization",
    23: "color_block",
    24: "high_sharpen",
    25: "contrast_change"
}

class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", split_idx: int = 0, crop_size: int = 224):
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        # CSV 파일에서 점수 로드
        scores_csv = pd.read_csv(self.root / "kadid10k.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # 이미지 경로 설정
        self.images = np.array([self.root / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([self.root / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        self.distortion_types = []
        self.distortion_levels = []

        for img in self.images:
            # 이미지 이름에서 왜곡 유형과 레벨 추출
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(img))
            if match:
                dist_type = distortion_types_mapping[int(match.group(1))]
                self.distortion_types.append(dist_type)
                self.distortion_levels.append(int(match.group(2)))

        self.distortion_types = np.array(self.distortion_types)
        self.distortion_levels = np.array(self.distortion_levels)

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # 패딩 제거
            self.images = self.images[split_idxs]
            self.ref_images = self.ref_images[split_idxs]
            self.mos = self.mos[split_idxs]
            self.distortion_types = self.distortion_types[split_idxs]
            self.distortion_levels = self.distortion_levels[split_idxs]

    def transform(self, image: Image) -> torch.Tensor:
        # Transform image to desired size and convert to tensor
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)
    
    def apply_distortion(self, image: torch.Tensor) -> torch.Tensor:
        # Convert tensor to PIL image
        pil_image = transforms.ToPILImage()(image)
        
        # Apply a random distortion (e.g., Gaussian Blur)
        if random.random() > 0.5:
            return transforms.ToTensor()(pil_image.filter(ImageFilter.GaussianBlur(radius=2)))
        
        return image  # Return original image if no distortion applied


    def __getitem__(self, index: int) -> dict:
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")

        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)

        # Create crops and stack images
        crops_A = [img_A_orig]
        crops_B = [img_B_orig]

        # Apply additional crops
        crops_A += [self.apply_distortion(img_A_orig) for _ in range(3)]
        crops_B += [self.apply_distortion(img_B_orig) for _ in range(3)]

        # Stack crops
        img_A = torch.stack(crops_A)  # Shape: [num_crops, 3, crop_size, crop_size]
        img_B = torch.stack(crops_B)  # Shape: [num_crops, 3, crop_size, crop_size]

        # Reshape to [1, num_crops, 3, crop_size, crop_size]
        img_A = img_A.unsqueeze(0)
        img_B = img_B.unsqueeze(0)

        return {
            "img_A_orig": img_A,
            "img_B_orig": img_B,
            "img_A_ds": img_A,  # img_A_ds를 추가합니다
            "img_B_ds": img_B,  # img_B_ds를 추가합니다
            "mos": self.mos[index],
        }

    def __len__(self):
        return len(self.images)

    def get_split_indices(self, split: int, phase: str) -> np.ndarray:
        split_file_path = self.root / "splits" / f"{phase}.npy"
        split_indices = np.load(split_file_path)[split]
        return split_indices
 """

import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import random
from PIL import Image, ImageFilter

# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "gaussian_blur",
    2: "lens_blur",
    3: "motion_blur",
    4: "color_diffusion",
    5: "color_shift",
    6: "color_quantization",
    7: "color_saturation_1",
    8: "color_saturation_2",
    9: "jpeg2000",
    10: "jpeg",
    11: "white_noise",
    12: "white_noise_color_component",
    13: "impulse_noise",
    14: "multiplicative_noise",
    15: "denoise",
    16: "brighten",
    17: "darken",
    18: "mean_shift",
    19: "jitter",
    20: "non_eccentricity_patch",
    21: "pixelate",
    22: "quantization",
    23: "color_block",
    24: "high_sharpen",
    25: "contrast_change"
}

class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", split_idx: int = 0, crop_size: int = 224):
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        # CSV 파일에서 점수 로드
        scores_csv = pd.read_csv(self.root)  # 수정된 부분
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # 이미지 경로 설정
        self.images = np.array([self.root.parent / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([self.root.parent / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        self.distortion_types = []
        self.distortion_levels = []

        for img in self.images:
            # 이미지 이름에서 왜곡 유형과 레벨 추출
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(img))
            if match:
                dist_type = distortion_types_mapping[int(match.group(1))]
                self.distortion_types.append(dist_type)
                self.distortion_levels.append(int(match.group(2)))

        self.distortion_types = np.array(self.distortion_types)
        self.distortion_levels = np.array(self.distortion_levels)

        if self.phase != "all":
            split_idxs = np.load(self.root.parent / "splits" / f"{self.phase}.npy")[split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # 패딩 제거
            self.images = self.images[split_idxs]
            self.ref_images = self.ref_images[split_idxs]
            self.mos = self.mos[split_idxs]
            self.distortion_types = self.distortion_types[split_idxs]
            self.distortion_levels = self.distortion_levels[split_idxs]

    def transform(self, image: Image, size: tuple = None) -> torch.Tensor:
        # 기본 크기 설정
        size = size or (self.crop_size, self.crop_size)
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])(image)
    
    def apply_distortion(self, image: torch.Tensor, exclude: list = None) -> torch.Tensor:
        exclude = exclude or []
        pil_image = transforms.ToPILImage()(image)
        distortions = [
            lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),  # Gaussian Blur
            lambda img: img.filter(ImageFilter.BoxBlur(1)),  # Box Blur
            lambda img: img.rotate(15),  # Rotate
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),  # Horizontal Flip
            lambda img: img.filter(ImageFilter.SHARPEN),  # Sharpen
        ]
        # 적용되지 말아야 할 왜곡 제외
        available_distortions = [dist for i, dist in enumerate(distortions) if i not in exclude]
        distortion = random.choice(available_distortions)
        distorted_image = distortion(pil_image)
        return transforms.ToTensor()(distorted_image)


    def __getitem__(self, index: int) -> dict:
        # Anchor 이미지로 distorted image 사용
        img_anchor = Image.open(self.images[index]).convert("RGB")
        img_positive = Image.open(self.ref_images[index]).convert("RGB")
        # Negative 이미지는 다른 index의 ref_images에서 선택
        negative_index = random.choice([i for i in range(len(self.ref_images)) if i != index])
        # 예를 들어, 다음 이미지로 선택
        img_negative = Image.open(self.ref_images[negative_index]).convert("RGB")

        img_anchor = self.transform(img_anchor)
        img_positive = self.transform(img_positive)
        # Hard Negative는 예상 크기로 리사이즈
        img_negative = self.transform(img_negative, size=(112, 112))

        # Distortion 추가
        img_anchor = self.apply_distortion(img_anchor)
        img_positive = self.apply_distortion(img_positive)
        img_negative = self.apply_distortion(img_negative, exclude=[0])  # 앵커와 다른 왜곡 적용


        return {
            "img_anchor": img_anchor,
            "img_positive": img_positive,
            "img_negative": img_negative,
            "mos": self.mos[index],
            "path": str(self.images[index]),  # 이미지 경로 추가
            "distortion_type": self.distortion_types[index],  # 왜곡 유형 추가
            "distortion_level": self.distortion_levels[index],  # 왜곡 레벨 추가
        }

    def __len__(self):
        return len(self.images)

    def get_split_indices(self, split: int, phase: str) -> np.ndarray:
        split_file_path = self.root.parent / "splits" / f"{phase}.npy"
        split_indices = np.load(split_file_path)[split]
        return split_indices