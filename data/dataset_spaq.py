""" import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from data.dataset_base_iqa import IQADataset

# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def center_corners_crop(image: Image, crop_size: int) -> list:
    width, height = image.size
    crops = []

    # Center crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    crops.append(image.crop((left, top, left + crop_size, top + crop_size)))

    # Four corners crop
    crops.append(image.crop((0, 0, crop_size, crop_size)))  # Top-left
    crops.append(image.crop((width - crop_size, 0, width, crop_size)))  # Top-right
    crops.append(image.crop((0, height - crop_size, crop_size, height)))  # Bottom-left
    crops.append(image.crop((width - crop_size, height - crop_size, width, height)))  # Bottom-right

    return crops

class SPAQDataset(IQADataset):
    def __init__(self, root: str, phase: str = "train", split_idx: int = 0, crop_size: int = 224):
        super().__init__(root, phase, split_idx, crop_size)

        # Load scores
        print("Loading scores from Excel...")
        scores_csv = pd.read_excel(self.root / "Annotations" / "MOS and Image attribute scores.xlsx")
        print("Scores CSV head:", scores_csv.head())  # Debugging output

        self.images = scores_csv["Image name"].values.tolist()
        self.images = np.array([self.root / "TestImage" / el for el in self.images])
        self.mos = np.array(scores_csv["MOS"].values.tolist())

        print(f"Loaded {len(self.images)} images with corresponding MOS scores.")

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))
            self.images = self.images[split_idxs]
            self.mos = self.mos[split_idxs]

    def __getitem__(self, index: int) -> dict:
        try:
            img_A_orig = Image.open(self.images[index]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {self.images[index]}: {e}")
            return {}

        img_A_orig = img_A_orig.resize((self.crop_size, self.crop_size), Image.BICUBIC)

        # Center crops for img_A_orig
        crops = center_corners_crop(img_A_orig, crop_size=self.crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]

        # Adjust the number of crops
        target_num_crops = 5  # Center crop + 4 corners
        crops += [crops[-1]] * (target_num_crops - len(crops)) if len(crops) < target_num_crops else crops[:target_num_crops]

        img_A = torch.stack(crops, dim=0)  # Shape: [num_crops, 3, crop_size, crop_size]

        # Normalize images
        img_A = self.normalize(img_A)

        # 이미지 B는 img_A와 다르게 처리할 수 있음 (예: 왜곡된 이미지 추가)
        img_B = img_A.clone()  # 예시로 원본 이미지를 복사하여 사용

        mos = self.mos[index]

        # Debugging: Check shapes
        print(f"img_A shape: {img_A.shape}, img_B shape: {img_B.shape}")

        # 배치 차원 추가하여 차원 변환
        batch_size = 1  # In DataLoader, this is dynamically set
        crops = torch.stack(crops, dim=0)  # Shape: [num_crops, 3, crop_size, crop_size]

        # Reshape to maintain batch size and num_crops dimension
        img_A = crops.unsqueeze(0)  # Shape: [1, num_crops, 3, crop_size, crop_size]

        # Since img_B is just a copy of img_A here, apply the same transformation
        img_B = img_A.clone()

        return {
            "img_A_orig": img_A,  # Shape: [1, num_crops, 3, crop_size, crop_size]
            "img_B_orig": img_B,
            "img_A_ds": img_A,
            "img_B_ds": img_B,
            "mos": self.mos[index],
        }

    def __len__(self):
        return len(self.images)
 """

import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from data.dataset_base_iqa import IQADataset
from torch.utils.data import Dataset

# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def center_corners_crop(image: Image, crop_size: int) -> list:
    """
    Center crop the image into four corners and the center.

    Args:
        image (PIL.Image): The input image.
        crop_size (int): The size of the crop.

    Returns:
        list: List of cropped images.
    """
    width, height = image.size
    crops = []

    # Center crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    crops.append(image.crop((left, top, left + crop_size, top + crop_size)))

    # Four corners crop
    crops.append(image.crop((0, 0, crop_size, crop_size)))  # Top-left
    crops.append(image.crop((width - crop_size, 0, width, crop_size)))  # Top-right
    crops.append(image.crop((0, height - crop_size, crop_size, height)))  # Bottom-left
    crops.append(image.crop((width - crop_size, height - crop_size, width, height)))  # Bottom-right

    return crops

class SPAQDataset(IQADataset):
    def __init__(self, root: str, phase: str = "train", split_idx: int = 0, crop_size: int = 224):
        super().__init__(root, phase, split_idx, crop_size)

        # Load scores from CSV
        print("Loading scores from CSV...")
        scores_csv = pd.read_csv(self.root / "Annotations" / "MOS_and_Image_attribute_scores.csv")
        print("Scores CSV head:", scores_csv.head())  # 디버깅 출력

        # Assuming 'Image name' is the correct column name
        self.images = scores_csv["Image name"].values.tolist()
        self.mos = scores_csv["MOS"].values.tolist()

        # Convert image paths to full paths
        self.images = np.array([self.root / "TestImage" / el for el in self.images])

        print(f"Loaded {len(self.images)} images with corresponding MOS scores.")

        if os.path.exists(self.root / "Annotations" / "MOS and Image attribute scores.csv"):
            print("CSV file found.")
        else:
            print("CSV file not found.")

    def __getitem__(self, index: int) -> dict:
        try:
            img_A_orig = Image.open(self.images[index]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {self.images[index]}: {e}")
            return {}

        img_A_orig = img_A_orig.resize((self.crop_size, self.crop_size), Image.BICUBIC)

        # Center crops for img_A_orig
        crops = center_corners_crop(img_A_orig, crop_size=self.crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]

        # Adjust the number of crops to ensure we have 5 crops
        target_num_crops = 5  # Center crop + 4 corners
        if len(crops) < target_num_crops:
            crops += [crops[-1]] * (target_num_crops - len(crops))
        else:
            crops = crops[:target_num_crops]

        img_A = torch.stack(crops, dim=0)  # Shape: [num_crops, 3, crop_size, crop_size]

        # Normalize images
        img_A = self.normalize(img_A)

        # Create img_B as a copy of img_A for demonstration (you can modify this as needed)
        img_B = img_A.clone()

        mos = self.mos[index]

        # Debugging: Check shapes
        print(f"img_A shape: {img_A.shape}, img_B shape: {img_B.shape}")

        return {
            "img_A_orig": img_A,  # Shape: [num_crops, 3, crop_size, crop_size]
            "img_B_orig": img_B,
            "img_A_ds": img_A,
            "img_B_ds": img_B,
            "mos": torch.tensor(mos, dtype=torch.float),
        }

    def __len__(self):
        return len(self.images)
