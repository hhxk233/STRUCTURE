import json
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class LoadingType(str, Enum):
    BOTH = "both"
    TXT_ONLY = "txt_only"
    IMG_ONLY = "img_only"


class CocoCaptionDataset(Dataset):
    """
    Minimal COCO captions dataset loader with optional text-only mode.
    """

    def __init__(
        self,
        annotation_file: Path,
        image_dir: Path,
        transform=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.loading_type = LoadingType.BOTH
        self.tokenizer = None
        self._tokenized = None

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"COCO image dir not found: {image_dir}")

        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
        rows = []
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            file_name = id_to_file[image_id]
            image_path = str(self.image_dir / file_name)
            rows.append(
                {
                    "image_id": image_id,
                    "image_name": file_name,
                    "image_path": image_path,
                    "caption": caption,
                }
            )
        self.df = pd.DataFrame(rows)

    def apply_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before calling apply_tokenizer().")
        texts = self.df["caption"].tolist()
        tokens = self.tokenizer(texts, padding="longest", return_tensors="pt")
        self._tokenized = tokens

    def __len__(self) -> int:
        return len(self.df)

    def _get_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self._get_image(row["image_path"])
        if self.loading_type == LoadingType.TXT_ONLY:
            if self._tokenized is None:
                raise ValueError("Tokenizer was not applied for TXT_ONLY mode.")
            token_inputs = {k: v[idx] for k, v in self._tokenized.items()}
            return image, token_inputs
        if self.loading_type == LoadingType.IMG_ONLY:
            return image
        return image, row["caption"]
