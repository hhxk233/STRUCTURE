from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.core.src.datasets.downstream_tasks.coco_dataset import LoadingType


class Flickr30kDataset(Dataset):
    """
    Minimal Flickr30k loader based on a metadata CSV with split, filename, caption.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        meta_path: Union[str, Path],
        split: Union[str, Sequence[str]] = "test",
        transform=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.meta_path = Path(meta_path)
        self.transform = transform
        self.loading_type = LoadingType.BOTH
        self.tokenizer = None
        self._tokenized = None

        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"Flickr30k metadata file not found: {self.meta_path}"
            )
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Flickr30k image directory not found: {self.root_dir}"
            )

        df = pd.read_csv(self.meta_path)
        split_values = [split] if isinstance(split, str) else list(split)
        if "split" in df.columns:
            df = df[df["split"].isin(split_values)]
        if "caption" not in df.columns:
            raise ValueError("Flickr30k CSV must contain a 'caption' column.")
        if "image_path" not in df.columns:
            if "image_name" in df.columns:
                df["image_path"] = df["image_name"].apply(
                    lambda x: str(self.root_dir / x)
                )
            elif "filename" in df.columns:
                df["image_path"] = df["filename"].apply(
                    lambda x: str(self.root_dir / x)
                )
            else:
                raise ValueError(
                    "Flickr30k CSV must include image_path, image_name, or filename."
                )
        if "image_name" not in df.columns:
            df["image_name"] = df["image_path"].apply(lambda x: Path(x).name)
        self.df = df.reset_index(drop=True)

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
