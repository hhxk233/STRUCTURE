from typing import Callable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class ImageTextDataset(Dataset):
    """
    Wrap a vision dataset to generate text prompts from class labels.
    """

    def __init__(
        self,
        dataset: Dataset,
        label_templates: Sequence[str],
        template_key: str = "label",
        precompute_captions: bool = True,
    ) -> None:
        self.dataset = dataset
        self.label_templates = list(label_templates)
        self.template_key = template_key
        self.precompute_captions = precompute_captions
        self._captions = None
        if self.precompute_captions:
            self._captions = [self._build_caption(i) for i in range(len(self.dataset))]

    def _build_caption(self, idx: int) -> str:
        _, target = self.dataset[idx]
        label = (
            self.dataset.classes[target]
            if hasattr(self.dataset, "classes")
            else str(target)
        )
        template = self.label_templates[0]
        return template.format(**{self.template_key: label})

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        if self._captions is None:
            caption = self._build_caption(idx)
        else:
            caption = self._captions[idx]
        return image, caption, target

    @staticmethod
    def collate_fn(batch: List[Tuple]):
        images = []
        texts = []
        targets = []
        has_targets = False
        for item in batch:
            if len(item) == 2:
                image, text = item
                target = None
            elif len(item) == 3:
                image, text, target = item
                has_targets = True
            else:
                raise ValueError(f"Unexpected batch item size: {len(item)}")

            images.append(image)
            texts.append(text)
            if target is not None:
                targets.append(target)

        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0)

        if isinstance(texts[0], dict):
            token_inputs = {k: torch.stack([t[k] for t in texts], dim=0) for k in texts[0]}
            texts = token_inputs

        if has_targets:
            targets = torch.tensor(targets)
            return images, texts, targets
        return images, texts
