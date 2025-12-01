from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.utils import safe_normalize


class Centering(Enum):
    none = 0
    mean = 1
    standard = 2


def center_embeddings(
    embeddings: torch.Tensor,
    centering_type: Centering,
) -> torch.Tensor:
    if centering_type == Centering.none:
        return embeddings
    elif centering_type == Centering.mean:
        return embeddings - embeddings.mean(0, keepdim=True)
    elif centering_type == Centering.standard:
        return (embeddings - embeddings.mean(0, keepdim=True)) / embeddings.std(
            0, keepdim=True
        )


class DistanceFunction(Enum):
    cosine = 0
    rbf = 1
    spearman = 2


def compute_similarity(
    embeddings: torch.Tensor,
    distance_type: DistanceFunction,
    temperature=None,
    gamma=None,
) -> torch.Tensor:
    if distance_type == DistanceFunction.cosine:
        assert temperature is not None
        return (embeddings @ embeddings.T) / temperature
    elif distance_type == DistanceFunction.rbf:
        assert gamma is not None
        distances = torch.cdist(embeddings, embeddings, p=2)
        return torch.exp(-gamma * distances.pow(2))
    elif distance_type == DistanceFunction.spearman:
        assert temperature is not None
        ranks = embeddings.argsort(dim=-1).argsort(dim=-1).float()
        ranks = F.normalize(ranks, p=2, dim=-1)
        return (ranks @ ranks.T) / temperature


def structure_reg(
    original_embeddings: torch.Tensor,
    aligned_embeddings: torch.Tensor,
    levels: int = 3,
    temperature: float = 0.1,
    gamma: float = 1.0,
    margin: float = 0.0,
    eps: float = 1e-12,
    weighting: str = "inverse",
    distance_type: DistanceFunction = DistanceFunction.cosine,
    centering_type: Centering = Centering.mean,
    center_first: bool = False,
):
    """
    Compute structure regularizer loss between original and aligned embeddings.
    Preserves relationships at multiple scales.
    """
    with torch.amp.autocast("cuda", enabled=False):
        # normalize embeddings for numerical stability
        if center_first:
            original_embeddings = center_embeddings(
                embeddings=original_embeddings,
                centering_type=centering_type,
            )
            aligned_embeddings = center_embeddings(
                embeddings=aligned_embeddings,
                centering_type=centering_type,
            )
        original_norm = safe_normalize(original_embeddings, p=2, dim=-1)
        aligned_norm = safe_normalize(aligned_embeddings, p=2, dim=-1)
        if not center_first:
            original_norm = center_embeddings(
                embeddings=original_norm,
                centering_type=centering_type,
            )
            aligned_norm = center_embeddings(
                embeddings=aligned_norm,
                centering_type=centering_type,
            )
        # compute similarity matrices with temperature scaling
        original_sim = compute_similarity(
            embeddings=original_norm,
            distance_type=distance_type,
            temperature=temperature,
            gamma=gamma,
        )
        aligned_sim = compute_similarity(
            embeddings=aligned_norm,
            distance_type=distance_type,
            temperature=temperature,
            gamma=gamma,
        )
        # apply robust softmax with stable log-sum-exp trick
        total_loss = 0
        for level in range(1, levels + 1):
            original_structure = torch.matrix_power(
                F.softmax(original_sim, dim=-1),
                level,
            )
            aligned_structure = torch.matrix_power(
                F.softmax(aligned_sim, dim=-1),
                level,
            )
            # use Jensen-Shannon divergence instead of KL (more stable)
            m = 0.5 * (original_structure + aligned_structure)
            js_div = 0.5 * (
                F.kl_div(
                    (aligned_structure + eps).log(), m + eps, reduction="batchmean"
                )
                + F.kl_div(
                    (original_structure + eps).log(), m + eps, reduction="batchmean"
                )
            )
            # add a margin in order to keep focusing on big improvements rather than small
            js_div = F.relu(js_div - margin)
            # apply scaling to keep loss magnitudes reasonable
            if weighting == "none":
                scaled_loss = js_div
            elif weighting == "inverse":
                # lower weight for higher levels
                scaled_loss = js_div * (1.0 / level)
            else:
                raise ValueError(f"Unknown weighting: {weighting}")
            total_loss += scaled_loss
    return total_loss / levels


class CLIPLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        normalize_latents: bool = False,
        warmup_steps: int = 500,
        structure_lambda: float = 10,
        structure_levels: int = 1,
        structure_weighting: str = "none",
        structure_margin: float = 0.0,
        structure_centering: str = "mean",
        structure_distance: str = "cosine",
        structure_centering_first: bool = False,
        structure_use_only_unimodal: bool = False,
    ):
        super().__init__()
        self.train_step = 0
        self.warmup_steps = warmup_steps

        self.temperature = temperature
        self.normalize_latents = normalize_latents

        self.structure_lambda_base = structure_lambda
        self.structure_levels = structure_levels
        self.structure_weighting = structure_weighting
        self.structure_margin = structure_margin
        self.structure_distance = DistanceFunction[structure_distance]
        self.structure_centering = Centering[structure_centering]
        self.structure_centering_first = structure_centering_first
        self.structure_use_only_unimodal = structure_use_only_unimodal

        # for schedulers (i.e. will be updated)
        self.structure_lambda = structure_lambda

    def name(self):
        name = "CLIPLoss"
        name += f"(temp={self.temperature}"
        name += f", norm={self.normalize_latents}"
        if self.structure_lambda > 0:
            name += f", structure_distance={self.structure_distance.name}"
            name += f", structure_centering={self.structure_centering.name}"
            name += f", structure_centering_first={self.structure_centering_first}"
            name += f", structure_lambda={self.structure_lambda_base}"
            name += f", structure_levels={self.structure_levels}"
            name += f", warmup_steps={self.warmup_steps}"
        name += ")"
        return name

    def step(self):
        self.structure_lambda = self.structure_lambda_base * min(
            1.0, self.train_step / self.warmup_steps
        )
        self.train_step += 1

    def forward(
        self,
        image_embeddings_aligned: torch.Tensor,
        text_embeddings_aligned: torch.Tensor,
        image_embeddings_original: torch.Tensor,
        text_embeddings_original: torch.Tensor,
        add_image_features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        add_text_features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if self.normalize_latents:
            image_embeddings_aligned = safe_normalize(
                image_embeddings_aligned, p=2, dim=1
            )
            text_embeddings_aligned = safe_normalize(
                text_embeddings_aligned, p=2, dim=1
            )

        # Standard CLIP contrastive loss
        logits = image_embeddings_aligned @ text_embeddings_aligned.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        clip_loss = (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        ) / 2

        loss_dict = {"clip_loss": clip_loss.item()}
        total_loss = clip_loss
        if self.structure_lambda > 0:
            if add_image_features is not None:
                add_image_features_original, add_image_features_aligned = (
                    add_image_features
                )
                if self.normalize_latents:
                    add_image_features_aligned = safe_normalize(
                        add_image_features_aligned, p=2, dim=1
                    )
                if self.structure_use_only_unimodal:
                    image_embeddings_original = add_image_features_original
                    image_embeddings_aligned = add_image_features_aligned
                else:
                    image_embeddings_original = torch.concat(
                        [image_embeddings_original, add_image_features_original], dim=0
                    )
                    image_embeddings_aligned = torch.concat(
                        [image_embeddings_aligned, add_image_features_aligned], dim=0
                    )
            img_structure_loss = structure_reg(
                image_embeddings_original,
                image_embeddings_aligned,
                levels=self.structure_levels,
                temperature=self.temperature,
                weighting=self.structure_weighting,
                margin=self.structure_margin,
                distance_type=self.structure_distance,
                centering_type=self.structure_centering,
                center_first=self.structure_centering_first,
            )

            if add_text_features is not None:
                add_text_features_original, add_text_features_aligned = (
                    add_text_features
                )
                if self.normalize_latents:
                    add_text_features_aligned = safe_normalize(
                        add_text_features_aligned, p=2, dim=1
                    )

                if self.structure_use_only_unimodal:
                    text_embeddings_original = add_text_features_original
                    text_embeddings_aligned = add_text_features_aligned
                else:
                    text_embeddings_original = torch.concat(
                        [text_embeddings_original, add_text_features_original], dim=0
                    )
                    text_embeddings_aligned = torch.concat(
                        [text_embeddings_aligned, add_text_features_aligned], dim=0
                    )
            txt_structure_loss = structure_reg(
                text_embeddings_original,
                text_embeddings_aligned,
                levels=self.structure_levels,
                temperature=self.temperature,
                weighting=self.structure_weighting,
                margin=self.structure_margin,
                distance_type=self.structure_distance,
                centering_type=self.structure_centering,
                center_first=self.structure_centering_first,
            )
            structure_loss = (img_structure_loss + txt_structure_loss) / 2
            loss_dict["structure_loss_wo_lambda"] = structure_loss.item()
            loss_dict["structure_loss"] = self.structure_lambda * structure_loss.item()
            total_loss += self.structure_lambda * structure_loss

        loss_dict["overall_loss"] = total_loss
        return loss_dict
