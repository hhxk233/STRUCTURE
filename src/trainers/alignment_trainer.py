import ast
import copy
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torchmetrics
import wandb
from deepspeed.runtime.lr_schedules import WarmupDecayLR, WarmupLR
from loguru import logger
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm, trange
from transformers import AutoImageProcessor, AutoModel

from src.alignment.alignment_factory import AlignmentFactory
from src.core.src.datasets.downstream_tasks.coco_dataset import LoadingType
from src.core.src.optimizers.utils import get_optimizer_type
from src.core.src.utils.plotting import embedding_plot, embedding_plot_w_markers
from src.core.src.utils.utils import EarlyStopping, clip_gradients, save_checkpoint
from src.dataset_preparation.data_utils import (
    FeatureDataset,
    _ensure_rgb_image,
    get_meta_dict,
)
from src.evaluation.consts import (
    DATASETS_TO_CLASSES,
    DATASETS_TO_TEMPLATES,
    SIMPLE_PROMPT_TEMPLATE,
)
from src.evaluation.alignment_metrics import (
    compute_cross_modal_purity_score,
    compute_purity_score,
    compute_silhouette_score,
)
from src.evaluation.retrieval import retrieval_metrics_df
from src.evaluation.zero_shot_classifier import (
    build_zero_shot_classifier,
    chunked_logits,
)
from src.loss.clip_loss import CLIPLoss
from src.measure_alignment import compute_score
from src.models.text.models import load_llm, load_tokenizer
from src.trainers.base_trainer import Trainer
from src.utils.utils import (
    continuity,
    safe_normalize,
    set_transform_dataset,
    trustworthiness,
)


class AlignmentTrainer(Trainer):
    def __init__(
        self,
        config: dict,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        llm_model_name: str,
        lvm_model_name: str,
        eval_zero_shot_datasets: Optional[List[DataLoader]] = None,
        eval_retrieval_datasets: Optional[List[DataLoader]] = None,
        cache_features: bool = False,
        print_model_summary: bool = True,
        wandb_logging: bool = True,
        wandb_project_name: str = "representation-alignment",
        wandb_notes: Optional[str] = None,
    ):
        self.exp_name = f"alignment-{AlignmentTrainer.get_model_name(llm_model_name)}-{AlignmentTrainer.get_model_name(lvm_model_name)}"
        config["llm_model_name"] = llm_model_name
        config["lvm_model_name"] = lvm_model_name
        config["experiment_name"] = self.exp_name
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            experiment_name=self.exp_name,
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
            wandb_notes=wandb_notes,
        )
        save_cfg = self.config["evaluation"].get("save_embeddings", {})
        if isinstance(save_cfg, bool):
            self.save_embeddings_enabled = save_cfg
            self.save_embeddings_top_k = 0
            self.save_embeddings_datasets = None
        else:
            self.save_embeddings_enabled = save_cfg.get("enabled", False)
            self.save_embeddings_top_k = save_cfg.get("top_k", 0)
            self.save_embeddings_datasets = save_cfg.get("datasets")
        self.save_embedding_layer_combinations = None
        self.cache_features = cache_features
        self.print_model_summary = print_model_summary
        self.save_path = Path(config["paths"]["save_path"])
        self.llm_model_name = llm_model_name
        self.lvm_model_name = lvm_model_name
        self.eval_zero_shot_datasets = eval_zero_shot_datasets
        self.eval_retrieval_datasets = eval_retrieval_datasets

        # cache dummies
        self.image_features_val = None
        self.image_features_train = None
        self.text_features_val = None
        self.text_features_train = None

        # the dataframe we use to store the scores
        self.df_scores_zero_shot = None
        self.df_scores_retrieval = None

        # make sure that our experiment folder is there
        (self.save_path / self.exp_name).mkdir(parents=True, exist_ok=True)
        run_name = wandb.run.name if wandb.run is not None else "offline"
        (self.save_path / run_name).mkdir(parents=True, exist_ok=True)

    def __del__(self):
        # do garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def get_model_name(m_name: str):
        return m_name.replace("/", "_").replace("-", "_")

    @staticmethod
    def get_feature_save_path(
        m_name: str, d_name: str, save_path: Path, suffix: str = ""
    ):
        m_name = AlignmentTrainer.get_model_name(m_name=m_name)
        return save_path / "features" / f"{m_name}-{d_name}-{suffix}.npy"

    @staticmethod
    def build_dataset_tag(dataset, dataset_name: str, dataset_size: Optional[int] = None):
        parts = [dataset_name]
        if hasattr(dataset, "annotation_file"):
            parts.append(Path(dataset.annotation_file).stem)
        if hasattr(dataset, "meta_path"):
            parts.append(Path(dataset.meta_path).stem)
        if hasattr(dataset, "root_dir"):
            parts.append(Path(dataset.root_dir).name)
        if hasattr(dataset, "image_dir"):
            parts.append(Path(dataset.image_dir).name)
        if dataset_size is None:
            try:
                dataset_size = len(dataset)
            except Exception:
                dataset_size = None
        if dataset_size is not None:
            parts.append(f"n{dataset_size}")
        return "-".join(parts)

    def add_exp_suffix_to_name(self, base_name: str):
        save_name = f"{base_name}"
        save_name += f"_{self.config['layer_selection']['n_samples']}"
        save_name += f"_{self.config['layer_selection']['metric']}"
        if self.config["layer_selection"].get("metric_kwargs", None) is not None:
            save_name += "_".join(
                map(str, self.config["layer_selection"]["metric_kwargs"].values())
            )
        return save_name

    def get_llm(self, llm_model_name: str):
        language_model = load_llm(llm_model_name)
        # since we're using huggingface's automapping
        # we don't need to move it to the device
        language_model = language_model.eval()
        tokenizer = load_tokenizer(llm_model_name)
        return language_model, tokenizer

    def get_lvm(self, lvm_model_name: str):
        if "/" in lvm_model_name or "dinov3" in lvm_model_name.lower():
            image_processor = AutoImageProcessor.from_pretrained(lvm_model_name)
            vision_model = AutoModel.from_pretrained(lvm_model_name)

            size = image_processor.size
            if isinstance(size, dict):
                height = size.get("height") or size.get("shortest_edge") or size.get(
                    "width"
                )
                width = size.get("width") or size.get("shortest_edge") or height
            else:
                height = width = size
            crop_size = getattr(image_processor, "crop_size", None)
            if isinstance(crop_size, dict):
                crop_h = crop_size.get("height") or crop_size.get("shortest_edge")
                crop_w = crop_size.get("width") or crop_h
            else:
                crop_h = crop_w = crop_size

            transform_steps = [_ensure_rgb_image]
            if getattr(image_processor, "do_resize", True):
                transform_steps.append(transforms.Resize((height, width)))
            if getattr(image_processor, "do_center_crop", True) and crop_h and crop_w:
                transform_steps.append(transforms.CenterCrop((crop_h, crop_w)))
            transform_steps.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=image_processor.image_mean,
                        std=image_processor.image_std,
                    ),
                ]
            )
            transform = transforms.Compose(transform_steps)

            class HFVisionWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    outputs = self.model(
                        pixel_values=x, output_hidden_states=True, return_dict=True
                    )
                    # drop the embedding output so indices match transformer blocks
                    hidden_states = outputs.hidden_states[1:]
                    return {
                        f"blocks.{i}.out": h for i, h in enumerate(hidden_states)
                    }

            vision_model = HFVisionWrapper(vision_model)
            vision_model = vision_model.to(self.device)
            vision_model = vision_model.eval()
            return vision_model, transform

        vision_model = timm.create_model(lvm_model_name, pretrained=True)
        transform = create_transform(
            **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
        )
        transform.transforms = [_ensure_rgb_image] + transform.transforms

        if "vit" in lvm_model_name:
            return_nodes = [
                f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))
            ]
        else:
            raise NotImplementedError(f"unknown model {lvm_model_name}")
        vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
        vision_model = vision_model.to(self.device)
        vision_model = vision_model.eval()
        return vision_model, transform

    def get_text_features(
        self,
        loader,
        llm_model_name: str,
        suffix: str = "",
        dataset_name: Optional[str] = None,
    ):
        if hasattr(loader.dataset, "precomputed_text_features"):
            return loader.dataset.precomputed_text_features
        dataset_ref = loader.dataset
        base_dataset = dataset_ref.dataset if isinstance(dataset_ref, Subset) else dataset_ref
        if hasattr(base_dataset, "name"):
            dataset_name = base_dataset.name
        elif dataset_name is None:
            dataset_name = type(base_dataset).__name__
        dataset_name = AlignmentTrainer.build_dataset_tag(
            base_dataset, dataset_name, dataset_size=len(loader.dataset)
        )
        save_path = AlignmentTrainer.get_feature_save_path(
            m_name=llm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=suffix,
        )

        if save_path.exists():
            llm_feats = torch.load(save_path, weights_only=False)["features"]
            logger.debug(f"Loaded features from: {save_path}")
            return llm_feats

        language_model, tokenizer = self.get_llm(llm_model_name=llm_model_name)
        dataset_ref.tokenizer = tokenizer
        if hasattr(dataset_ref, "loading_type"):
            # for optimizing the loading and looping
            dataset_ref.loading_type = LoadingType.TXT_ONLY
        _df = dataset_ref.df.copy() if hasattr(dataset_ref, "df") else None
        if hasattr(dataset_ref, "apply_tokenizer"):
            dataset_ref.apply_tokenizer()
        # ensure this is still the same ordering
        if _df is not None and hasattr(dataset_ref, "df"):
            if not dataset_ref.df.equals(_df):
                logger.warning("Dataset dataframe changed during tokenization.")
        del _df

        llm_feats = []
        for batch in tqdm(loader, total=len(loader), file=sys.stdout):
            if len(batch) == 2:
                _, token_inputs = batch
            elif len(batch) == 3:
                _, token_inputs, _ = batch
            else:
                raise ValueError(f"Unexpected batch item size: {len(batch)}")

            if not isinstance(token_inputs, dict):
                token_inputs = tokenizer(
                    list(token_inputs), padding="longest", return_tensors="pt"
                )
            token_inputs = {
                k: v.to(self.device).long() for (k, v) in token_inputs.items()
            }
            with torch.no_grad():
                if "olmo" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                else:
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                    )
                if self.config["features"]["pool_txt"] == "avg":
                    # swap the backsize to the first dimension
                    # (BS, Layers, Tokens, Dim)
                    feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                    # make the mask compatible with the dimension
                    mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                    # average along the token dimension
                    feats = (feats * mask).sum(2) / mask.sum(2)
                elif self.config["features"]["pool_txt"] == "last":
                    feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                    feats = torch.stack(feats).permute(1, 0, 2)
                elif self.config["features"]["pool_txt"] == "none":
                    assert self.config["features"].get("layer_txt") is not None
                    feats = torch.stack(list(llm_output["hidden_states"]))
                    # permute to dim: (bs, layers, tokens, dim)
                    feats = feats.permute(1, 0, 2, 3)
                    # select only the layer we care about, otherwise we don't have enough memory
                    feats = feats[:, self.config["features"]["layer_txt"], :, :]
                else:
                    raise NotImplementedError(
                        f"unknown pooling {self.config['features']['pool_txt']}"
                    )
                llm_feats.append(feats.cpu())
        llm_feats = torch.cat(llm_feats).cpu()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"features": llm_feats}
        if hasattr(loader.dataset, "df"):
            save_dict["dataframe"] = loader.dataset.df
        torch.save(save_dict, save_path)
        logger.debug(f"Saved features to: {save_path}")
        del language_model
        return llm_feats

    def get_image_features(
        self,
        loader,
        lvm_model_name: str,
        suffix: str = "",
        dataset_name: Optional[str] = None,
    ):
        if hasattr(loader.dataset, "precomputed_image_features"):
            return loader.dataset.precomputed_image_features
        dataset_ref = loader.dataset
        base_dataset = dataset_ref.dataset if isinstance(dataset_ref, Subset) else dataset_ref
        if hasattr(base_dataset, "name"):
            dataset_name = base_dataset.name
        elif dataset_name is None:
            dataset_name = type(base_dataset).__name__
        dataset_name = AlignmentTrainer.build_dataset_tag(
            base_dataset, dataset_name, dataset_size=len(loader.dataset)
        )
        save_path = AlignmentTrainer.get_feature_save_path(
            m_name=lvm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=suffix,
        )

        if save_path.exists():
            lvm_feats = torch.load(save_path, weights_only=False)["features"]
            logger.debug(f"Loaded features from: {save_path}")
            return lvm_feats

        vision_model, image_transform = self.get_lvm(lvm_model_name=lvm_model_name)
        set_transform_dataset(
            dataset=loader.dataset,
            image_transform=image_transform,
        )

        lvm_feats = []
        for batch in tqdm(loader, total=len(loader), file=sys.stdout):
            images, _ = batch
            with torch.no_grad():
                images = images.to(self.device, non_blocking=True)
                lvm_output = vision_model(images)
                if self.config["features"]["pool_img"] == "cls":
                    # extract the class token for all layers
                    feats = [v[:, 0, :] for v in lvm_output.values()]
                    feats = torch.stack(feats).permute(1, 0, 2)
                elif self.config["features"]["pool_img"] == "none":
                    assert self.config["features"].get("layer_img") is not None
                    feats = torch.stack(list(lvm_output.values()))
                    # permute to dim: (bs, layers, tokens, dim)
                    feats = feats.permute(1, 0, 2, 3)
                    # select only the layer we care about, otherwise we don't have enough memory
                    feats = feats[:, self.config["features"]["layer_img"], :, :]
                else:
                    raise NotImplementedError(
                        f"unknown pooling {self.config['features']['pool_img']}"
                    )
                lvm_feats.append(feats.cpu())
        lvm_feats = torch.cat(lvm_feats).cpu()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"features": lvm_feats}
        if hasattr(loader.dataset, "df"):
            save_dict["dataframe"] = loader.dataset.df
        torch.save(save_dict, save_path)
        logger.debug(f"Saved features to: {save_path}")
        del vision_model
        return lvm_feats

    def compute_layer_alignment(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        # compute similarity between the models on training set
        logger.debug("Computing alignment between modalities")
        alignment_csv = (
            self.save_path
            / self.exp_name
            / (self.add_exp_suffix_to_name("df_alignment") + ".csv")
        )
        # only read from memory when we're not subsampling!
        if (
            alignment_csv.exists()
            and self.n_random_subsample_train is None
            and self.n_random_subsample_val is None
        ):
            df_alignment = pd.read_csv(alignment_csv)
            df_alignment["indices"] = df_alignment["indices"].apply(ast.literal_eval)
            logger.debug('Loaded "df_alignment" from disk.')
        else:
            if self.config["layer_selection"]["type"] == "random":
                sel_samples = np.random.choice(
                    image_features.shape[0],
                    min(
                        self.config["layer_selection"]["n_samples"],
                        image_features.shape[0],
                    ),
                    replace=False,
                )
            else:
                raise ValueError(
                    f"Unknown layer selection type: {self.config['layer_selection']['type']}"
                )
            _, _, alignment_list = compute_score(
                x_feats=image_features[sel_samples].float().to(self.device),
                y_feats=text_features[sel_samples].float().to(self.device),
                metric=self.config["layer_selection"]["metric"],
                **self.config["layer_selection"].get("metric_kwargs", {}),
            )
            df_alignment = pd.DataFrame(alignment_list)
            df_alignment["indices_x"] = df_alignment["indices"].apply(lambda x: x[0])
            df_alignment["indices_y"] = df_alignment["indices"].apply(lambda x: x[1])
            # remove all scores from the concatenated layers, since we're not interested in them!
            df_alignment = df_alignment[
                (df_alignment["indices_x"] != -1) & (df_alignment["indices_y"] != -1)
            ]
            df_alignment.to_csv(alignment_csv, index=False)

        n_score_bins = self.config["layer_selection"]["n_score_bins"]
        sampled_csv = (
            self.save_path
            / self.exp_name
            / (
                self.add_exp_suffix_to_name("sampled_df_alignment")
                + f"_bins{n_score_bins}"
                + ".csv"
            )
        )
        if (
            sampled_csv.exists()
            and self.n_random_subsample_train is None
            and self.n_random_subsample_val is None
        ):
            sampled_df_alignment = pd.read_csv(sampled_csv)
            sampled_df_alignment["indices"] = sampled_df_alignment["indices"].apply(
                ast.literal_eval
            )
            logger.debug('Loaded "sampled_df_alignment" from disk.')
        else:
            df_alignment["quantile_bin"] = pd.qcut(
                df_alignment["alignment_score"],
                q=self.config["layer_selection"]["n_score_bins"],
                labels=False,
                duplicates="drop",
            )
            sampled_df_alignment = (
                df_alignment.groupby("quantile_bin", group_keys=False)
                .apply(
                    lambda x: x.sample(n=1, random_state=self.config["random_state"])
                )
                .reset_index(drop=True)
            )
            sampled_df_alignment = sampled_df_alignment.sort_values(
                by="alignment_score",
                ascending=False,
            )
            # make sure that the min and max is sampled
            df_alignment = df_alignment.sort_values(
                by="alignment_score",
                ascending=False,
            )
            series_highest = df_alignment.iloc[0]
            series_lowest = df_alignment.iloc[-1]
            sampled_df_alignment.iloc[0] = series_highest
            sampled_df_alignment.iloc[-1] = series_lowest
            # make sure that we are always as well including the last layers as well
            df_alignment = df_alignment.sort_values(
                by=["indices_x", "indices_y"],
                ascending=False,
            )
            last_layer = df_alignment.iloc[0].copy()
            last_layer["indices"] = (-1, -1)
            last_layer["indices_x"] = -1
            last_layer["indices_y"] = -1
            sampled_df_alignment.loc[len(sampled_df_alignment)] = last_layer
            # make sure we drop the duplicates, if any
            sampled_df_alignment.drop_duplicates(subset="indices", inplace=True)
            sampled_df_alignment.to_csv(sampled_csv, index=False)

        if self.config["layer_selection"]["best_only"]:
            logger.debug("Selecting only best layer to align")
            sampled_df_alignment = sampled_df_alignment.sort_values(
                by="alignment_score",
                ascending=False,
            )
            sampled_df_alignment = sampled_df_alignment.iloc[:1]
        elif self.config["layer_selection"]["last_only"]:
            logger.debug("Selecting only last layer to align")
            sampled_df_alignment = sampled_df_alignment[
                sampled_df_alignment["indices"] == (-1, -1)
            ]
        elif self.config["layer_selection"]["n_last_layers"] is not None:
            n_last_layers = self.config["layer_selection"]["n_last_layers"]
            df_alignment = df_alignment.sort_values(
                by=["indices_x", "indices_y"],
                ascending=False,
            )
            last_layer = df_alignment.iloc[0].copy()
            last_layer_index_x = last_layer["indices_x"]
            last_layer_index_y = last_layer["indices_y"]
            sel_layers_x = list(
                range(last_layer_index_x - n_last_layers + 1, last_layer_index_x + 1)
            )
            sel_layers_y = list(
                range(last_layer_index_y - n_last_layers + 1, last_layer_index_y + 1)
            )
            sampled_df_alignment = df_alignment[
                (df_alignment["indices_x"].isin(sel_layers_x))
                & (df_alignment["indices_y"].isin(sel_layers_y))
            ].copy()

        return sampled_df_alignment

    def _should_save_embeddings(
        self,
        alignment_layer_combination: Tuple[int, int],
        dataset_name: str,
    ) -> bool:
        if not self.save_embeddings_enabled:
            return False
        if self.save_embeddings_datasets is not None:
            if dataset_name not in self.save_embeddings_datasets:
                return False
        if self.save_embedding_layer_combinations is not None:
            return alignment_layer_combination in self.save_embedding_layer_combinations
        return True

    def _save_embeddings(
        self,
        dataset_name: str,
        alignment_layer_combination: Tuple[int, int],
        suffix: str,
        payload: dict,
    ) -> None:
        run_name = wandb.run.name if wandb.run is not None else "offline"
        save_dir = self.save_path / run_name / "saved_embeddings"
        save_dir.mkdir(parents=True, exist_ok=True)
        image_layer_idx, text_layer_idx = alignment_layer_combination
        save_path = (
            save_dir
            / f"{dataset_name}_img{image_layer_idx}_txt{text_layer_idx}_{suffix}.pt"
        )
        torch.save(payload, save_path)

    def fit(
        self,
        n_random_subsample_train: Optional[int] = None,
        n_random_subsample_val: Optional[int] = None,
        additional_unimodal_data: Optional[Dict[str, list]] = None,
        n_random_additional_feats: Optional[int] = None,
    ):
        def _subsample_loader(loader: DataLoader, n_samples: Optional[int]):
            if n_samples is None:
                return loader
            dataset = loader.dataset
            if n_samples >= len(dataset):
                return loader
            generator = torch.Generator().manual_seed(self.config["random_state"])
            indices = torch.randperm(len(dataset), generator=generator)[:n_samples]
            subset = Subset(dataset, indices.tolist())
            return DataLoader(
                subset,
                batch_size=loader.batch_size,
                drop_last=False,
                shuffle=False,
                pin_memory=loader.pin_memory,
                num_workers=loader.num_workers,
                collate_fn=loader.collate_fn,
            )

        train_loader = _subsample_loader(self.train_dataset, n_random_subsample_train)
        val_loader = _subsample_loader(self.val_dataset, n_random_subsample_val)

        # pre-compute the embeddings from both modalities
        # first embed the validation set since we're returning
        # the models for the training set
        image_val_suffix = f"val-{self.config['features']['pool_img']}"
        if self.image_features_val is None:
            if self.config["features"].get("layer_img") is not None:
                image_val_suffix += f"_layer-{self.config['features']['layer_img']}"
            image_features_val = self.get_image_features(
                loader=val_loader,
                lvm_model_name=self.lvm_model_name,
                suffix=image_val_suffix,
            )
        else:
            image_features_val = self.image_features_val

        text_val_suffix = f"val-{self.config['features']['pool_txt']}"
        if self.text_features_val is None:
            if self.config["features"].get("layer_img") is not None:
                text_val_suffix += f"_layer-{self.config['features']['layer_txt']}"
            text_features_val = self.get_text_features(
                loader=val_loader,
                llm_model_name=self.llm_model_name,
                suffix=text_val_suffix,
            )
        else:
            text_features_val = self.text_features_val

        if self.image_features_train is None:
            image_features_train = self.get_image_features(
                loader=train_loader,
                lvm_model_name=self.lvm_model_name,
                suffix=image_val_suffix.replace("val-", "train-"),
            )
        else:
            image_features_train = self.image_features_train

        if self.text_features_train is None:
            text_features_train = self.get_text_features(
                loader=train_loader,
                llm_model_name=self.llm_model_name,
                suffix=text_val_suffix.replace("val-", "train-"),
            )
        else:
            text_features_train = self.text_features_train
        image_dim = image_features_train.shape[-1]
        text_dim = text_features_train.shape[-1]

        # check that we have the same samples
        assert image_features_train.shape[0] == text_features_train.shape[0]
        assert image_features_val.shape[0] == text_features_val.shape[0]

        # cache features if wanted
        self.image_features_val = image_features_val
        self.image_features_train = image_features_train
        self.text_features_val = text_features_val
        self.text_features_train = text_features_train

        if (
            self.config["training"]["drop_duplicates"]
            and hasattr(self.train_dataset.dataset, "df")
            and self.train_dataset.dataset.df is not None
            and "image_path" in self.train_dataset.dataset.df.columns
        ):
            sel_train_indices = (
                self.train_dataset.dataset.df.groupby("image_path").cumcount()
                < self.config["training"]["n_dup_samples"]
            )
            image_features_train = image_features_train[sel_train_indices]
            text_features_train = text_features_train[sel_train_indices]

            sel_val_indices = (
                self.val_dataset.dataset.df.groupby("image_path").cumcount()
                < self.config["training"]["n_dup_samples"]
            )
            image_features_val = image_features_val[sel_val_indices]
            text_features_val = text_features_val[sel_val_indices]

        if (
            n_random_subsample_train is not None
            and n_random_subsample_train < image_features_train.shape[0]
        ):
            logger.debug(f"Subsampling train set to {n_random_subsample_train}")
            self.n_random_subsample_train = n_random_subsample_train
            wandb.run.tags = wandb.run.tags + (
                f"TRAIN subsample {n_random_subsample_train}",
            )

            random_sequence = torch.randperm(image_features_train.shape[0])[
                :n_random_subsample_train
            ]
            image_features_train = image_features_train[random_sequence]
            text_features_train = text_features_train[random_sequence]
        if (
            n_random_subsample_val is not None
            and n_random_subsample_val < image_features_val.shape[0]
        ):
            logger.debug(f"Subsampling validation set to {n_random_subsample_val}")
            self.n_random_subsample_val = n_random_subsample_val
            wandb.run.tags = wandb.run.tags + (
                f"VAL subsample {n_random_subsample_val}",
            )

            random_sequence = torch.randperm(image_features_val.shape[0])[
                :n_random_subsample_val
            ]
            image_features_val = image_features_val[random_sequence]
            text_features_val = text_features_val[random_sequence]

        logger.debug(
            f"TRAIN - img: {image_features_train.shape}, txt: {text_features_train.shape}"
        )
        logger.debug(
            f"VAL - img: {image_features_val.shape}, txt: {text_features_val.shape}"
        )

        # additional unimodal data
        additional_image_features = None
        additional_text_features = None
        if additional_unimodal_data is not None:
            additional_image_features = []
            additional_text_features = []
            for modality, modality_datasets in additional_unimodal_data.items():
                for m_dataset_name, m_dataset in modality_datasets:
                    if modality == "text":
                        add_text_features = self.get_text_features(
                            loader=m_dataset,
                            llm_model_name=self.llm_model_name,
                            suffix=text_val_suffix.replace("val-", "train-"),
                            dataset_name=m_dataset_name,
                        )
                        additional_text_features.append(add_text_features)
                    else:
                        add_image_features = self.get_image_features(
                            loader=m_dataset,
                            lvm_model_name=self.lvm_model_name,
                            suffix=image_val_suffix.replace("val-", "train-"),
                            dataset_name=m_dataset_name,
                        )
                        additional_image_features.append(add_image_features)
            if len(additional_image_features) > 0:
                additional_image_features = torch.cat(additional_image_features).cpu()
            else:
                additional_image_features = None
            if len(additional_text_features) > 0:
                additional_text_features = torch.cat(additional_text_features).cpu()
            else:
                additional_text_features = None

        # only compute the best alignment if not specified
        if (
            self.config["features"].get("layer_img") is None
            and self.config["features"].get("layer_txt") is None
        ):
            sampled_df_alignment = self.compute_layer_alignment(
                image_features=image_features_train,
                text_features=text_features_train,
            )
        else:
            sampled_df_alignment = pd.DataFrame(columns=["indices", "alignment_score"])
            sampled_df_alignment.loc[len(sampled_df_alignment)] = [
                (
                    self.config["features"]["layer_img"],
                    self.config["features"]["layer_txt"],
                ),
                np.nan,
            ]

        # for each sampled combination
        # train the alignment between the representations
        print(sampled_df_alignment)
        if self.save_embeddings_enabled and self.save_embeddings_top_k:
            df_sorted = sampled_df_alignment.copy()
            if "alignment_score" in df_sorted.columns and df_sorted[
                "alignment_score"
            ].notna().any():
                df_sorted = df_sorted.sort_values(
                    by="alignment_score", ascending=False
                )
            self.save_embedding_layer_combinations = [
                tuple(x) for x in df_sorted["indices"].tolist()[: self.save_embeddings_top_k]
            ]
        comb_iter = sampled_df_alignment.iterrows()
        for i_comb, (_, layer_series) in enumerate(comb_iter):
            layer_comb = layer_series["indices"]
            image_layer_idx, text_layer_idx = layer_comb
            layer_comb_score = layer_series["alignment_score"]
            layer_comb_str = f"img_{image_layer_idx}_txt_{text_layer_idx}"

            layer_image_features_train = image_features_train[:, image_layer_idx, :]
            layer_text_features_train = text_features_train[:, text_layer_idx, :]
            layer_image_features_val = image_features_val[:, image_layer_idx, :]
            layer_text_features_val = text_features_val[:, text_layer_idx, :]

            layer_additional_image_features = None
            if additional_image_features is not None:
                layer_additional_image_features = additional_image_features[
                    :, image_layer_idx, :
                ]

            layer_additional_text_features = None
            if additional_text_features is not None:
                layer_additional_text_features = additional_text_features[
                    :, text_layer_idx, :
                ]

            if (
                len(sampled_df_alignment) == 1
                or i_comb == len(sampled_df_alignment) - 1
            ):
                # clean up the memory if we're only doing one comb or its the last
                del image_features_train, text_features_train
                del image_features_val, text_features_val
                if additional_image_features is not None:
                    del additional_image_features
                if additional_text_features is not None:
                    del additional_text_features

            l_add_img_feats = []
            for add_img_feat_paths in self.config["features"].get(
                "add_img_feat_paths", []
            ):
                if Path(add_img_feat_paths).exists():
                    add_img_feats = torch.load(add_img_feat_paths, weights_only=False)[
                        "image_feats"
                    ]
                    l_add_img_feats.append(add_img_feats)
                    logger.debug(f"Loaded features from: {add_img_feat_paths}")
            if len(l_add_img_feats) > 1:
                l_add_img_feats = torch.cat(l_add_img_feats, dim=0)

            l_add_txt_feats = []
            for add_txt_feat_paths in self.config["features"].get(
                "add_txt_feat_paths", []
            ):
                if Path(add_txt_feat_paths).exists():
                    add_txt_feats = torch.load(add_txt_feat_paths, weights_only=False)[
                        "text_feats"
                    ]
                    l_add_txt_feats.append(add_txt_feats)
                    logger.debug(f"Loaded features from: {add_txt_feat_paths}")
            if len(l_add_txt_feats) > 1:
                l_add_txt_feats = torch.cat(l_add_txt_feats, dim=0)

            if n_random_additional_feats == 0:
                del l_add_img_feats, l_add_txt_feats
            else:
                if (
                    n_random_additional_feats is not None
                    and n_random_additional_feats < l_add_img_feats.shape[0]
                ):
                    logger.debug(f"Subsampling LAION to {n_random_additional_feats}")
                    wandb.run.tags = wandb.run.tags + (
                        f"LAION subsample {n_random_additional_feats}",
                    )
                    random_sequence = torch.randperm(l_add_img_feats.shape[0])[
                        :n_random_additional_feats
                    ]
                    l_add_img_feats = l_add_img_feats[random_sequence]
                    l_add_txt_feats = l_add_txt_feats[random_sequence]
                if len(l_add_img_feats) > 1:
                    layer_image_features_train = torch.cat(
                        (layer_image_features_train, l_add_img_feats), dim=0
                    )
                    logger.debug(
                        f"New train dim image: {layer_image_features_train.shape}"
                    )
                if len(l_add_txt_feats) > 1:
                    layer_text_features_train = torch.cat(
                        (layer_text_features_train, l_add_txt_feats), dim=0
                    )
                    logger.debug(
                        f"New train dim text: {layer_text_features_train.shape}"
                    )

            log_dict = {
                f"{layer_comb_str}/meta/layer_comb": layer_comb,
                f"{layer_comb_str}/meta/layer_comb_score": layer_comb_score,
            }
            if self.n_random_subsample_train is not None:
                log_dict["meta/n_random_subsample_train"] = (
                    self.n_random_subsample_train
                )
            if self.n_random_subsample_val is not None:
                log_dict["meta/n_random_subsample_val"] = self.n_random_subsample_val

            if self.config["training"]["visualize_original_embeddings"]:
                # visualize the original embedding space
                fig_emb_image = embedding_plot(
                    X=layer_image_features_val.float().numpy(),
                    return_figure=True,
                )
                fig_emb_text = embedding_plot(
                    X=layer_text_features_val.float().numpy(),
                    return_figure=True,
                )
                log_dict[f"{layer_comb_str}/embedding_plot/val_original_emb_image"] = (
                    wandb.Image(fig_emb_image)
                )
                log_dict[f"{layer_comb_str}/embedding_plot/val_original_emb_text"] = (
                    wandb.Image(fig_emb_text)
                )
                plt.close(fig_emb_image)
                plt.close(fig_emb_text)
            if self.wandb_logging:
                wandb.log(log_dict)
            del log_dict

            logger.info(
                f"Training alignment for layers {layer_comb} (score: {layer_comb_score:.4f})"
            )

            # define the loss function
            self.loss = CLIPLoss(
                **self.config["training"]["clip_loss"],
            )
            self.loss = self.loss.to(self.device)

            alignment_image = AlignmentFactory.create(
                self.config["training"]["alignment_layer_name"],
                input_dim=image_dim,
                **self.config["training"]["alignment_layer_kwargs"],
            )
            alignment_text = AlignmentFactory.create(
                self.config["training"]["alignment_layer_name"],
                input_dim=text_dim,
                **self.config["training"]["alignment_layer_kwargs"],
            )
            if self.config["training"]["wandb_watch"]:
                wandb.watch(models=[alignment_image, alignment_text], log="all")

            if self.print_model_summary:
                print("*" * 20 + " Alignment Image " + "*" * 20)
                input_size = (self.train_batch_size,)
                summary(
                    alignment_image,
                    input_size=input_size + (image_dim,),
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                )
                print("*" * 20 + " Alignment Text " + "*" * 20)
                summary(
                    alignment_text,
                    input_size=input_size + (text_dim,),
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                )

            optimizer_cls = get_optimizer_type(
                optimizer_name=self.config["training"]["optimizer_name"].lower(),
            )
            if self.config["training"].get("use_lr_finder", False):
                logger.debug("Running learning rate finder...")
                optimal_lr = self.find_optimal_learning_rate(
                    image_features_train=layer_image_features_train,
                    text_features_train=layer_text_features_train,
                    alignment_image=alignment_image,
                    alignment_text=alignment_text,
                    optimizer_cls=optimizer_cls,
                    wandb_prefix=f"{layer_comb_str}/",
                    **self.config["training"]["lr_finder"],
                )
                logger.debug(f"LR finder complete. Using learning rate: {optimal_lr}")
                self.config["training"]["learning_rate"] = optimal_lr

        params = list(alignment_image.parameters()) + list(
            alignment_text.parameters()
        )
        loss_params = [p for p in self.loss.parameters() if p.requires_grad]
        if loss_params:
            params += loss_params

            optimizer = optimizer_cls(
                params=params,
                lr=self.config["training"]["learning_rate"],
                **self.config["training"]["optimizer_kwargs"],
            )
            if self.config["training"]["scheduler_name"] is None:
                scheduler = None
            if self.config["training"]["scheduler_name"] == "WarmupDecayLR":
                scheduler = WarmupDecayLR(
                    optimizer,
                    total_num_steps=self.config["training"]["n_epochs"]
                    * max(
                        (len(layer_image_features_train) // self.train_batch_size), 1
                    ),
                    **self.config["training"]["scheduler_kwargs"],
                )
            elif self.config["training"]["scheduler_name"] == "WarmupLR":
                scheduler = WarmupLR(
                    optimizer,
                    **self.config["training"]["scheduler_kwargs"],
                )
            elif self.config["training"]["scheduler_name"] == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.config["training"]["scheduler_epoch_cycles"]
                    * max(
                        (len(layer_image_features_train) // self.train_batch_size), 1
                    ),
                )
            else:
                raise ValueError(
                    f"Unknown learning rate scheduler: {self.config['training']['scheduler_name']}"
                )

            if self.config["training"]["early_stopping"]:
                early_stopping = EarlyStopping(
                    patience=self.config["training"]["early_stopping_patience"],
                    log_messages=True,
                )
            best_epoch = 0
            best_val_loss = float("inf")
            best_weights_alignment_image = copy.deepcopy(alignment_image.state_dict())
            best_weights_alignment_text = copy.deepcopy(alignment_text.state_dict())

            train_step = 0
            for epoch in (pbar := trange(self.config["training"]["n_epochs"])):
                alignment_image = alignment_image.to(self.device)
                alignment_text = alignment_text.to(self.device)

                steps, train_loss = self.train(
                    epoch=epoch,
                    train_step=train_step,
                    image_features=layer_image_features_train,
                    text_features=layer_text_features_train,
                    alignment_image=alignment_image,
                    alignment_text=alignment_text,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    additional_image_features=layer_additional_image_features,
                    additional_text_features=layer_additional_text_features,
                    wandb_prefix=f"{layer_comb_str}/",
                )
                train_step += steps

                with torch.no_grad():
                    val_loss = self.validate(
                        epoch=epoch,
                        train_step=train_step,
                        image_features=layer_image_features_val,
                        text_features=layer_text_features_val,
                        alignment_image=alignment_image,
                        alignment_text=alignment_text,
                        wandb_prefix=f"{layer_comb_str}/",
                    )
                pbar.set_description(
                    f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = val_loss
                    best_weights_alignment_image = copy.deepcopy(
                        alignment_image.cpu().state_dict()
                    )
                    best_weights_alignment_text = copy.deepcopy(
                        alignment_text.cpu().state_dict()
                    )

                if self.config["training"]["early_stopping"]:
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        break

            if self.config["training"]["early_stopping"]:
                # load the best model (if using early stopping)
                alignment_image.load_state_dict(best_weights_alignment_image)
                alignment_text.load_state_dict(best_weights_alignment_text)

            # save the alignment
            if self.config["training"]["wandb_watch"]:
                wandb.unwatch(models=[alignment_image, alignment_text])
            save_dict = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "train_step": train_step,
                "alignment_text": alignment_text,
                "alignment_image": alignment_image,
                "optimizer": optimizer.state_dict(),
                "config": self.config,
                "loss": self.loss.state_dict(),
            }
            save_checkpoint(
                run_dir=self.save_path
                / wandb.run.name
                / f"{layer_comb}_{layer_comb_score:.4f}",
                save_dict=save_dict,
                epoch=epoch,
            )

            # evaluate
            res_dict = {
                "layer_comb": layer_comb,
                "layer_comb_alignment": layer_comb_score,
                "epoch": epoch,
                "train_step": train_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            with torch.no_grad():
                self.evaluate_retrieval(
                    epoch=epoch,
                    train_step=train_step,
                    alignment_image=alignment_image,
                    alignment_text=alignment_text,
                    alignment_layer_combination=layer_comb,
                    alignment_layer_combination_str=layer_comb_str,
                    additional_result_dict=res_dict,
                )
                gc.collect()
                self.evaluate_zero_shot_classification(
                    epoch=epoch,
                    train_step=train_step,
                    alignment_image=alignment_image,
                    alignment_text=alignment_text,
                    alignment_layer_combination=layer_comb,
                    alignment_layer_combination_str=layer_comb_str,
                    additional_result_dict=res_dict,
                )
        # first stop the wandb run
        wandb.run.finish()
        wandb.finish()

    def train(
        self,
        epoch: int,
        train_step: int,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        alignment_image: torch.nn.Module,
        alignment_text: torch.nn.Module,
        optimizer,
        scheduler=None,
        additional_image_features: Optional[torch.Tensor] = None,
        additional_text_features: Optional[torch.Tensor] = None,
        wandb_prefix: str = "",
    ):
        num_samples = image_features.shape[0]

        # randomly shuffle the embeddings since we didn't shuffle before
        random_sequence = torch.randperm(num_samples)
        image_features = image_features[random_sequence]
        text_features = text_features[random_sequence]

        # NOTE: ablation from reviewers (fixed set for R_S)
        random_sequence_fixed = torch.randperm(num_samples)[: self.train_batch_size]
        image_features_fixed = image_features[random_sequence_fixed]
        text_features_fixed = text_features[random_sequence_fixed]

        # in order to efficiently loop over the splits we use splits and modulo
        if additional_image_features is not None:
            random_sequence = torch.randperm(additional_image_features.shape[0])
            additional_image_features = additional_image_features[random_sequence]
            additional_image_features = torch.split(
                additional_image_features, self.train_batch_size, dim=0
            )
        if additional_text_features is not None:
            random_sequence = torch.randperm(additional_text_features.shape[0])
            additional_text_features = additional_text_features[random_sequence]
            additional_text_features = torch.split(
                additional_text_features, self.train_batch_size, dim=0
            )

        alignment_image.train()
        alignment_text.train()

        l_aligned_image_feats = []
        l_aligned_text_feats = []

        # FuseMix setup
        mixup_alpha = self.config["training"].get("mixup_alpha", 0.0)

        loss_metric = torchmetrics.MeanMetric().to(self.device)
        for i in range(0, num_samples, self.train_batch_size):
            end_i = i + self.train_batch_size
            if end_i > num_samples and mixup_alpha > 0.0:
                continue  # Skip last batch if it's not full, to simplify mixup

            image_feats = image_features[i:end_i]
            text_feats = text_features[i:end_i]
            image_feats = image_feats.float().to(self.device)
            text_feats = text_feats.float().to(self.device)

            if self.config["training"].get("fixed_structure", False):
                image_features_fixed = image_features_fixed.float().to(self.device)
                text_features_fixed = text_features_fixed.float().to(self.device)

            if mixup_alpha > 0.0:
                # To get a second batch, we can simply roll the original tensor
                # This is an efficient way to pair each sample with a different one
                roll_amount = self.train_batch_size // 2
                image_feats2 = torch.roll(image_features, shifts=roll_amount, dims=0)[
                    i:end_i
                ]
                text_feats2 = torch.roll(text_features, shifts=roll_amount, dims=0)[
                    i:end_i
                ]
                image_feats2 = image_feats2.float().to(self.device)
                text_feats2 = text_feats2.float().to(self.device)
                # Apply Mixup
                # Sample a single interpolation coefficient
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                # Create the augmented features by interpolating between the two batches
                image_feats = lam * image_feats + (1 - lam) * image_feats2
                text_feats = lam * text_feats + (1 - lam) * text_feats2

            # step scheduler of the loss function
            self.loss.step()

            # zero out the gradients
            optimizer.zero_grad()

            # forward pass through alignment layers
            aligned_image_feats = alignment_image(image_feats)
            aligned_text_feats = alignment_text(text_feats)

            # additional unimodal data
            loss_kwargs = {}
            if additional_image_features is not None:
                N_splits_img = len(additional_image_features)
                add_image_feats = additional_image_features[i % N_splits_img]
                add_image_feats = add_image_feats.float().to(self.device)
                add_aligned_image_feats = alignment_image(add_image_feats)
                loss_kwargs["add_image_features"] = (
                    add_image_feats,
                    add_aligned_image_feats,
                )

            if additional_text_features is not None:
                N_splits_txt = len(additional_text_features)
                add_text_feats = additional_text_features[i % N_splits_txt]
                add_text_feats = add_text_feats.float().to(self.device)
                add_aligned_text_feats = alignment_text(add_text_feats)
                loss_kwargs["add_text_features"] = (
                    add_text_feats,
                    add_aligned_text_feats,
                )

            if self.config["training"].get("fixed_structure", False):
                # hack to do what they want
                self.loss.structure_use_only_unimodal = True
                aligned_image_features_fixed = alignment_image(image_features_fixed)
                aligned_text_features_fixed = alignment_text(text_features_fixed)
                loss_kwargs["add_image_features"] = (
                    image_features_fixed,
                    aligned_image_features_fixed,
                )
                loss_kwargs["add_text_features"] = (
                    text_features_fixed,
                    aligned_text_features_fixed,
                )

            # backward pass with clip loss
            loss_dict = self.loss(
                image_embeddings_aligned=aligned_image_feats,
                text_embeddings_aligned=aligned_text_feats,
                image_embeddings_original=image_feats,
                text_embeddings_original=text_feats,
                **loss_kwargs,
            )
            loss = loss_dict["overall_loss"]
            loss_metric.update(loss, weight=image_feats.size(0))
            loss.backward()
            clip_grad = self.config["training"]["clip_grad"]
            if clip_grad:
                _ = clip_gradients(alignment_image, clip_grad)
                _ = clip_gradients(alignment_text, clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # speeds up the training by only adding if we have yet to fill up the buffer
            if len(l_aligned_image_feats) * self.train_batch_size < 10_000:
                l_aligned_image_feats.append(aligned_image_feats.detach().cpu())
                l_aligned_text_feats.append(aligned_text_feats.detach().cpu())

            loss_dict = {f"{wandb_prefix}{k}": v for k, v in loss_dict.items()}
            log_dict = loss_dict | {
                f"{wandb_prefix}learning_rate": optimizer.param_groups[0]["lr"],
                f"{wandb_prefix}weight_decay": optimizer.param_groups[0][
                    "weight_decay"
                ],
                f"{wandb_prefix}structure_lambda": self.loss.structure_lambda,
                f"{wandb_prefix}train_loss": loss,
                "counters/epoch": epoch,
                "counters/train_step": train_step + i,
            }
            if self.wandb_logging:
                wandb.log(log_dict)

        log_dict = {
            f"{wandb_prefix}train_loss_avg": loss_metric.compute().item(),
            "counters/epoch": epoch,
            "counters/train_step": train_step + i,
        }
        if (
            self.config["training"].get("log_structural_preservation", False)
            or self.config["training"].get("log_repr_similarity", False)
            or epoch % self.config["training"]["embedding_visualization"] == 0
        ):
            l_aligned_image_feats = torch.cat(l_aligned_image_feats).cpu()
            l_aligned_text_feats = torch.cat(l_aligned_text_feats).cpu()
        if self.config["training"].get("log_structural_preservation", False):
            n_samples = self.config["layer_selection"]["n_samples"]
            for mod, original, aligned in [
                ("image", image_features, l_aligned_image_feats),
                ("text", text_features, l_aligned_text_feats),
            ]:
                for k in self.config["training"].get(
                    "structural_preservation_k", [100]
                ):
                    tw = trustworthiness(
                        X=original[:n_samples].float().to(self.device),
                        Z=aligned[:n_samples].float().to(self.device),
                        k=k,
                        use_approx=True,
                    )
                    cont = continuity(
                        X=original[:n_samples].float().to(self.device),
                        Z=aligned[:n_samples].float().to(self.device),
                        k=k,
                        use_approx=True,
                    )
                    log_dict[f"{wandb_prefix}trustworthiness@{k}_{mod}_train"] = tw
                    log_dict[f"{wandb_prefix}continuity@{k}_{mod}_train"] = cont
        if self.config["training"].get("log_repr_similarity", False):
            n_samples = self.config["layer_selection"]["n_samples"]
            alignment_score_img, _, _ = compute_score(
                x_feats=image_features[:n_samples].float().to(self.device),
                y_feats=l_aligned_image_feats[:n_samples].float().to(self.device),
                metric=self.config["layer_selection"]["metric"],
                show_progress=False,
                **self.config["layer_selection"].get("metric_kwargs", {}),
            )
            alignment_score_txt, _, _ = compute_score(
                x_feats=text_features[:n_samples].float().to(self.device),
                y_feats=l_aligned_text_feats[:n_samples].float().to(self.device),
                metric=self.config["layer_selection"]["metric"],
                show_progress=False,
                **self.config["layer_selection"].get("metric_kwargs", {}),
            )
            log_dict[
                f"{wandb_prefix}{self.config['layer_selection']['metric']}_image_train"
            ] = alignment_score_img
            log_dict[
                f"{wandb_prefix}{self.config['layer_selection']['metric']}_text_train"
            ] = alignment_score_txt
        if epoch % self.config["training"]["embedding_visualization"] == 0:
            l_aligned_feats = torch.cat([l_aligned_image_feats, l_aligned_text_feats])
            l_aligned_targets = np.ones((len(l_aligned_feats),))
            l_aligned_targets[: len(l_aligned_image_feats)] = 0
            label_dict = {0: "images", 1: "texts"}

            fig_emb = embedding_plot(
                X=l_aligned_feats.numpy(),
                y=l_aligned_targets,
                label_dict=label_dict,
                return_figure=True,
            )
            log_dict[f"{wandb_prefix}train_aligned_emb"] = wandb.Image(fig_emb)
            log_dict[f"{wandb_prefix}train_modality_gap"] = (
                l_aligned_image_feats.mean(dim=0) - l_aligned_text_feats.mean(dim=0)
            ).norm(p=2)
            plt.close(fig_emb)
            plt.close("all")

        if self.wandb_logging:
            wandb.log(log_dict)
        del log_dict

        return i, loss_metric.compute().item()

    def validate(
        self,
        epoch: int,
        train_step: int,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        alignment_image: torch.nn.Module,
        alignment_text: torch.nn.Module,
        wandb_prefix: str = "",
    ):
        num_samples = image_features.shape[0]

        alignment_image.eval()
        alignment_text.eval()

        l_aligned_image_feats = []
        l_aligned_text_feats = []

        loss_metric_val = torchmetrics.MeanMetric().to(self.device)
        for i in range(0, num_samples, self.train_batch_size):
            image_feats = image_features[i : i + self.train_batch_size]
            text_feats = text_features[i : i + self.train_batch_size]
            image_feats = image_feats.float().cuda()
            text_feats = text_feats.float().cuda()

            aligned_image_feats = alignment_image(image_feats)
            aligned_text_feats = alignment_text(text_feats)

            # backward pass with clip loss
            loss_dict = self.loss(
                image_embeddings_aligned=aligned_image_feats,
                text_embeddings_aligned=aligned_text_feats,
                image_embeddings_original=image_feats,
                text_embeddings_original=text_feats,
            )
            # compute the median cosine similarity
            cos = torch.nn.functional.cosine_similarity(
                aligned_image_feats, aligned_text_feats, dim=-1
            )
            median_cos = cos.median().item()
            loss = loss_dict["overall_loss"]
            loss_metric_val.update(loss, weight=image_feats.size(0))
            loss_dict = {f"{wandb_prefix}val_{k}": v for k, v in loss_dict.items()}
            loss_dict[f"{wandb_prefix}val_median_cos"] = median_cos
            log_dict = loss_dict | {
                "counters/epoch": epoch,
                "counters/train_step": train_step + i,
            }
            if self.wandb_logging:
                wandb.log(log_dict)

            # speeds up the training by only adding if we have yet to fill up the buffer
            if len(l_aligned_image_feats) * self.train_batch_size < 10_000:
                l_aligned_image_feats.append(aligned_image_feats.cpu())
                l_aligned_text_feats.append(aligned_text_feats.cpu())

        log_dict = {
            f"{wandb_prefix}val_loss_avg": loss_metric_val.compute().item(),
            "counters/epoch": epoch,
            "counters/train_step": train_step,
        }

        if (
            self.config["training"].get("log_repr_similarity", False)
            or self.config["training"].get("log_structural_preservation", False)
            or epoch % self.config["training"]["embedding_visualization"] == 0
        ):
            l_aligned_image_feats = torch.cat(l_aligned_image_feats).cpu()
            l_aligned_text_feats = torch.cat(l_aligned_text_feats).cpu()
        if self.config["training"].get("log_structural_preservation", False):
            n_samples = self.config["layer_selection"]["n_samples"]
            for mod, original, aligned in [
                ("image", image_features, l_aligned_image_feats),
                ("text", text_features, l_aligned_text_feats),
            ]:
                for k in self.config["training"].get(
                    "structural_preservation_k", [100]
                ):
                    tw = trustworthiness(
                        X=original[:n_samples].float().to(self.device),
                        Z=aligned[:n_samples].float().to(self.device),
                        k=k,
                        use_approx=True,
                    )
                    cont = continuity(
                        X=original[:n_samples].float().to(self.device),
                        Z=aligned[:n_samples].float().to(self.device),
                        k=k,
                        use_approx=True,
                    )
                    log_dict[f"{wandb_prefix}trustworthiness@{k}_{mod}_val"] = tw
                    log_dict[f"{wandb_prefix}continuity@{k}_{mod}_val"] = cont
        if self.config["training"].get("log_repr_similarity", False):
            n_samples = self.config["layer_selection"]["n_samples"]
            alignment_score_img, _, _ = compute_score(
                x_feats=image_features[:n_samples].float().to(self.device),
                y_feats=l_aligned_image_feats[:n_samples].float().to(self.device),
                metric=self.config["layer_selection"]["metric"],
                show_progress=False,
                **self.config["layer_selection"].get("metric_kwargs", {}),
            )
            alignment_score_txt, _, _ = compute_score(
                x_feats=text_features[:n_samples].float().to(self.device),
                y_feats=l_aligned_text_feats[:n_samples].float().to(self.device),
                metric=self.config["layer_selection"]["metric"],
                show_progress=False,
                **self.config["layer_selection"].get("metric_kwargs", {}),
            )
            log_dict[
                f"{wandb_prefix}{self.config['layer_selection']['metric']}_image_val"
            ] = alignment_score_img
            log_dict[
                f"{wandb_prefix}{self.config['layer_selection']['metric']}_text_val"
            ] = alignment_score_txt
        if epoch % self.config["training"]["embedding_visualization"] == 0:
            l_aligned_feats = torch.cat([l_aligned_image_feats, l_aligned_text_feats])
            l_aligned_targets = np.ones((len(l_aligned_feats),))
            l_aligned_targets[: len(l_aligned_image_feats)] = 0
            label_dict = {0: "images", 1: "texts"}

            fig_emb = embedding_plot(
                X=l_aligned_feats.numpy(),
                y=l_aligned_targets,
                label_dict=label_dict,
                return_figure=True,
            )
            log_dict[f"{wandb_prefix}val_aligned_emb"] = wandb.Image(fig_emb)
            log_dict[f"{wandb_prefix}val_modality_gap"] = (
                l_aligned_image_feats.mean(dim=0) - l_aligned_text_feats.mean(dim=0)
            ).norm(p=2)
            plt.close(fig_emb)
            plt.close("all")

        if self.wandb_logging:
            wandb.log(log_dict)
        del log_dict

        return loss_metric_val.compute().item()

    def evaluate_zero_shot_classification(
        self,
        epoch: int,
        train_step: int,
        alignment_image: torch.nn.Module,
        alignment_text: torch.nn.Module,
        alignment_layer_combination: Tuple[int, int],
        alignment_layer_combination_str: str,
        additional_result_dict: Dict[str, str],
    ):
        result_dict = additional_result_dict.copy()
        image_layer_idx, text_layer_idx = alignment_layer_combination
        if self.eval_zero_shot_datasets is None:
            return

        # move the layers and set evaluation mode
        alignment_image.eval()
        alignment_text.eval()

        alignment_image = alignment_image.to(self.device)
        alignment_text = alignment_text.to(self.device)

        vision_model, image_transform = self.get_lvm(lvm_model_name=self.lvm_model_name)
        language_model, tokenizer = self.get_llm(llm_model_name=self.llm_model_name)
        for eval_dataset_name, e_dataset in self.eval_zero_shot_datasets:
            set_transform_dataset(
                dataset=e_dataset,
                image_transform=image_transform,
            )

            save_path_vision = AlignmentTrainer.get_feature_save_path(
                m_name=self.lvm_model_name,
                d_name=eval_dataset_name,
                save_path=self.save_path,
                suffix=f"eval-{self.config['features']['pool_img']}",
            )
            save_path_language = AlignmentTrainer.get_feature_save_path(
                m_name=self.llm_model_name,
                d_name=eval_dataset_name,
                save_path=self.save_path,
                suffix=f"eval-{self.config['features']['pool_txt']}",
            )

            dataset_key = eval_dataset_name.lower()
            dataset_classes = DATASETS_TO_CLASSES.get(dataset_key)
            if dataset_classes is None:
                if hasattr(e_dataset, "classes"):
                    dataset_classes = list(e_dataset.classes)
                else:
                    raise KeyError(
                        f"Missing class list for zero-shot dataset: {eval_dataset_name}"
                    )
            zero_shot_classifier = build_zero_shot_classifier(
                language_model=language_model,
                alignment_layer=alignment_text,
                tokenizer=tokenizer,
                dataset=e_dataset,
                layer_index=text_layer_idx,
                classnames=dataset_classes,
                templates=(
                    DATASETS_TO_TEMPLATES.get(dataset_key, SIMPLE_PROMPT_TEMPLATE)
                    if self.config["evaluation"]["use_extended_prompts"]
                    else SIMPLE_PROMPT_TEMPLATE
                ),
                num_classes_per_batch=self.config["evaluation"][
                    "num_classes_per_batch"
                ],
                device=self.device,
                pool_txt=self.config["features"]["pool_txt"],
                save_path=save_path_language,
                sample_by_sample_embedding=self.config["evaluation"][
                    "sample_by_sample_embedding"
                ],
            )
            # we move it to the cpu since in the loop we move chunks back
            # (used to optimize memory for big models)
            zero_shot_classifier = zero_shot_classifier.cpu()

            eval_loader = DataLoader(
                e_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.config["evaluation"]["num_workers"],
                drop_last=False,
                shuffle=False,
                pin_memory=False,
            )

            if save_path_vision is not None and save_path_vision.exists():
                cached = True
                feature_dataset = FeatureDataset(
                    feature_file=save_path_vision,
                    feature_name="features",
                    target_name="targets",
                )
                feature_loader = DataLoader(
                    feature_dataset,
                    batch_size=self.eval_batch_size,
                    num_workers=self.config["evaluation"]["num_workers"],
                    drop_last=False,
                    shuffle=False,
                    pin_memory=False,
                )
            else:
                cached = False
                lvm_feats = []

            i = 0
            all_targets = []

            metrics_kwargs = {"task": "multiclass", "num_classes": len(dataset_classes)}
            metrics_dict = {
                "top1_acc_micro": torchmetrics.classification.Accuracy(
                    top_k=1,
                    average="micro",
                    **metrics_kwargs,
                ),
                "top1_acc_macro": torchmetrics.classification.Accuracy(
                    top_k=1,
                    average="macro",
                    **metrics_kwargs,
                ),
                "top1_f1_micro": torchmetrics.classification.F1Score(
                    top_k=1,
                    average="micro",
                    **metrics_kwargs,
                ),
                "top1_f1_macro": torchmetrics.classification.F1Score(
                    top_k=1,
                    average="macro",
                    **metrics_kwargs,
                ),
                "top1_f1_weighted": torchmetrics.classification.F1Score(
                    top_k=1,
                    average="weighted",
                    **metrics_kwargs,
                ),
                "top1_f1_per_class": torchmetrics.classification.F1Score(
                    top_k=1,
                    average="none",
                    **metrics_kwargs,
                ),
                "confusion_matrix": torchmetrics.ConfusionMatrix(
                    **metrics_kwargs,
                ),
            }
            if len(dataset_classes) >= 5:
                metrics_dict = metrics_dict | {
                    "top5_acc_micro": torchmetrics.classification.Accuracy(
                        top_k=5,
                        average="micro",
                        **metrics_kwargs,
                    ),
                    "top5_acc_macro": torchmetrics.classification.Accuracy(
                        top_k=5,
                        average="macro",
                        **metrics_kwargs,
                    ),
                }

            l_original_image_feats = []
            l_aligned_image_feats = []
            for batch in tqdm(
                feature_loader if cached else eval_loader,
                total=len(eval_loader),
                desc=eval_dataset_name,
                file=sys.stdout,
            ):
                if cached:
                    lvm_output, target = batch
                    lvm_output = lvm_output.to(self.device)
                else:
                    if len(batch) == 2:
                        images, target = batch
                    elif len(batch) == 3:
                        images, _, target = batch
                    else:
                        raise ValueError(f"Unknown length of batch: {len(batch)}")

                    images = images.to(self.device, non_blocking=True)
                    lvm_output = vision_model(images)
                    if self.config["features"]["pool_img"] == "cls":
                        # extract the class token for all layers
                        lvm_output = [v[:, 0, :] for v in lvm_output.values()]
                        lvm_output = torch.stack(lvm_output).permute(1, 0, 2)
                    else:
                        raise NotImplementedError(
                            f"unknown pooling {self.config['features']['pool_img']}"
                        )
                    lvm_feats.append(lvm_output.cpu())

                # lvm_output = (batch_size, dim)
                lvm_output = lvm_output[:, image_layer_idx, :].float()
                l_original_image_feats.append(lvm_output.cpu())

                lvm_output = alignment_image(lvm_output)
                lvm_output = safe_normalize(lvm_output, p=2, dim=-1)
                l_aligned_image_feats.append(lvm_output.cpu())

                # compute the logits by measuring the similarity
                logits = 100.0 * chunked_logits(
                    lvm_output,
                    zero_shot_classifier,
                    device=self.device,
                )
                all_targets.append(target.detach().cpu().numpy())
                for m in metrics_dict.values():
                    m.update(logits.cpu(), target.cpu())
                i += self.eval_batch_size

            if (
                not cached
                and save_path_vision is not None
                and not save_path_vision.exists()
            ):
                lvm_feats = torch.cat(lvm_feats).cpu()
                save_path_vision.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"features": lvm_feats, "targets": np.concatenate(all_targets)}
                    | get_meta_dict(e_dataset),
                    save_path_vision,
                )
                logger.debug(f"Saved eval features to: {save_path_vision}")

            if self.config["evaluation"].get("log_structural_preservation", False):
                n_samples = self.config["layer_selection"]["n_samples"]
                l_original_image_feats = torch.cat(l_original_image_feats).cpu()
                l_aligned_image_feats = torch.cat(l_aligned_image_feats).cpu()
                for mod, original, aligned in [
                    ("image", l_original_image_feats, l_aligned_image_feats),
                ]:
                    for k in self.config["evaluation"].get(
                        "structural_preservation_k", [100]
                    ):
                        tw = trustworthiness(
                            X=original[:n_samples].float().to(self.device),
                            Z=aligned[:n_samples].float().to(self.device),
                            k=k,
                            use_approx=True,
                        )
                        cont = continuity(
                            X=original[:n_samples].float().to(self.device),
                            Z=aligned[:n_samples].float().to(self.device),
                            k=k,
                            use_approx=True,
                        )
                        result_dict[
                            f"{eval_dataset_name}/trustworthiness@{k}_{mod}_train"
                        ] = tw
                        result_dict[
                            f"{eval_dataset_name}continuity@{k}_{mod}_train"
                        ] = cont

            log_str = f"{eval_dataset_name.capitalize()} -"
            for m_name, m in metrics_dict.items():
                if "per_class" in m_name:
                    score = m.compute().detach().numpy().tolist()
                    logger.info(m_name)
                    for i, s in enumerate(score):
                        logger.info(f"  - Class {dataset_classes[i]}: {s:.4f}")
                    result_dict[f"{eval_dataset_name}/{m_name}/std"] = (
                        torch.std(m.compute()).detach().item()
                    )
                elif "confusion_matrix" in m_name:
                    if len(dataset_classes) < 30:
                        fig_, ax_ = m.plot(labels=dataset_classes)
                        result_dict[f"{eval_dataset_name}/{m_name}"] = wandb.Image(fig_)
                        plt.close(fig_)
                        plt.close("all")
                else:
                    score = m.compute().item()
                    log_str += f" {m_name}: {score:.3f},"
                    result_dict[f"{eval_dataset_name}/{m_name}"] = score
            if self._should_save_embeddings(
                alignment_layer_combination, eval_dataset_name
            ):
                if type(l_aligned_image_feats) is not torch.Tensor:
                    l_aligned_image_feats = torch.cat(l_aligned_image_feats).cpu()
                payload = {
                    "image_embeds": l_aligned_image_feats.cpu(),
                    "text_embeds": zero_shot_classifier.cpu(),
                    "targets": np.concatenate(all_targets),
                    "classnames": dataset_classes,
                    "layer_combination": alignment_layer_combination,
                }
                self._save_embeddings(
                    dataset_name=eval_dataset_name,
                    alignment_layer_combination=alignment_layer_combination,
                    suffix="zero_shot",
                    payload=payload,
                )
            logger.info(log_str[:-1])
            log_dict = {
                f"{alignment_layer_combination_str}/{k}": v
                for k, v in result_dict.items()
            } | {
                "counters/epoch": epoch,
                "counters/train_step": train_step,
            }
            if self.config["evaluation"]["plot_embedding_space"]:
                if type(l_aligned_image_feats) is not torch.Tensor:
                    l_aligned_image_feats = torch.cat(l_aligned_image_feats).cpu()
                fig_emb = embedding_plot_w_markers(
                    X=l_aligned_image_feats.numpy(),
                    y=np.concatenate(all_targets),
                    text_X=zero_shot_classifier.cpu().numpy(),
                    text_y=np.arange(len(dataset_classes)),
                    label_dict={i: x for i, x in enumerate(dataset_classes)},
                )
                log_dict[
                    f"{alignment_layer_combination_str}/{eval_dataset_name}/val_aligned_emb"
                ] = wandb.Image(fig_emb)
                plt.close(fig_emb)
                plt.close("all")

            if self.wandb_logging:
                wandb.log(log_dict)
            del log_dict

        if self.df_scores_zero_shot is None:
            self.df_scores_zero_shot = pd.DataFrame(columns=list(result_dict.keys()))
        self.df_scores_zero_shot.loc[len(self.df_scores_zero_shot)] = pd.Series(
            result_dict
        )
        run_name = wandb.run.name if wandb.run is not None else "offline"
        self.df_scores_zero_shot.to_csv(
            f"{self.save_path / run_name / self.add_exp_suffix_to_name('zero_shot_results')}.csv",
            index=False,
        )

    def evaluate_retrieval(
        self,
        epoch: int,
        train_step: int,
        alignment_image: torch.nn.Module,
        alignment_text: torch.nn.Module,
        alignment_layer_combination: Tuple[int, int],
        alignment_layer_combination_str: str,
        additional_result_dict: Dict[str, str],
    ):
        result_dict = additional_result_dict.copy()
        image_layer_idx, text_layer_idx = alignment_layer_combination
        if self.eval_retrieval_datasets is None:
            return

        # move the layers and set evaluation mode
        alignment_image.eval()
        alignment_text.eval()

        alignment_image = alignment_image.to(self.device)
        alignment_text = alignment_text.to(self.device)

        for eval_dataset_name, e_dataset in self.eval_retrieval_datasets:
            eval_loader = DataLoader(
                e_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.config["evaluation"]["num_workers"],
                drop_last=False,
                shuffle=False,
                pin_memory=False,
            )
            image_features_val = self.get_image_features(
                loader=eval_loader,
                lvm_model_name=self.lvm_model_name,
                suffix=f"eval-{self.config['features']['pool_img']}",
            )
            text_features_val = self.get_text_features(
                loader=eval_loader,
                llm_model_name=self.llm_model_name,
                suffix=f"eval-{self.config['features']['pool_txt']}",
            )
            # drop duplicates for fair comparison
            if (
                self.config["evaluation"]["drop_duplicates"]
                and hasattr(eval_loader.dataset, "df")
                and "image_path" in eval_loader.dataset.df.columns
            ):
                unique_val_indices = eval_loader.dataset.df.drop_duplicates(
                    subset="image_path"
                ).index
                image_features_val = image_features_val[unique_val_indices]
                text_features_val = text_features_val[unique_val_indices]

            df = e_dataset.df if hasattr(e_dataset, "df") else None
            subset_cfg = self.config["evaluation"].get("retrieval_subset", {})
            if subset_cfg:
                subset_size = subset_cfg.get("size")
                if subset_size and subset_size < len(image_features_val):
                    seed = subset_cfg.get("seed", self.config["random_state"])
                    rng = np.random.default_rng(seed)
                    subset_indices = rng.choice(
                        len(image_features_val), size=subset_size, replace=False
                    )
                    subset_indices = np.sort(subset_indices)
                    image_features_val = image_features_val[subset_indices]
                    text_features_val = text_features_val[subset_indices]
                    if df is not None:
                        df = df.iloc[subset_indices].reset_index(drop=True)
                        logger.info(
                            f"Using retrieval subset of {len(df)} samples for {eval_dataset_name}"
                        )
            num_samples = image_features_val.shape[0]

            aligned_image_feats = []
            aligned_text_feats = []
            for i in tqdm(
                range(0, num_samples, self.eval_batch_size),
                total=num_samples,
                desc=eval_dataset_name,
                file=sys.stdout,
            ):
                image_feats = image_features_val[
                    i : i + self.eval_batch_size, image_layer_idx
                ]
                text_feats = text_features_val[
                    i : i + self.eval_batch_size, text_layer_idx
                ]
                image_feats = image_feats.float().to(self.device)
                text_feats = text_feats.float().to(self.device)

                image_feats = alignment_image(image_feats)
                text_feats = alignment_text(text_feats)

                aligned_image_feats.append(image_feats)
                aligned_text_feats.append(text_feats)

            aligned_image_feats = torch.cat(aligned_image_feats).cpu()
            aligned_text_feats = torch.cat(aligned_text_feats).cpu()
            if self._should_save_embeddings(
                alignment_layer_combination, eval_dataset_name
            ):
                payload = {
                    "image_embeds": aligned_image_feats,
                    "text_embeds": aligned_text_feats,
                    "layer_combination": alignment_layer_combination,
                }
                if df is not None:
                    payload["dataframe"] = df
                self._save_embeddings(
                    dataset_name=eval_dataset_name,
                    alignment_layer_combination=alignment_layer_combination,
                    suffix="retrieval",
                    payload=payload,
                )

            recalls_i2t = retrieval_metrics_df(
                image_embeds=aligned_image_feats,
                text_embeds=aligned_text_feats,
                df=df,
                image_column="image_path",
                k_values=[1, 5, 10],
                batch_size=self.eval_batch_size,
            )
            recalls_t2i = retrieval_metrics_df(
                image_embeds=aligned_text_feats,
                text_embeds=aligned_image_feats,
                df=df,
                image_column="image_path",
                k_values=[1, 5, 10],
                batch_size=self.eval_batch_size,
            )
            recalls_i2t = {f"I2T-{k}": v for k, v in recalls_i2t.items()}
            recalls_t2i = {f"T2I-{k}": v for k, v in recalls_t2i.items()}
            recalls = recalls_i2t | recalls_t2i
            if "I2T-R@1" in recalls and "T2I-R@1" in recalls:
                recalls["R@1-avg"] = (recalls["I2T-R@1"] + recalls["T2I-R@1"]) / 2

            alignment_cfg = self.config["evaluation"].get("alignment_metrics", {})
            alignment_enabled = (
                alignment_cfg
                if isinstance(alignment_cfg, bool)
                else alignment_cfg.get("enabled", False)
            )
            if alignment_enabled:
                if df is None:
                    logger.warning(
                        f"Skipping alignment metrics for {eval_dataset_name}: missing dataframe"
                    )
                else:
                    label_column = (
                        alignment_cfg.get("label_column")
                        if isinstance(alignment_cfg, dict)
                        else None
                    )
                    if label_column is None:
                        if "image_id" in df.columns:
                            label_column = "image_id"
                        elif "image_path" in df.columns:
                            label_column = "image_path"
                        elif "image_name" in df.columns:
                            label_column = "image_name"
                    if label_column is None or label_column not in df.columns:
                        logger.warning(
                            f"Skipping alignment metrics for {eval_dataset_name}: "
                            f"label column not found"
                        )
                    else:
                        labels = df[label_column].astype(str).tolist()
                        purity = compute_cross_modal_purity_score(
                            aligned_image_feats,
                            aligned_text_feats,
                            labels,
                            batch_size=alignment_cfg.get(
                                "batch_size", self.eval_batch_size
                            ),
                        )
                        pooled_embeds = torch.cat(
                            [aligned_image_feats, aligned_text_feats], dim=0
                        )
                        pooled_labels = labels + labels
                        silhouette = compute_silhouette_score(
                            pooled_embeds,
                            pooled_labels,
                            metric=alignment_cfg.get("silhouette_metric", "cosine"),
                            sample_size=alignment_cfg.get("silhouette_sample_size"),
                            random_state=alignment_cfg.get(
                                "seed", self.config["random_state"]
                            ),
                        )
                        recalls["Purity"] = purity
                        recalls["Silhouette"] = silhouette

            log_str = f"{eval_dataset_name.capitalize()} -"
            for m_name, score in recalls.items():
                log_str += f" {m_name}: {score:.3f},"
                result_dict[f"{eval_dataset_name}/{m_name}"] = score
            logger.info(log_str[:-1])
            log_dict = {
                f"{alignment_layer_combination_str}/{k}": v
                for k, v in result_dict.items()
            } | {
                "counters/epoch": epoch,
                "counters/train_step": train_step,
            }

            if self.config["evaluation"]["plot_embedding_space"]:
                l_aligned_feats = torch.cat(
                    [aligned_image_feats, aligned_text_feats]
                ).cpu()
                l_aligned_targets = np.ones((len(l_aligned_feats),))
                l_aligned_targets[: len(aligned_image_feats)] = 0
                label_dict = {0: "images", 1: "texts"}

                fig_emb = embedding_plot(
                    X=l_aligned_feats.numpy(),
                    y=l_aligned_targets,
                    label_dict=label_dict,
                    return_figure=True,
                )
                log_dict[
                    f"{alignment_layer_combination_str}/{eval_dataset_name}/val_aligned_emb"
                ] = wandb.Image(fig_emb)
                log_dict[
                    f"{alignment_layer_combination_str}/{eval_dataset_name}/modality_gap"
                ] = (
                    aligned_image_feats.mean(dim=0) - aligned_text_feats.mean(dim=0)
                ).norm(
                    p=2
                )
                plt.close(fig_emb)
                plt.close("all")

            if self.wandb_logging:
                wandb.log(log_dict)
            del log_dict

        if self.df_scores_retrieval is None:
            self.df_scores_retrieval = pd.DataFrame(columns=list(result_dict.keys()))
        self.df_scores_retrieval.loc[len(self.df_scores_retrieval)] = pd.Series(
            result_dict
        )
        run_name = wandb.run.name if wandb.run is not None else "offline"
        self.df_scores_retrieval.to_csv(
            f"{self.save_path / run_name / self.add_exp_suffix_to_name('retrieval_results')}.csv",
            index=False,
        )
