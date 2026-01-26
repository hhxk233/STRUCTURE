import gc
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
import torchvision.transforms as transforms
import wandb
from loguru import logger
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.core.src.datasets.image_text_dataset import ImageTextDataset
from src.core.src.utils.plotting import embedding_plot_w_markers
from src.dataset_preparation.data_utils import FeatureDataset, get_meta_dict
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
from src.trainers.alignment_trainer import AlignmentTrainer
from src.utils.utils import set_transform_dataset


class CLIPEvalTrainer(AlignmentTrainer):
    def __init__(
        self,
        config: dict,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        llm_model_name: str,
        lvm_model_name: str,
        eval_zero_shot_datasets: Optional[List[DataLoader]] = None,
        eval_retrieval_datasets: Optional[List[DataLoader]] = None,
        print_model_summary: bool = True,
        wandb_logging: bool = True,
        wandb_project_name: str = "representation-alignment-CSA",
        wandb_notes: Optional[str] = None,
    ):
        super().__init__(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            llm_model_name=llm_model_name,
            lvm_model_name=lvm_model_name,
            eval_zero_shot_datasets=eval_zero_shot_datasets,
            eval_retrieval_datasets=eval_retrieval_datasets,
            print_model_summary=print_model_summary,
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
            wandb_notes=wandb_notes,
        )
        self._pca_model: Optional[PCA] = None

    def _maybe_fit_pca(self, image_features: torch.Tensor, text_features: torch.Tensor):
        pca_dim = self.config["evaluation"].get("pca_dim")
        if not pca_dim:
            return None
        if self._pca_model is not None:
            return self._pca_model
        features = torch.cat([image_features, text_features], dim=0).cpu().numpy()
        pca = PCA(n_components=pca_dim, random_state=self.config["random_state"])
        pca.fit(features)
        self._pca_model = pca
        logger.info(f"Fitted PCA with dim={pca_dim}")
        return self._pca_model

    def _apply_pca(self, features: torch.Tensor):
        if self._pca_model is None:
            return features
        transformed = self._pca_model.transform(features.cpu().numpy())
        return torch.tensor(transformed)

    def fit(
        self,
        n_random_subsample_train: Optional[int] = None,
        n_random_subsample_val: Optional[int] = None,
        additional_unimodal_data: Optional[Dict[str, list]] = None,
    ):
        res_dict = {}
        with torch.no_grad():
            self.evaluate_retrieval(
                epoch=0,
                train_step=0,
                alignment_layer_combination_str="img_-1_txt_-1",
                additional_result_dict=res_dict,
            )
            gc.collect()
            self.evaluate_zero_shot_classification(
                epoch=0,
                train_step=0,
                alignment_layer_combination_str="img_-1_txt_-1",
                additional_result_dict=res_dict,
            )

    def evaluate_zero_shot_classification(
        self,
        epoch: int,
        train_step: int,
        alignment_layer_combination_str: str,
        additional_result_dict: Dict[str, str],
    ):
        result_dict = additional_result_dict.copy()
        if self.eval_zero_shot_datasets is None:
            return

        model = CLIPModel.from_pretrained(self.lvm_model_name).to(self.device)
        processor = CLIPProcessor.from_pretrained(self.lvm_model_name)
        feat_processor = processor.image_processor  # holds mean/std
        language_model = model
        tokenizer = processor.tokenizer
        for eval_dataset_name, e_dataset in self.eval_zero_shot_datasets:
            image_transform = transforms.Compose(
                [
                    transforms.Resize((336, 336)),
                    transforms.ToTensor(),
                ]
            )
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
                tokenizer=tokenizer,
                dataset=e_dataset,
                layer_index=None,
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
            zero_shot_classifier = self._apply_pca(zero_shot_classifier)

            eval_loader = DataLoader(
                e_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.config["evaluation"]["num_workers"],
                drop_last=False,
                shuffle=False,
                pin_memory=True,
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
                    pin_memory=True,
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

            l_aligned_image_feats = []

            pbar = tqdm(
                feature_loader if cached else eval_loader,
                total=len(eval_loader),
                desc=eval_dataset_name,
                file=sys.stdout,
            )
            for batch in pbar:
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
                    # normalize per CLIP spec
                    mean = torch.tensor(
                        feat_processor.image_mean, device=self.device
                    ).view(1, 3, 1, 1)
                    std = torch.tensor(
                        feat_processor.image_std, device=self.device
                    ).view(1, 3, 1, 1)
                    pixel_values = (images.to(self.device) - mean) / std
                    # forward
                    feats = model.get_image_features(pixel_values=pixel_values)
                    # normalize
                    lvm_output = feats / feats.norm(p=2, dim=-1, keepdim=True)
                    lvm_feats.append(lvm_output.cpu())

                # lvm_output = (batch_size, dim)
                lvm_output = lvm_output.float().cpu()
                lvm_output = self._apply_pca(lvm_output)
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

            # save the eval features from the llm if they don't exist yet
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

            log_str = f"{eval_dataset_name.capitalize()} -"
            for m_name, m in metrics_dict.items():
                score = m.compute().item()
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
        self.df_scores_zero_shot.to_csv(
            f"{self.save_path / wandb.run.name / self.add_exp_suffix_to_name('zero_shot_results')}.csv",
            index=False,
        )

    def get_text_features(
        self,
        loader,
        llm_model_name: str,
        suffix: str = "",
        dataset_name: Optional[str] = None,
    ):
        if hasattr(loader.dataset, "name"):
            dataset_name = loader.dataset.name
        elif dataset_name is None:
            dataset_name = type(loader.dataset).__name__
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

        model = CLIPModel.from_pretrained(llm_model_name).to(self.device)
        processor = CLIPProcessor.from_pretrained(llm_model_name)

        llm_feats = []
        for batch in tqdm(loader, total=len(loader), file=sys.stdout):
            if len(batch) == 2:
                _, texts = batch
            elif len(batch) == 3:
                _, texts, _ = batch
            else:
                raise ValueError(f"Unexpected batch item size: {len(batch)}")
            with torch.no_grad():
                inputs = processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(self.device)
                feats = model.get_text_features(**inputs)
                llm_feats.append(feats.cpu())
        llm_feats = torch.cat(llm_feats).cpu()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"features": llm_feats}
        if hasattr(loader.dataset, "df"):
            save_dict["dataframe"] = loader.dataset.df
        torch.save(save_dict, save_path)
        logger.debug(f"Saved features to: {save_path}")
        del model, processor
        return llm_feats

    def get_image_features(
        self,
        loader,
        lvm_model_name: str,
        suffix: str = "",
        dataset_name: Optional[str] = None,
    ):
        if hasattr(loader.dataset, "name"):
            dataset_name = loader.dataset.name
        elif dataset_name is None:
            dataset_name = type(loader.dataset).__name__
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

        model = CLIPModel.from_pretrained(lvm_model_name).to(self.device)
        processor = CLIPProcessor.from_pretrained(lvm_model_name)
        feat_processor = processor.image_processor  # holds mean/std

        image_transform = transforms.Compose(
            [
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
            ]
        )
        set_transform_dataset(
            dataset=loader.dataset,
            image_transform=image_transform,
        )

        lvm_feats = []
        for batch in tqdm(loader, total=len(loader), file=sys.stdout):
            images, _ = batch
            if isinstance(images, list):
                if torch.is_tensor(images[0]):
                    images = torch.stack(images, dim=0)
                else:
                    images = torch.stack(
                        [transforms.ToTensor()(img) for img in images], dim=0
                    )
            with torch.no_grad():
                # normalize per CLIP spec
                mean = torch.tensor(feat_processor.image_mean, device=self.device).view(
                    1, 3, 1, 1
                )
                std = torch.tensor(feat_processor.image_std, device=self.device).view(
                    1, 3, 1, 1
                )
                pixel_values = (images.to(self.device) - mean) / std
                # forward
                feats = model.get_image_features(pixel_values=pixel_values)
                # normalize
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                lvm_feats.append(feats.cpu())
        lvm_feats = torch.cat(lvm_feats).cpu()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"features": lvm_feats}
        if hasattr(loader.dataset, "df"):
            save_dict["dataframe"] = loader.dataset.df
        torch.save(save_dict, save_path)
        logger.debug(f"Saved features to: {save_path}")
        del model, processor, feat_processor
        return lvm_feats

    def evaluate_retrieval(
        self,
        epoch: int,
        train_step: int,
        alignment_layer_combination_str: str,
        additional_result_dict: Dict[str, str],
    ):
        result_dict = additional_result_dict.copy()
        if self.eval_retrieval_datasets is None:
            return

        for eval_dataset_name, e_dataset in self.eval_retrieval_datasets:
            eval_loader = DataLoader(
                e_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.config["evaluation"]["num_workers"],
                drop_last=False,
                shuffle=False,
                pin_memory=True,
                collate_fn=ImageTextDataset.collate_fn,
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
            self._maybe_fit_pca(image_features_val, text_features_val)
            image_features_val = self._apply_pca(image_features_val)
            text_features_val = self._apply_pca(text_features_val)
            recalls_i2t = retrieval_metrics_df(
                image_embeds=image_features_val,
                text_embeds=text_features_val,
                df=df,
                image_column="image_path",
                k_values=[1, 5, 10],
                batch_size=self.eval_batch_size,
            )
            recalls_t2i = retrieval_metrics_df(
                image_embeds=text_features_val,
                text_embeds=image_features_val,
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
                            image_features_val,
                            text_features_val,
                            labels,
                            batch_size=alignment_cfg.get(
                                "batch_size", self.eval_batch_size
                            ),
                        )
                        pooled_embeds = torch.cat(
                            [image_features_val, text_features_val], dim=0
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

            if self.wandb_logging:
                wandb.log(log_dict)
            del log_dict

        if self.df_scores_retrieval is None:
            self.df_scores_retrieval = pd.DataFrame(columns=list(result_dict.keys()))
        self.df_scores_retrieval.loc[len(self.df_scores_retrieval)] = pd.Series(
            result_dict
        )
        self.df_scores_retrieval.to_csv(
            f"{self.save_path / wandb.run.name / self.add_exp_suffix_to_name('retrieval_results')}.csv",
            index=False,
        )
