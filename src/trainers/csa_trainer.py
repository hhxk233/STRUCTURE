import gc
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
import wandb
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.alignment.cca_class import NormalizedCCA
from src.core.src.utils.plotting import embedding_plot, embedding_plot_w_markers
from src.dataset_preparation.data_utils import FeatureDataset, get_meta_dict
from src.evaluation.consts import (
    DATASETS_TO_CLASSES,
    DATASETS_TO_TEMPLATES,
    SIMPLE_PROMPT_TEMPLATE,
)
from src.evaluation.retrieval import retrieval_metrics_df
from src.evaluation.zero_shot_classifier import (
    build_zero_shot_classifier,
    chunked_logits,
)
from src.trainers.alignment_trainer import AlignmentTrainer
from src.utils.utils import safe_normalize, set_transform_dataset


class CSATrainer(AlignmentTrainer):
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

    def fit(
        self,
        n_random_subsample_train: Optional[int] = None,
        n_random_subsample_val: Optional[int] = None,
        additional_unimodal_data: Optional[Dict[str, list]] = None,
    ):
        # pre-compute the embeddings from both modalities
        # first embed the validation set since we're returning
        # the models for the training set
        image_val_suffix = f"val-{self.config['features']['pool_img']}"
        if self.config["features"].get("layer_img") is not None:
            image_val_suffix += f"_layer-{self.config['features']['layer_img']}"
        text_val_suffix = f"val-{self.config['features']['pool_txt']}"
        if self.config["features"].get("layer_img") is not None:
            text_val_suffix += f"_layer-{self.config['features']['layer_txt']}"
        image_features_train = self.get_image_features(
            loader=self.train_dataset,
            lvm_model_name=self.lvm_model_name,
            suffix=image_val_suffix.replace("val-", "train-"),
        )
        text_features_train = self.get_text_features(
            loader=self.train_dataset,
            llm_model_name=self.llm_model_name,
            suffix=text_val_suffix.replace("val-", "train-"),
        )

        # check that we have the same samples
        assert image_features_train.shape[0] == text_features_train.shape[0]

        if (
            self.config["training"]["drop_duplicates"]
            and hasattr(self.train_dataset.dataset, "df")
            and "image_path" in self.train_dataset.dataset.df.columns
        ):
            unique_train_indices = self.train_dataset.dataset.df.drop_duplicates(
                subset="image_path"
            ).index
            image_features_train = image_features_train[unique_train_indices]
            text_features_train = text_features_train[unique_train_indices]

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

        logger.debug(
            f"TRAIN - img: {image_features_train.shape}, txt: {text_features_train.shape}"
        )

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
        comb_iter = sampled_df_alignment.iterrows()
        for i_comb, (_, layer_series) in enumerate(comb_iter):
            layer_comb = layer_series["indices"]
            image_layer_idx, text_layer_idx = layer_comb
            layer_comb_score = layer_series["alignment_score"]
            layer_comb_str = f"img_{image_layer_idx}_txt_{text_layer_idx}"

            layer_image_features_train = image_features_train[:, image_layer_idx, :]
            layer_text_features_train = text_features_train[:, text_layer_idx, :]

            if (
                len(sampled_df_alignment) == 1
                or i_comb == len(sampled_df_alignment) - 1
            ):
                # clean up the memory if we're only doing one comb or its the last
                del image_features_train, text_features_train

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

            logger.info(
                f"Training alignment for layers {layer_comb} (score: {layer_comb_score:.4f})"
            )

            cca_model = NormalizedCCA(
                batch_size=self.train_batch_size,
                device=self.device,
                **self.config["training"]["cca_kwargs"],
            )
            cca_model.fit_transform_train_data(
                layer_image_features_train.float().numpy(),
                layer_text_features_train.float().numpy(),
            )

            # evaluate
            res_dict = {
                "layer_comb": layer_comb,
                "layer_comb_alignment": layer_comb_score,
            }
            with torch.no_grad():
                self.evaluate_retrieval(
                    epoch=0,
                    train_step=0,
                    cca_model=cca_model,
                    alignment_layer_combination=layer_comb,
                    alignment_layer_combination_str=layer_comb_str,
                    additional_result_dict=res_dict,
                )
                gc.collect()
                self.evaluate_zero_shot_classification(
                    epoch=0,
                    train_step=0,
                    cca_model=cca_model,
                    alignment_layer_combination=layer_comb,
                    alignment_layer_combination_str=layer_comb_str,
                    additional_result_dict=res_dict,
                )

    def evaluate_zero_shot_classification(
        self,
        epoch: int,
        train_step: int,
        cca_model,
        alignment_layer_combination: Tuple[int, int],
        alignment_layer_combination_str: str,
        additional_result_dict: Dict[str, str],
    ):
        result_dict = additional_result_dict.copy()
        image_layer_idx, text_layer_idx = alignment_layer_combination
        if self.eval_zero_shot_datasets is None:
            return

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

            dataset_classes = DATASETS_TO_CLASSES[eval_dataset_name.lower()]
            zero_shot_classifier = build_zero_shot_classifier(
                language_model=language_model,
                tokenizer=tokenizer,
                dataset=e_dataset,
                layer_index=text_layer_idx,
                classnames=dataset_classes,
                templates=(
                    DATASETS_TO_TEMPLATES[eval_dataset_name.lower()]
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
                lvm_output = lvm_output[:, image_layer_idx, :].float().cpu()
                lvm_output, zs_clf = cca_model.transform_data(
                    lvm_output.float().numpy(),
                    zero_shot_classifier.float().numpy(),
                )
                lvm_output = torch.Tensor(lvm_output)
                zs_clf = torch.Tensor(zs_clf)
                lvm_output = safe_normalize(lvm_output, p=2, dim=-1)
                l_aligned_image_feats.append(lvm_output.cpu())

                # compute the logits by measuring the similarity
                logits = 100.0 * chunked_logits(
                    lvm_output,
                    zs_clf,
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

    def evaluate_retrieval(
        self,
        epoch: int,
        train_step: int,
        cca_model,
        alignment_layer_combination: Tuple[int, int],
        alignment_layer_combination_str: str,
        additional_result_dict: Dict[str, str],
    ):
        result_dict = additional_result_dict.copy()
        image_layer_idx, text_layer_idx = alignment_layer_combination
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

            aligned_image_feats, aligned_text_feats = cca_model.transform_data(
                image_features_val[:, image_layer_idx, :].float().numpy(),
                text_features_val[:, text_layer_idx, :].float().numpy(),
            )
            aligned_image_feats = torch.Tensor(aligned_image_feats)
            aligned_text_feats = torch.Tensor(aligned_text_feats)

            df = e_dataset.df if hasattr(e_dataset, "df") else None
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
        self.df_scores_retrieval.to_csv(
            f"{self.save_path / wandb.run.name / self.add_exp_suffix_to_name('retrieval_results')}.csv",
            index=False,
        )
