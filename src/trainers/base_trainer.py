import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger
from matplotlib.ticker import LogLocator
from torch.utils.data import DataLoader

from src.core.src.utils.utils import fix_random_seeds


class Trainer(ABC, object):
    def __init__(
        self,
        config: dict,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        experiment_name: Optional[str] = "",
        wandb_logging: bool = True,
        wandb_project_name: str = "representation-alignment",
        wandb_notes: Optional[str] = None,
    ):
        self.config = config
        self.arch_name = experiment_name
        self.device = self._get_device()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.start_epoch = 1

        # useful configs
        self.train_batch_size = self.config["training"]["batch_size"]
        self.eval_batch_size = self.config["evaluation"]["batch_size"]

        self.n_random_subsample_train: Optional[int] = None
        self.n_random_subsample_val: Optional[int] = None

        fix_random_seeds(seed=self.config["random_state"])

        self.wandb_logging = wandb_logging
        if self.wandb_logging:
            import wandb

            if wandb.run is None:
                wandb.init(
                    config=self.config,
                    project=wandb_project_name,
                    group=experiment_name,
                    notes=wandb_notes,
                )
                run_name = f"{experiment_name}-{wandb.run.name}"
                wandb.run.name = run_name
                try:
                    wandb.run.save()
                except TypeError:
                    # wandb>=0.24 expects a glob string; skip to avoid crash
                    logger.debug("wandb.run.save() requires glob_str; skipping.")
            self.run_dir = Path(wandb.run.dir)
        else:
            current_directory = self.config.get("work_dir", os.getcwd())
            final_directory = os.path.join(
                current_directory,
                f"{experiment_name}",
            )
            if not os.path.exists(final_directory):
                os.makedirs(final_directory)
            self.run_dir = Path(final_directory)
            logger.debug(f"Run directory of model: {self.run_dir}")

        # used when using tokenizers in parallel operations
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # allow backward pass of the autograd engine to print traceback
        # of the forward operation that created the failing backward operation
        torch.autograd.set_detect_anomaly(True)
        # optimize various tensor operations automatically
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if self.train_dataset is not None:
            logger.debug(
                f"Data loaded: there are "
                f"{len(self.train_dataset.dataset)} train images and "
                f"{len(self.train_dataset)} batches "
                f"with a batch size of {self.config['training']['batch_size']}."
            )
        if self.val_dataset is not None:
            no_val_samples = len(self.val_dataset.dataset)
            if hasattr(self.val_dataset, "sampler"):
                no_val_samples = len(self.val_dataset.sampler)
            logger.debug(
                f"Data loaded: there are "
                f"{no_val_samples} val images and "
                f"{len(self.val_dataset)} batches "
                f"with a batch size of {self.config['training']['batch_size']}."
            )

    @abstractmethod
    def fit(
        self,
        n_random_subsample_train: Optional[int] = None,
        n_random_subsample_val: Optional[int] = None,
        additional_unimodal_data: Optional[Dict[str, list]] = None,
    ):
        pass

    @property
    def get_ckp_path(self) -> Path:
        if self.config["training"].get("fine_tune_from"):
            return (
                self.run_dir / self.config["training"]["fine_tune_from"] / "checkpoints"
            )
        else:
            return self.run_dir / "checkpoints"

    def _get_device(self):
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )
        logger.debug(f"Running on: {device}")
        return device

    def update_optim_from_schedulers(
        self,
        optimizer,
        lr_schedule,
        wd_schedule,
        n_iter: int,
    ):
        # update weight decay and LR according to their schedule
        # but only if wanted
        for i, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None and self.config["training"].get(
                "use_lr_scheduler", True
            ):
                param_group["lr"] = lr_schedule[n_iter]
            if i == 0:  # only the first group is regularized
                if wd_schedule is not None and self.config["training"].get(
                    "use_wd_scheduler", True
                ):
                    param_group["weight_decay"] = wd_schedule[n_iter]

    def find_optimal_learning_rate(
        self,
        image_features_train: torch.Tensor,
        text_features_train: torch.Tensor,
        alignment_image: torch.nn.Module,
        alignment_text: torch.nn.Module,
        optimizer_cls,
        num_iter: int = 100,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        wandb_prefix: str = "",
    ):
        # save original model state to restore after finding LR
        alignment_image_state = deepcopy(alignment_image.state_dict())
        alignment_text_state = deepcopy(alignment_text.state_dict())

        alignment_image = alignment_image.to(self.device).train()
        alignment_text = alignment_text.to(self.device).train()

        # create optimizer with very small learning rate
        params = list(alignment_image.parameters()) + list(alignment_text.parameters())
        optimizer = optimizer_cls(
            params=params,
            lr=start_lr,
            **self.config["training"]["optimizer_kwargs"],
        )

        # create a log-spaced learning rate schedule
        mult = (end_lr / start_lr) ** (1 / num_iter)

        lrs = []
        losses = []
        best_loss = float("inf")

        # use a subset of data for faster evaluation
        subset_size = min(image_features_train.shape[0], 5000)
        indices = torch.randperm(image_features_train.shape[0])[:subset_size]
        image_features_subset = image_features_train[indices].float()
        text_features_subset = text_features_train[indices].float()

        batch_size = min(self.train_batch_size, subset_size)
        logger.debug(
            f"Running simple LR finder with {num_iter} iterations from {start_lr} to {end_lr}"
        )

        try:
            for i in range(num_iter):
                batch_idx = torch.randperm(subset_size)[:batch_size]
                img_feats = image_features_subset[batch_idx].to(self.device)
                txt_feats = text_features_subset[batch_idx].to(self.device)

                optimizer.zero_grad()

                aligned_img_feats = alignment_image(img_feats)
                aligned_txt_feats = alignment_text(txt_feats)

                loss_dict = self.loss(
                    image_embeddings_aligned=aligned_img_feats,
                    text_embeddings_aligned=aligned_txt_feats,
                    image_embeddings_original=img_feats,
                    text_embeddings_original=txt_feats,
                )
                loss = loss_dict["overall_loss"]
                loss.backward()

                lr = optimizer.param_groups[0]["lr"]
                lrs.append(lr)
                loss_value = loss.item()
                losses.append(loss_value)

                if loss_value < best_loss:
                    best_loss = loss_value

                # check for loss divergence (5x the best loss)
                if loss_value > 5 * best_loss or not np.isfinite(loss_value):
                    logger.debug(f"Loss diverged at lr={lr:.7f}, stopping early")
                    break

                optimizer.step()

                for param_group in optimizer.param_groups:
                    param_group["lr"] *= mult

                if i % 10 == 0 or i == num_iter - 1:
                    logger.debug(
                        f"Iter {i}/{num_iter}: LR: {lr:.7f}, Loss: {loss_value:.5f}"
                    )

            alignment_image.load_state_dict(alignment_image_state)
            alignment_text.load_state_dict(alignment_text_state)

            if len(losses) <= 1:
                logger.warning("LR finder didn't collect enough data points")
                return self.config["training"]["learning_rate"]

            losses = np.array(losses)
            lrs = np.array(lrs)

            # smoothing
            smoothed_losses = []
            smooth_factor = 0.05
            for i, loss in enumerate(losses):
                if i == 0:
                    smoothed_losses.append(loss)
                else:
                    smoothed_losses.append(
                        smoothed_losses[-1] * smooth_factor + loss * (1 - smooth_factor)
                    )
            smoothed_losses = np.array(smoothed_losses)

            gradients = np.gradient(
                smoothed_losses, np.log10(lrs)
            )  # Take gradient with respect to log10 of LR
            min_loss_idx = np.argmin(smoothed_losses)

            # find steepest descent point
            # if we have enough data before min_loss_idx
            valid_indices = list(range(1, min(min_loss_idx, len(gradients) - 1)))
            if len(valid_indices) > 0:
                steep_idx = valid_indices[np.argmin(gradients[valid_indices])]
            else:
                steep_idx = max(0, min_loss_idx - 1)

            steep_lr = lrs[steep_idx]

            # calculate suggested LR (divide by factor for stability)
            lr_divisor = 5.0  # Common heuristic
            suggested_lr = steep_lr / lr_divisor

            # ensure we don't go lower than our starting point
            suggested_lr = max(suggested_lr, start_lr * 10)

            if self.wandb_logging:
                fig = self.plot_lr_finder_results(
                    lrs=lrs,
                    losses=losses,
                    smoothed_losses=smoothed_losses,
                    gradients=gradients,
                    min_loss_idx=min_loss_idx,
                    steep_idx=steep_idx,
                    suggested_lr=suggested_lr,
                )
                wandb.log({f"{wandb_prefix}lr_finder_visualization": wandb.Image(fig)})
                plt.close(fig)

            logger.debug(f"Simple LR finder suggests learning rate: {suggested_lr:.7f}")
            return suggested_lr

        except Exception as e:
            logger.error(f"Simple LR finder failed with error: {str(e)}")
            alignment_image.load_state_dict(alignment_image_state)
            alignment_text.load_state_dict(alignment_text_state)
            return self.config["training"]["learning_rate"]

    def plot_lr_finder_results(
        self,
        lrs: np.ndarray,
        losses: np.ndarray,
        smoothed_losses: np.ndarray,
        gradients: np.ndarray,
        min_loss_idx: int,
        steep_idx: int,
        suggested_lr: float,
    ):
        """
        Generate a detailed visualization of learning rate finder results.

        Args:
            lrs: Array of learning rates
            losses: Array of raw loss values
            smoothed_losses: Array of smoothed loss values
            gradients: Array of gradients (rate of change of loss)
            min_loss_idx: Index of minimum loss
            steep_idx: Index of steepest descent
            suggested_lr: Suggested learning rate

        Returns:
            matplotlib Figure object
        """

        min_loss_lr = lrs[min_loss_idx]
        steep_lr = lrs[steep_idx]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]}
        )

        # plot 1: Loss vs Learning Rate
        ax1.plot(
            lrs,
            losses,
            "o-",
            color="lightblue",
            alpha=0.6,
            markersize=4,
            label="Raw loss",
        )
        ax1.plot(
            lrs, smoothed_losses, "-", color="blue", linewidth=2, label="Smoothed loss"
        )

        ax1.scatter(
            [min_loss_lr],
            [smoothed_losses[min_loss_idx]],
            color="darkgreen",
            s=100,
            zorder=5,
            label="Minimum loss",
        )
        ax1.scatter(
            [steep_lr],
            [smoothed_losses[steep_idx]],
            color="purple",
            s=100,
            zorder=5,
            label="Steepest descent",
        )
        ax1.axvline(
            x=suggested_lr,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Suggested LR: {suggested_lr:.2e}",
        )
        ax1.axvspan(
            suggested_lr / 3,
            suggested_lr * 3,
            alpha=0.1,
            color="green",
            label="Recommended range",
        )
        ax1.annotate(
            f"Min Loss: {smoothed_losses[min_loss_idx]:.4f}",
            xy=(min_loss_lr, smoothed_losses[min_loss_idx]),
            xytext=(min_loss_lr * 2, smoothed_losses[min_loss_idx] * 0.9),
            arrowprops=dict(arrowstyle="->", color="darkgreen"),
        )
        ax1.annotate(
            f"Steepest: {smoothed_losses[steep_idx]:.4f}",
            xy=(steep_lr, smoothed_losses[steep_idx]),
            xytext=(steep_lr * 2, smoothed_losses[steep_idx] * 1.1),
            arrowprops=dict(arrowstyle="->", color="purple"),
        )

        ax1.set_xscale("log")
        ax1.set_xlabel("Learning Rate", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title(
            "Learning Rate Finder: Loss vs. Learning Rate",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

        min_smooth_loss = np.min(smoothed_losses)
        median_loss = np.median(smoothed_losses[smoothed_losses < min_smooth_loss * 2])
        y_max = min(np.max(smoothed_losses), median_loss * 2)
        ax1.set_ylim(min_smooth_loss * 0.8, y_max)

        ax1.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax1.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1), numticks=10)
        )

        # plot 2: Gradient of loss curve
        ax2.plot(lrs, gradients, "-", color="crimson", linewidth=2)
        ax2.scatter([steep_lr], [gradients[steep_idx]], color="purple", s=100, zorder=5)
        ax2.axvline(x=suggested_lr, color="red", linestyle="--", linewidth=2)

        # gradient plot
        ax2.set_xscale("log")
        ax2.set_xlabel("Learning Rate", fontsize=12)
        ax2.set_ylabel("Gradient (dL/d(log LR))", fontsize=12)
        ax2.set_title("Rate of Change in Loss", fontsize=12)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

        # set x-axis limits to match the top plot
        ax2.set_xlim(ax1.get_xlim())

        # vertical line at minimum gradient
        min_grad_idx = np.argmin(gradients)
        ax2.axvline(x=lrs[min_grad_idx], color="gray", linestyle=":", linewidth=1)

        plt.tight_layout()
        return fig
