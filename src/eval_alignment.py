import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
import yaml
from loguru import logger

from src.core.src.utils.loader import Loader, merge_dicts
from src.dataset_preparation.data_utils import get_datasets, get_default_transforms
from src.core.src.datasets.image_text_dataset import ImageTextDataset
def load_dataset(
    dataset_name: str,
    data_path: Path,
    batch_size: int = 16,
    num_workers: int = 1,
    label_templates=None,
    template_key: str = "label",
    precompute_captions: bool = True,
):
    if label_templates is None:
        label_templates = ["a photo of a {label}"]
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_dataset, val_dataset = get_datasets(
        dataset=dataset_name,
        transform=transform,
        root_dir=data_path,
    )

    if dataset_name != "coco" and dataset_name != "flickr30":
        train_dataset = ImageTextDataset(
            dataset=train_dataset,
            label_templates=label_templates,
            template_key=template_key,
            precompute_captions=precompute_captions,
        )
        val_dataset = ImageTextDataset(
            dataset=val_dataset,
            label_templates=label_templates,
            template_key=template_key,
            precompute_captions=precompute_captions,
        )
        train_dataset.name = dataset_name
        val_dataset.name = dataset_name

    from torch.utils.data import DataLoader

    train_dataset = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=ImageTextDataset.collate_fn,
    )
    val_dataset = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=ImageTextDataset.collate_fn,
    )

    return train_dataset, val_dataset

from src.trainers.alignment_trainer import AlignmentTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved alignment checkpoint (no training)."
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--image_layer_idx", type=int, required=True)
    parser.add_argument("--text_layer_idx", type=int, required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config_path)
    if not cfg_path.exists():
        raise ValueError(f"Unable to find config yaml file: {cfg_path}")
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    config = merge_dicts(config.get("defaults", {}), config.get("overrides", {}))

    data_path = Path(config["paths"]["data_path"])
    train_dataset, val_dataset = load_dataset(
        dataset_name=config["features"]["dataset"],
        data_path=data_path,
        batch_size=config["features"]["batch_size"],
        num_workers=config["features"]["num_workers"],
        label_templates=config["features"]["label_templates"],
        template_key=config["features"]["template_key"],
        precompute_captions=config["features"]["precompute_captions"],
    )

    eval_zero_shot_datasets = []
    eval_retrieval_datasets = []
    for d_name, l_data in [
        ("zero_shot_datasets", eval_zero_shot_datasets),
        ("retrieval_datasets", eval_retrieval_datasets),
    ]:
        for dataset_name in config["evaluation"][d_name]:
            try:
                _, ds_val = get_datasets(
                    dataset=dataset_name,
                    transform=get_default_transforms(),
                    root_dir=data_path,
                )
                l_data.append((dataset_name, ds_val))
                logger.info(
                    f"Successfully loaded '{dataset_name}', test size: {len(ds_val)}"
                )
            except Exception as e:
                logger.error(f"Error on {dataset_name}: {e}")

    trainer = AlignmentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_zero_shot_datasets=eval_zero_shot_datasets,
        eval_retrieval_datasets=eval_retrieval_datasets,
        wandb_logging=False,
        **config["alignment"],
    )

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    alignment_image = checkpoint.get("alignment_image")
    alignment_text = checkpoint.get("alignment_text")
    if alignment_image is None or alignment_text is None:
        raise ValueError("Checkpoint missing alignment_image/alignment_text modules.")

    layer_comb = (args.image_layer_idx, args.text_layer_idx)
    layer_str = f"img_{args.image_layer_idx}_txt_{args.text_layer_idx}"
    res_dict = {"layer_comb": layer_comb}
    with torch.no_grad():
        trainer.evaluate_retrieval(
            epoch=0,
            train_step=0,
            alignment_image=alignment_image,
            alignment_text=alignment_text,
            alignment_layer_combination=layer_comb,
            alignment_layer_combination_str=layer_str,
            additional_result_dict=res_dict,
        )
        trainer.evaluate_zero_shot_classification(
            epoch=0,
            train_step=0,
            alignment_image=alignment_image,
            alignment_text=alignment_text,
            alignment_layer_combination=layer_comb,
            alignment_layer_combination_str=layer_str,
            additional_result_dict=res_dict,
        )


if __name__ == "__main__":
    main()
