import argparse
from pathlib import Path
from typing import List

import torchvision.transforms as transforms
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from src.core.src.datasets.image_text_dataset import ImageTextDataset
from src.core.src.utils.loader import Loader, merge_dicts
from src.dataset_preparation.data_utils import get_datasets, get_default_transforms
from src.trainers.alignment_trainer import AlignmentTrainer
from src.trainers.clip_eval_trainer import CLIPEvalTrainer
from src.trainers.csa_trainer import CSATrainer


def load_dataset(
    dataset_name: str,
    data_path: Path,
    batch_size: int = 16,
    num_workers: int = 1,
    label_templates: List[str] = ["a photo of a {label}"],
    template_key: str = "label",
    precompute_captions: bool = True,
):
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

    # since we're purely using the train dataset
    # for obtaining the embeddings we don't need to shuffle
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


parser = argparse.ArgumentParser(
    description="Experiments for the Representation Alignment.",
)
parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the config yaml.",
)
parser.add_argument(
    "--wandb_notes",
    type=str,
    help="Notes for the wandb run.",
)
args = parser.parse_args()

if __name__ == "__main__":
    args.config_path = Path(args.config_path)
    if not args.config_path.exists():
        raise ValueError(f"Unable to find config yaml file: {args.config_path}")
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    # merge defaults with overrides (overrides take precedence)
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

    # additional unimodal data
    additional_unimodal_data = None
    text_unimodal_data = []
    image_unimodal_data = []
    if config["training"]["unimodal_data"]["use"]:
        for modality in ["text", "image"]:
            if config["training"]["unimodal_data"][modality] is None:
                continue
            for dataset_name in config["training"]["unimodal_data"][modality]:
                orig_dataset_name = dataset_name
                use_val_set = False
                range_from = None
                range_to = None
                if "_val" in dataset_name:
                    dataset_name = dataset_name.replace("_val", "")
                    use_val_set = True
                if "_" in dataset_name:
                    range_from = int(dataset_name.split("_")[1])
                    range_to = int(dataset_name.split("_")[2])
                    dataset_name = dataset_name.split("_")[0]
                try:
                    ds_train, ds_val = get_datasets(
                        dataset=dataset_name,
                        transform=get_default_transforms(),
                        root_dir=data_path,
                    )
                    if use_val_set:
                        ds_train = ds_val
                    if range_from is not None and range_to is not None:
                        ds_train.df = ds_train.df.iloc[range_from:range_to]
                        ds_train.df.reset_index(drop=True, inplace=True)
                    if config["training"]["unimodal_data"].get("samples", None):
                        ds_train.df = ds_train.df.sample(
                            n=config["training"]["unimodal_data"]["samples"]
                        )
                        ds_train.df.reset_index(drop=True, inplace=True)
                    train_loader = DataLoader(
                        ds_train,
                        batch_size=config["features"]["batch_size"],
                        drop_last=False,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=config["features"]["num_workers"],
                    )
                    if modality == "text":
                        text_unimodal_data.append((orig_dataset_name, train_loader))
                    else:
                        image_unimodal_data.append((orig_dataset_name, train_loader))
                    logger.info(
                        f"Successfully loaded unimodal data '{orig_dataset_name}', train size: {len(ds_train)}"
                    )
                except Exception as e:
                    logger.error(f"Error on {dataset_name}: {e}")
        additional_unimodal_data = {}
        additional_unimodal_data["text"] = text_unimodal_data
        additional_unimodal_data["image"] = image_unimodal_data

    # our evaluation datasets
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

    trainer_kwargs = {
        "config": config,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "eval_zero_shot_datasets": eval_zero_shot_datasets,
        "eval_retrieval_datasets": eval_retrieval_datasets,
        "wandb_notes": args.wandb_notes,
    }
    trainer_kwargs = trainer_kwargs | config["alignment"]
    if "cca" in config["training"].keys() and config["training"]["cca"]:
        trainer = CSATrainer(**trainer_kwargs)
    elif "clip" in config["training"].keys() and config["training"]["clip"]:
        trainer = CLIPEvalTrainer(**trainer_kwargs)
    else:
        trainer = AlignmentTrainer(**trainer_kwargs)
    trainer.fit(additional_unimodal_data=additional_unimodal_data)
    del trainer
