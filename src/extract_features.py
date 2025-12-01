import argparse
import gc
import os

import timm
import torch
from datasets import load_dataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import trange

import src.utils.alignment_utils as utils
from src.models.tasks import get_models
from src.models.text.models import load_llm, load_tokenizer


def extract_llm_features(filenames, dataset, args):
    """Extract features from language models.

    Args:
        filenames: list of language model names by huggingface identifiers
        dataset: huggingface dataset
        args: argparse arguments
    """

    texts = [str(x["text"][args.caption_idx]) for x in dataset]

    for llm_model_name in filenames[::-1]:
        save_path = utils.to_feature_filename(
            args.output_dir,
            args.dataset,
            args.subset,
            llm_model_name,
            pool=args.pool,
            prompt=args.prompt,
            caption_idx=args.caption_idx,
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{llm_model_name}")
        print(f"save_path: \t{save_path}")

        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue

        language_model = load_llm(
            llm_model_name,
            qlora=args.qlora,
            force_download=args.force_download,
        )
        llm_param_count = sum([p.numel() for p in language_model.parameters()])
        tokenizer = load_tokenizer(llm_model_name)

        tokens = tokenizer(texts, padding="longest", return_tensors="pt")
        llm_feats, losses, bpb_losses = [], [], []

        # hack to get around HF mapping data incorrectly when using model-parallel
        device = next(language_model.parameters()).device

        for i in trange(0, len(dataset), args.batch_size):
            # get embedding cuda device
            token_inputs = {
                k: v[i : i + args.batch_size].to(device).long()
                for (k, v) in tokens.items()
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
                # llm_output shape: (BS, Tokens, Dim)
                ret = utils.cross_entropy_loss(token_inputs, llm_output)
                if ret is not None:
                    loss, avg_loss = ret
                    losses.extend(avg_loss.cpu())

                    bpb = utils.cross_entropy_to_bits_per_unit(
                        loss.cpu(), texts[i : i + args.batch_size], unit="byte"
                    )
                    bpb_losses.extend(bpb)

                # make sure to do all the processing in cpu to avoid memory problems
                if args.pool == "avg":
                    # swap the backsize to the first dimension
                    # (BS, Layers, Tokens, Dim)
                    feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                    # make the mask compatible with the dimension
                    mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                    # average along the token dimension
                    feats = (feats * mask).sum(2) / mask.sum(2)
                elif args.pool == "last":
                    feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                    feats = torch.stack(feats).permute(1, 0, 2)
                else:
                    raise NotImplementedError(f"unknown pooling {args.pool}")
                llm_feats.append(feats.cpu())

        if len(losses) > 0:
            print(f"average loss:\t{torch.stack(losses).mean().item()}")
        save_dict = {
            "feats": torch.cat(llm_feats).cpu(),
            "num_params": llm_param_count,
            "mask": tokens["attention_mask"].cpu(),
        }
        if len(bpb_losses) > 0:
            save_dict["bpb"] = torch.stack(bpb_losses).mean()
        if len(losses) > 0:
            save_dict["loss"] = torch.stack(losses).mean()
        torch.save(save_dict, save_path)

        del language_model, tokenizer, llm_feats, llm_output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    return


def extract_lvm_features(filenames, dataset, args):
    """
    Extract features from vision models.

    Args:
        filenames: list of vision model names by timm identifiers
        image_file_paths: list of image file paths
        args: argparse arguments
    """
    assert args.pool == "cls", "pooling is not supported for lvm features"

    for lvm_model_name in filenames:
        save_path = utils.to_feature_filename(
            args.output_dir,
            args.dataset,
            args.subset,
            lvm_model_name,
            pool=args.pool,
            prompt=None,
            caption_idx=None,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{lvm_model_name}")
        print(f"save_path: \t{save_path}")

        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue

        vision_model = timm.create_model(lvm_model_name, pretrained=True).cuda().eval()
        lvm_param_count = sum([p.numel() for p in vision_model.parameters()])

        transform = create_transform(
            **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
        )

        if "vit" in lvm_model_name:
            return_nodes = [
                f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))
            ]
        elif "conv" in lvm_model_name:
            # select feature extraction points (e.g., after each stage)
            return_nodes = {
                "stages.0.blocks.1": "stage1",
                "stages.1.blocks.1": "stage2",
                "stages.2.blocks.1": "stage3",
                "stages.3.blocks.1": "stage4",
            }
        else:
            raise NotImplementedError(f"unknown model {lvm_model_name}")

        vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
        lvm_feats = []

        for i in trange(0, len(dataset), args.batch_size):
            with torch.no_grad():
                ims = torch.stack(
                    [
                        transform(dataset[j]["image"])
                        for j in range(i, i + args.batch_size)
                    ]
                ).cuda()
                lvm_output = vision_model(ims)

                if args.pool == "cls":
                    if "vit" in lvm_model_name:
                        feats = [v[:, 0, :] for v in lvm_output.values()]
                        feats = torch.stack(feats).permute(1, 0, 2).cpu()
                    elif "conv" in lvm_model_name:
                        feats = [
                            torch.nn.functional.adaptive_avg_pool2d(f, 1)
                            .squeeze(-1)
                            .squeeze(-1)
                            .cpu()
                            for f in lvm_output.values()
                        ]
                    else:
                        raise NotImplementedError(f"unknown model {lvm_model_name}")
                lvm_feats.append(feats)

        if "conv" in lvm_model_name:
            # Transpose list of lists to group features by layer across batches
            # Before zip: lvm_feats[batch][layer] -> After zip: lvm_feats[layer][batch]
            lvm_feats = list(zip(*lvm_feats))
            # Concatenate batch features for each layer
            # Each tensor now has shape (total_images, C) for a given layer
            lvm_feats = [torch.cat(layer_feats, dim=0) for layer_feats in lvm_feats]
        else:
            lvm_feats = torch.cat(lvm_feats)

        torch.save({"feats": lvm_feats, "num_params": lvm_param_count}, save_path)

        del vision_model, transform, lvm_feats, lvm_output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pool", type=str, default="avg", choices=["avg", "cls"])
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--dataset", type=str, default="minhuh/prh")
    parser.add_argument("--subset", type=str, default="wit_1024")
    parser.add_argument("--caption_idx", type=int, default=0)
    parser.add_argument("--modelset", type=str, default="val", choices=["val", "test"])
    parser.add_argument(
        "--modality", type=str, default="all", choices=["vision", "language", "all"]
    )
    parser.add_argument("--output_dir", type=str, default="./results/features")
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()

    if args.qlora:
        print("QLoRA is set to True. The alignment score will be slightly off.")

    llm_models, lvm_models = get_models(args.modelset, modality=args.modality)

    # load dataset once outside
    dataset = load_dataset(
        args.dataset,
        revision=args.subset,
        split="train",
        cache_dir="HuggingFaceCache/",
    )

    if args.modality in ["all", "language"]:
        # extract all language model features
        extract_llm_features(llm_models, dataset, args)

    if args.modality in ["all", "vision"]:
        # extract all vision model features
        extract_lvm_features(lvm_models, dataset, args)
