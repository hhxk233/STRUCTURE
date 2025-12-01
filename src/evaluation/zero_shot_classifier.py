from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
from loguru import logger
from transformers import CLIPModel

from src.dataset_preparation.data_utils import get_meta_dict
from src.utils.utils import log_spherical_embedding_stats, safe_normalize


def chunked_logits(
    image_feats,
    zero_shot_classifier,
    chunk_size=1024,
    device: str = "cuda",
):
    # image_feats: (B, D) on GPU
    # classifier: (C, D) on CPU or GPU
    image_feats = image_feats.to(device)
    C, D = zero_shot_classifier.shape
    logits = []
    for start in range(0, C, chunk_size):
        end = min(start + chunk_size, C)
        w_chunk = zero_shot_classifier[start:end].to(device)
        l_chunk = image_feats @ w_chunk.T
        logits.append(l_chunk)
        # free GPU memory for the next chunk
        del w_chunk, l_chunk
        torch.cuda.empty_cache()
    return torch.cat(logits, dim=1)


def build_zero_shot_classifier(
    language_model,
    tokenizer,
    classnames: Sequence[str],
    templates: Sequence[Union[Callable, str]],
    dataset,
    layer_index: Optional[int] = None,
    alignment_layer: Optional[torch.nn.Module] = None,
    num_classes_per_batch: int = 10,
    device: Union[str, torch.device] = "cpu",
    pool_txt: str = "last",
    save_path: Optional[Path] = None,
    sample_by_sample_embedding: bool = False,
):
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0

    use_format = isinstance(templates[0], str)
    num_classes = len(classnames)
    num_templates = len(templates)

    texts = [
        template.format(c) if use_format else template(c)
        for c in classnames
        for template in templates
    ]
    tokens = tokenizer(
        texts,
        padding="longest",
        return_tensors="pt",
    )

    if save_path is not None and save_path.exists():
        cached = True
        llm_feats = torch.load(save_path, weights_only=False)["features"]
        logger.info(f"Loaded eval features from: {save_path}")
        log_spherical_embedding_stats(
            embeddings=(
                llm_feats[:, layer_index, :] if layer_index is not None else llm_feats
            ),
            log_prefix="Original Text",
        )
    else:
        cached = False
        llm_feats = []
    zeroshot_weights = []
    for i in range(0, len(texts), num_classes_per_batch * num_templates):
        if cached:
            class_embeddings = llm_feats[i : i + num_classes_per_batch * num_templates]
            class_embeddings = class_embeddings.to(device)
        else:
            token_inputs = {
                k: v[i : i + num_classes_per_batch * num_templates].to(device).long()
                for (k, v) in tokens.items()
            }
            with torch.no_grad():
                if sample_by_sample_embedding:
                    class_embeddings = []
                    for j in range(token_inputs["input_ids"].shape[0]):
                        single_class_embedding = language_model(
                            input_ids=token_inputs["input_ids"][j].unsqueeze(0),
                            attention_mask=token_inputs["attention_mask"][j].unsqueeze(
                                0
                            ),
                        )
                        class_embeddings.append(single_class_embedding)
                    # this is now of dimension (BS, Layers, Tokens, Dim)
                    class_embeddings = torch.stack(
                        [torch.stack(x["hidden_states"]) for x in class_embeddings],
                        dim=0,
                    )
                    class_embeddings = class_embeddings.squeeze(2)
                else:
                    if isinstance(language_model, CLIPModel):
                        class_embeddings = language_model.get_text_features(
                            **token_inputs
                        )
                    else:
                        class_embeddings = language_model(
                            input_ids=token_inputs["input_ids"],
                            attention_mask=token_inputs["attention_mask"],
                        )
                        # swap the backsize to the first dimension
                        # (BS, Layers, Tokens, Dim)
                        class_embeddings = torch.stack(
                            class_embeddings["hidden_states"]
                        ).permute(1, 0, 2, 3)

                if layer_index is not None:
                    if pool_txt == "avg":
                        # make the mask compatible with the dimension
                        mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                        # average along the token dimension
                        class_embeddings = (class_embeddings * mask).sum(2) / mask.sum(
                            2
                        )
                    elif pool_txt == "last":
                        class_embeddings = class_embeddings[:, -1, :, :]
                    elif pool_txt == "none":
                        # select only the layer we care about, otherwise we don't have enough memory
                        class_embeddings = class_embeddings[:, layer_index, :, :]
                    else:
                        raise NotImplementedError(f"unknown pooling {pool_txt}")
                llm_feats.append(class_embeddings.cpu())

        if pool_txt == "none":
            class_embeddings = class_embeddings.float()
        else:
            if layer_index is not None:
                # class_embeddings = (num_classes_per_batch * num_templates, Layers, Dim)
                # select the layer we have aligned
                class_embeddings = class_embeddings[:, layer_index, :].float()
            else:
                class_embeddings = class_embeddings.float()
        # pass it through the alignment layer (for all classes * num_templates)
        if alignment_layer is not None:
            class_embeddings = alignment_layer(class_embeddings)
        # reshape the embeddings so we have the template dimension
        class_embeddings = class_embeddings.reshape(
            class_embeddings.shape[0] // num_templates,
            num_templates,
            -1,
        )
        # norm in order to ensure that each template contributes the same
        # average and norm again
        class_embeddings = safe_normalize(class_embeddings, p=2, dim=-1)
        class_embeddings = class_embeddings.mean(dim=1)
        class_embeddings = safe_normalize(class_embeddings, p=2, dim=-1)
        zeroshot_weights.append(class_embeddings.cpu())
    zeroshot_weights = torch.concat(zeroshot_weights, dim=0).cpu()
    # make sure that we have the correct dimensions
    assert num_classes == zeroshot_weights.shape[0]
    log_spherical_embedding_stats(
        embeddings=zeroshot_weights,
        log_prefix="Zero-Shot Anchors Aligned",
    )

    # save the eval features from the llm if they don't exist yet
    if not cached and save_path is not None and not save_path.exists():
        llm_feats = torch.cat(llm_feats).cpu()
        log_spherical_embedding_stats(
            embeddings=(
                llm_feats[:, layer_index, :] if layer_index is not None else llm_feats
            ),
            log_prefix="Original Text",
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": llm_feats} | get_meta_dict(dataset), save_path)
        logger.info(f"Saved eval features to: {save_path}")
    return zeroshot_weights
