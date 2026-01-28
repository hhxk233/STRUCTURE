## MLP Embedding Visualization Script

This document explains how `scripts/plot_structure_mlp_embeddings.py` builds the
image-text alignment plot and which steps are involved from data selection to
rendering.

### Overview

The script produces a 2D visualization that aligns paired image and caption
embeddings using an MLP alignment layer. It also draws thumbnails, text icons,
links, and a caption legend for quick inspection.

### High-Level Flow

1. **Select Samples**
   - If COCO annotations are available, the script finds a 4-category cycle of
     overlapping image labels and samples `num_samples` images evenly across the
     pairs.
   - Otherwise it loads a subset from `subset_dir` and selects images per class.

2. **Prepare Captions**
   - Captions are concatenated per image, trimmed to the first sentence, and
     truncated to a maximum word count.
   - The displayed caption is wrapped into two lines and always ends with `...`.

3. **Extract Features**
   - The script uses `AlignmentTrainer` to extract multi-layer features from
     the image encoder (CLS token) and the text encoder (average pooled tokens).
   - Features are cached if available.

4. **Train Alignment Layer**
   - A lightweight MLP alignment layer (`ResLowRankHead`) is trained with a
     CLIP-style contrastive loss and STRUCTURE regularization.
   - The checkpoint stores the alignment modules and configuration.

5. **Project to 2D**
   - The script loads the latest checkpoint, parses the trained layer
     combination, and projects the corresponding layer features.
   - If training used normalized latents, the same normalization is applied
     before plotting.

6. **Render the Plot**
   - Points are normalized into a unit square and optionally split into
     separate y-bands for image/text modalities.
   - Images, emoji/text icons, links, and distances are drawn on top.
   - A caption legend is placed to the right using matching icons.

### Key Outputs

- `mlp_image_text_aligned.png`: main visualization
- `mlp_selected_samples.csv`: sampled image/caption metadata
- `mlp_image_embeds.npy`, `mlp_text_embeds.npy`: 2D embeddings used for plotting
- `selected_images/`: image thumbnails saved for reference

### Example Command

```
source /home/yuheng/ICML/.venv_mlp_rs/bin/activate
PYTHONPATH=/home/yuheng/task/STRUCTURE python scripts/plot_structure_mlp_embeddings.py \
  --out_dir /home/yuheng/task/STRUCTURE/results/mlp_vis_6k \
  --num_labels 4 \
  --num_samples 10 \
  --dim_alignment 2 \
  --tries 1 \
  --comic_font_path /home/yuheng/task/STRUCTURE/results/mlp_vis_6k/Comic-Sans-MS-Regular-2.ttf
```

### Common Styling Controls

- `--caption_font_size`: caption legend text size
- `--legend_icon_zoom`: size of legend icons
- `--legend_title_size`: legend title font size
- `--image_alpha`: thumbnail transparency
- `--emoji_px`, `--chat_marker_size`: icon sizes
- `--draw_links`, `--show_distances`: link rendering controls
- `--split_modalities`: split image/text into separate y-bands

