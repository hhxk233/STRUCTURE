# With Limited Data for Multimodal Alignment, Let the STRUCTURE Guide You

<div align="center">

### [NeurIPS 2025]

[![Paper](https://img.shields.io/badge/arXiv-2506.16895-b31b1b.svg)](https://arxiv.org/abs/2506.16895)
[![OpenReview](https://img.shields.io/badge/OpenReview-IkvQqD7hk3-8c1b13.svg)](https://openreview.net/forum?id=IkvQqD7hk3)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://brbiclab.epfl.ch/projects/structure/)

**[Fabian Gröger](https://fabiangroeger.com/)** · **[Shuo Wen](https://wenshuo128.github.io/)** · **[Huyen Le](https://people.epfl.ch/thao.le)** · **[Maria Brbić](https://brbiclab.epfl.ch/team/)**

</div>

---

## Overview

![STRUCTURE Teaser](https://brbiclab.epfl.ch/wp-content/uploads/2025/06/STRUCTURE_1.png)

Multimodal models have demonstrated powerful capabilities in complex tasks requiring multimodal alignment, including zero-shot classification and cross-modal retrieval.
However, existing models typically rely on millions of paired multimodal samples, which are prohibitively expensive or infeasible to obtain in many domains.

In this work, we explore the feasibility of building multimodal models with **limited amounts of paired data** by aligning pretrained unimodal foundation models.
We show that high-quality alignment is possible with as few as **tens of thousands of paired samples** — less than 1% of the data typically used in the field.

## Key Contributions

- **STRUCTURE Regularization**: An effective technique that preserves the neighborhood geometry of the latent space of unimodal encoders
- **Layer Selection**: Demonstration that aligning last layers is often suboptimal, with benefits from aligning layers with highest representational similarity
- **Strong Results**: 51.6% average relative improvement in classification and 91.8% in retrieval tasks across 24 benchmarks
- **Broad Applicability**: Can be readily incorporated into existing alignment methods

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.1.2+
- CUDA 11.8+ (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mlbio-epfl/STRUCTURE.git
cd STRUCTURE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Prepare your datasets using the provided scripts. For example, for COCO:
```bash
# COCO dataset will be downloaded automatically
# Place in ./data/coco/
```

For other datasets, see the `src/dataset_preparation/` directory for preparation scripts.

### 2. Training

Train an alignment model with limited data:
```bash
python src/train_subsampled_alignment.py --config_path configs/losses_lin/clip_base_best.yaml
```

Train with full alignment:
```bash
python src/train_alignment.py --config_path configs/losses_lin/clip_base_best.yaml
```

### 3. Evaluation

Extract features:
```bash
python src/extract_features.py --config_path configs/default.yaml
```

Measure alignment quality:
```bash
python src/measure_alignment.py --config_path configs/metrics/clip_mutual_knn_rice.yaml
```

## Configuration

The repository uses YAML configuration files located in `configs/`:

- `configs/default.yaml` - Base configuration
- `configs/losses_lin/` - Linear alignment layer configurations
- `configs/losses_mlp/` - MLP alignment layer configurations
- `configs/ablations/` - Ablation study configurations
- `configs/csa/` - CSA configurations
- `configs/metrics/` - Alignment metrics configurations

## Project Structure

```
representation-alignment/
├── configs/              # Configuration files
├── src/
│   ├── alignment/       # Alignment layer implementations
│   ├── trainers/        # Training logic
│   ├── models/          # Model implementations
│   ├── loss/            # Loss functions
│   ├── evaluation/      # Evaluation metrics
│   ├── dataset_preparation/  # Dataset preparation scripts
│   └── utils/           # Utility functions
├── data/                # Dataset directory (created during setup)
└── requirements.txt     # Python dependencies
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{groger2025structure,
  title={With Limited Data for Multimodal Alignment, Let the {STRUCTURE} Guide You},
  author={Gr{\"o}ger, Fabian and Wen, Shuo and Le, Huyen and Brbic, Maria},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=IkvQqD7hk3}
}
```
