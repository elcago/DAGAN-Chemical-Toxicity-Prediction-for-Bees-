# DAGAN: Predicting Pesticide Toxicity to Honey Bees

DAGAN (Descriptor-Augmented Graph Attention Network) combines molecular graphs with physicochemical descriptors through bidirectional cross-attention and gated fusion to predict acute pesticide toxicity to honey bees.

## Repository structure

```text
DAGAN_HoneyBee_Toxicity/
├── configs/default.yaml
├── data/README.md
├── scripts/
│   ├── descriptor_selection.py
│   ├── molecular_diversity.py
│   ├── train_dagan.py
│   ├── train_baselines.py
│   ├── train_graphsage.py
│   ├── train_chemberta.py
│   ├── explain_shap.py
│   └── explain_gnn.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── requirements.txt
└── LICENSE
```

## Data

This project uses **ApisTox**, published by Adamczyk, Poziemski, and Siedlecki (2025). The official dataset repository provides the final CSV as `outputs/dataset_final.csv`.

1. Download `dataset_final.csv` from the official ApisTox repository.
2. Copy it to:

```text
data/dataset_final.csv
```

Official ApisTox repository: https://github.com/j-adamczyk/ApisTox_dataset

Zenodo record: https://zenodo.org/records/13350981

Dataset paper: https://doi.org/10.1038/s41597-024-04232-w

The dataset is not redistributed here. Follow the ApisTox license and citation requirements.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for your CUDA version before installing the remaining requirements.

## Run the analyses

### 1. Descriptor selection

Computes Shapiro-Wilk tests, Mann-Whitney U tests with Bonferroni correction, and Pearson correlations.

```bash
python scripts/descriptor_selection.py
```

### 2. Molecular diversity

Computes radius-2, 2048-bit Morgan fingerprints and all pairwise Tanimoto similarities.

```bash
python scripts/molecular_diversity.py
```

### 3. Train DAGAN

```bash
python scripts/train_dagan.py
```

Outputs are written to `results/dagan/`, including the best checkpoint, test metrics, predictions, split indices, and descriptor scaler.

The default configuration follows the manuscript where specified:

- Hidden dimension: 64
- GAT attention heads: 8
- Learning rate: 1e-4
- Dropout: 0.2
- Batch size: 64
- Maximum epochs: 300
- Early-stopping patience: 20
- Focal-loss toxic-class weight: 0.71
- Focal-loss gamma: 2

### 4. Baselines

Traditional fingerprint baselines:

```bash
python scripts/train_baselines.py
```

GraphSAGE:

```bash
python scripts/train_graphsage.py
```

ChemBERTa (optional):

```bash
python scripts/train_chemberta.py
```

### 5. Explainability

Descriptor-level SHAP:

```bash
python scripts/explain_shap.py --max-samples 50
```

Graph-level GNNExplainer:

```bash
python scripts/explain_gnn.py --max-samples 100
```

`explain_shap.py` keeps each graph fixed and perturbs the descriptor vector. Increase `--max-samples` to explain more test compounds.

## Molecular representation

Each molecule is parsed from SMILES with RDKit. The graph stream uses nine OGB-style atom properties: atomic number, chirality, degree, formal charge, hybridization, aromaticity, hydrogen count, radical-electron count, and ring membership. The code reserves an additional channel so the concatenated node representation is fixed at **174 dimensions**, matching the manuscript specification.

The default descriptor vector contains the six descriptors retained by the manuscript’s selection procedure:

- MolWt
- TPSA
- MolLogP
- FractionCSP3
- NumRotatableBonds
- NumAromaticRings


