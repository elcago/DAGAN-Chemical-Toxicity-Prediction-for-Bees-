from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .features import compute_descriptors, molecule_to_graph


def _resolve_column(df: pd.DataFrame, requested: str, candidates: Sequence[str]) -> str:
    if requested in df.columns:
        return requested
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(f"Could not find column '{requested}'. Available columns: {list(df.columns)}")


def normalize_binary_label(value, positive_labels) -> int:
    if pd.isna(value):
        raise ValueError("Missing label")
    if value in positive_labels or str(value) in {str(x) for x in positive_labels}:
        return 1
    text = str(value).strip().lower()
    if text in {"toxic", "highly toxic", "moderately toxic", "positive", "yes"}:
        return 1
    if text in {"non-toxic", "nontoxic", "non toxic", "negative", "no", "0", "false"}:
        return 0
    try:
        return int(float(value) > 0)
    except Exception as exc:
        raise ValueError(f"Unrecognized binary label: {value}") from exc


def load_apistox(cfg: Dict) -> pd.DataFrame:
    path = Path(cfg["csv_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download ApisTox dataset_final.csv and place it at this path."
        )
    df = pd.read_csv(path)
    smiles_col = _resolve_column(df, cfg.get("smiles_column", "smiles"), ["smiles", "SMILES"])
    label_col = _resolve_column(df, cfg.get("label_column", "label"), ["label", "binary_label", "toxicity"])
    out = df.copy()
    out["smiles"] = out[smiles_col].astype(str)
    out["target"] = [normalize_binary_label(v, cfg.get("positive_labels", [1, "toxic"])) for v in out[label_col]]
    out = out.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    return out


def add_descriptors(df: pd.DataFrame, descriptor_names: List[str]) -> pd.DataFrame:
    records = []
    keep = []
    for idx, smi in enumerate(df["smiles"]):
        try:
            records.append(compute_descriptors(smi, descriptor_names))
            keep.append(idx)
        except Exception:
            continue
    out = df.iloc[keep].copy().reset_index(drop=True)
    desc_df = pd.DataFrame(records)
    return pd.concat([out, desc_df], axis=1)


def split_indices(df: pd.DataFrame, cfg: Dict, seed: int):
    idx = np.arange(len(df))
    strat = df["target"] if cfg.get("stratify", True) else None
    train_frac = cfg.get("train_fraction", 0.6)
    val_frac = cfg.get("val_fraction", 0.2)
    test_frac = cfg.get("test_fraction", 0.2)
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1")

    train_idx, temp_idx = train_test_split(
        idx, test_size=1.0 - train_frac, random_state=seed, stratify=strat
    )
    temp_y = df.iloc[temp_idx]["target"] if strat is not None else None
    test_share_of_temp = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_share_of_temp, random_state=seed, stratify=temp_y
    )
    return train_idx, val_idx, test_idx


def fit_scaler(df: pd.DataFrame, train_idx, descriptor_names: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df.iloc[train_idx][descriptor_names].values)
    return scaler


def build_graph_dataset(
    df: pd.DataFrame,
    indices,
    descriptor_names: List[str],
    scaler: StandardScaler,
) -> List[Data]:
    data_list = []
    for i in indices:
        row = df.iloc[int(i)]
        try:
            x, edge_index, edge_attr = molecule_to_graph(row["smiles"])
        except ValueError:
            continue
        desc = scaler.transform([row[descriptor_names].values.astype(float)])[0]
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            descriptors=torch.tensor(desc, dtype=torch.float32).view(1, -1),
            y=torch.tensor([int(row["target"])], dtype=torch.long),
            sample_index=torch.tensor([int(i)], dtype=torch.long),
        )
        data.smiles = row["smiles"]
        data_list.append(data)
    return data_list


def make_loaders(df: pd.DataFrame, data_cfg: Dict, training_cfg: Dict, seed: int):
    descriptors = data_cfg["descriptors"]
    train_idx, val_idx, test_idx = split_indices(df, data_cfg, seed)
    scaler = fit_scaler(df, train_idx, descriptors)
    train_ds = build_graph_dataset(df, train_idx, descriptors, scaler)
    val_ds = build_graph_dataset(df, val_idx, descriptors, scaler)
    test_ds = build_graph_dataset(df, test_idx, descriptors, scaler)
    batch_size = training_cfg.get("batch_size", 64)
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }
    return loaders, scaler, {"train": train_idx, "val": val_idx, "test": test_idx}
