from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import add_descriptors, load_apistox
from src.features import DESCRIPTOR_FUNCTIONS
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/descriptor_selection.csv")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    all_names = list(DESCRIPTOR_FUNCTIONS.keys())
    df = load_apistox({**cfg["data"], "csv_path": ROOT / cfg["data"]["csv_path"]})
    df = add_descriptors(df, all_names)

    rows = []
    for name in all_names:
        tox = df.loc[df.target == 1, name].values
        non = df.loc[df.target == 0, name].values
        raw_p = mannwhitneyu(tox, non, alternative="two-sided").pvalue
        rows.append({
            "descriptor": name,
            "shapiro_toxic_p": shapiro(tox).pvalue if len(tox) <= 5000 else np.nan,
            "shapiro_nontoxic_p": shapiro(non).pvalue if len(non) <= 5000 else np.nan,
            "mann_whitney_raw_p": raw_p,
            "bonferroni_p": min(raw_p * len(all_names), 1.0),
        })
    out = pd.DataFrame(rows).sort_values("bonferroni_p")
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    corr = df[all_names].corr(method="pearson")
    corr.to_csv(out_path.with_name("descriptor_correlations.csv"))
    print(out.to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
