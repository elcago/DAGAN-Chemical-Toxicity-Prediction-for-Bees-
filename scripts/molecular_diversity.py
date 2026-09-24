from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_apistox
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="figures/tanimoto_histogram.png")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    df = load_apistox({**cfg["data"], "csv_path": ROOT / cfg["data"]["csv_path"]})

    fps = []
    for smi in df.smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
    sims = np.asarray(sims)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(sims, bins=40)
    plt.xlabel("Pairwise Tanimoto similarity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"pairs={len(sims):,}; mean={sims.mean():.4f}; median={np.median(sims):.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
