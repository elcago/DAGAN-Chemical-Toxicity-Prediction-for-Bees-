from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import add_descriptors, load_apistox, split_indices
from src.metrics import classification_metrics
from src.utils import load_config, save_json, set_seed


def fp_array(fp):
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return fp_array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))


def maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return fp_array(MACCSkeys.GenMACCSKeys(mol))


def graph_fp(smiles):
    # RDKit fingerprint baseline.
    mol = Chem.MolFromSmiles(smiles)
    return fp_array(Chem.RDKFingerprint(mol, fpSize=2048))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/baselines")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])

    data_cfg = {**cfg["data"], "csv_path": ROOT / cfg["data"]["csv_path"]}
    df = load_apistox(data_cfg)
    train_idx, val_idx, test_idx = split_indices(df, cfg["data"], cfg["seed"])
    trainval = np.concatenate([train_idx, val_idx])
    y = df.target.values

    reps = {
        "ECFP4": np.vstack([ecfp4(s) for s in df.smiles]),
        "MACCSFP": np.vstack([maccs(s) for s in df.smiles]),
        "GraphFP": np.vstack([graph_fp(s) for s in df.smiles]),
    }
    models = {
        "ECFP4_SVM": ("ECFP4", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=cfg["seed"])),
        "ECFP4_kNN": ("ECFP4", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "MACCSFP_SVM": ("MACCSFP", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=cfg["seed"])),
        "MACCSFP_ANN": ("MACCSFP", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=cfg["seed"])),
        "GraphFP_SVM": ("GraphFP", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=cfg["seed"])),
    }

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, (rep_name, model) in models.items():
        X = reps[rep_name]
        model.fit(X[trainval], y[trainval])
        prob = model.predict_proba(X[test_idx])[:, 1]
        results[name] = classification_metrics(y[test_idx], prob, cfg["training"]["threshold"])
        joblib.dump(model, out_dir / f"{name}.joblib")
        print(name, results[name])

    save_json(results, out_dir / "metrics.json")
    pd.DataFrame({k: {m: v for m, v in d.items() if not isinstance(v, list)} for k, d in results.items()}).T.to_csv(out_dir / "metrics.csv")


if __name__ == "__main__":
    main()
