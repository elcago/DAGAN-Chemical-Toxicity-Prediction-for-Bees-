from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import add_descriptors, load_apistox, make_loaders
from src.features import edge_feature_dim, node_feature_dim
from src.losses import BinaryFocalLoss
from src.metrics import classification_metrics
from src.model import DAGAN
from src.utils import ensure_dir, get_device, load_config, save_json, set_seed


def evaluate(model, loader, device):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)[:, 1]
            ys.extend(batch.y.view(-1).cpu().numpy().tolist())
            probs.extend(prob.cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results/dagan")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    device = get_device()
    out_dir = ensure_dir(ROOT / args.output)

    df = load_apistox({**cfg["data"], "csv_path": ROOT / cfg["data"]["csv_path"]})
    df = add_descriptors(df, cfg["data"]["descriptors"])
    loaders, scaler, split_idx = make_loaders(df, cfg["data"], cfg["training"], cfg["seed"])

    model = DAGAN(
        node_dim=node_feature_dim(),
        edge_dim=edge_feature_dim(),
        descriptor_dim=len(cfg["data"]["descriptors"]),
        hidden_dim=cfg["model"]["hidden_dim"],
        heads=cfg["model"]["attention_heads"],
        dropout=cfg["model"]["dropout"],
        descriptor_hidden_dim=cfg["model"]["descriptor_hidden_dim"],
        fusion_dim=cfg["model"]["fusion_dim"],
    ).to(device)

    loss_fn = BinaryFocalLoss(
        cfg["training"]["alpha_toxic"], cfg["training"]["gamma"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    best_val_mcc = -1.0
    patience = 0
    history = []
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        epoch_losses = []
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        y_val, p_val = evaluate(model, loaders["val"], device)
        val_metrics = classification_metrics(y_val, p_val, cfg["training"]["threshold"])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} loss={record['train_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_mcc={val_metrics['mcc']:.4f}"
        )

        if val_metrics["mcc"] > best_val_mcc:
            best_val_mcc = val_metrics["mcc"]
            patience = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    y_test, p_test = evaluate(model, loaders["test"], device)
    test_metrics = classification_metrics(y_test, p_test, cfg["training"]["threshold"])

    save_json(test_metrics, out_dir / "test_metrics.json")
    save_json({"history": history}, out_dir / "history.json")
    joblib.dump(scaler, out_dir / "descriptor_scaler.joblib")
    np.savez(
        out_dir / "split_indices.npz",
        train=split_idx["train"], val=split_idx["val"], test=split_idx["test"]
    )
    np.savez(out_dir / "test_predictions.npz", y_true=y_test, y_prob=p_test)
    print("Test metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()
