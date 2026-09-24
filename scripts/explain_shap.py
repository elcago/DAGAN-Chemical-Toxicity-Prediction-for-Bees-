from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from torch_geometric.data import Batch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data import add_descriptors, build_graph_dataset, load_apistox
from src.features import edge_feature_dim, node_feature_dim
from src.model import DAGAN
from src.utils import get_device, load_config, set_seed


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--model-dir',default='results/dagan'); p.add_argument('--max-samples',type=int,default=50); p.add_argument('--out',default='figures/shap_summary.png'); a=p.parse_args()
    cfg=load_config(ROOT/a.config); set_seed(cfg['seed']); device=get_device(); model_dir=ROOT/a.model_dir
    df=load_apistox({**cfg['data'],'csv_path':ROOT/cfg['data']['csv_path']}); df=add_descriptors(df,cfg['data']['descriptors'])
    scaler=joblib.load(model_dir/'descriptor_scaler.joblib'); splits=np.load(model_dir/'split_indices.npz'); test_idx=splits['test'][:a.max_samples]
    graphs=build_graph_dataset(df,test_idx,cfg['data']['descriptors'],scaler)
    model=DAGAN(node_feature_dim(),edge_feature_dim(),len(cfg['data']['descriptors']),cfg['model']['hidden_dim'],cfg['model']['attention_heads'],cfg['model']['dropout'],cfg['model']['descriptor_hidden_dim'],cfg['model']['fusion_dim']).to(device)
    model.load_state_dict(torch.load(model_dir/'best_model.pt',map_location=device)); model.eval()

    X=np.vstack([g.descriptors.numpy().reshape(-1) for g in graphs])
    background=X[:min(cfg['explainability']['shap_background_size'],len(X))]
    shap_rows=[]
    for g,x0 in zip(graphs,X):
        def predict_desc(z):
            batch=[]
            for row in np.asarray(z):
                gi=deepcopy(g); gi.descriptors=torch.tensor(row,dtype=torch.float32).view(1,-1); batch.append(gi)
            b=Batch.from_data_list(batch).to(device)
            with torch.no_grad(): return torch.softmax(model(b),1)[:,1].cpu().numpy()
        explainer=shap.KernelExplainer(predict_desc,background)
        vals=explainer.shap_values(x0.reshape(1,-1),nsamples=min(200,2*X.shape[1]+64))
        vals=np.asarray(vals)
        if vals.ndim==3: vals=vals[...,0]
        shap_rows.append(vals.reshape(-1))
    S=np.vstack(shap_rows)
    out=ROOT/a.out; out.parent.mkdir(parents=True,exist_ok=True)
    shap.summary_plot(S,X,feature_names=cfg['data']['descriptors'],show=False)
    plt.tight_layout(); plt.savefig(out,dpi=300,bbox_inches='tight'); print(f'Saved {out}')

if __name__=='__main__': main()
