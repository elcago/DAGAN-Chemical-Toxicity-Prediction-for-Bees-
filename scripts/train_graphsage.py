from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import add_descriptors, load_apistox, make_loaders
from src.features import node_feature_dim
from src.metrics import classification_metrics
from src.utils import get_device, load_config, save_json, set_seed


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.fc = torch.nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, data):
        x = F.relu(self.c1(data.x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.c2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.fc(x)


def eval_model(model, loader, device):
    model.eval(); ys=[]; probs=[]
    with torch.no_grad():
        for b in loader:
            b=b.to(device); p=torch.softmax(model(b),1)[:,1]
            ys.extend(b.y.view(-1).cpu().tolist()); probs.extend(p.cpu().tolist())
    return classification_metrics(ys, probs)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/default.yaml'); parser.add_argument('--out',default='results/graphsage'); args=parser.parse_args()
    cfg=load_config(ROOT/args.config); set_seed(cfg['seed']); device=get_device()
    df=load_apistox({**cfg['data'],'csv_path':ROOT/cfg['data']['csv_path']}); df=add_descriptors(df,cfg['data']['descriptors'])
    loaders,_,_=make_loaders(df,cfg['data'],cfg['training'],cfg['seed'])
    model=GraphSAGE(node_feature_dim(),cfg['model']['hidden_dim'],cfg['model']['dropout']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate']); best=-1; patience=0
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    for epoch in range(cfg['training']['epochs']):
        model.train()
        for b in loaders['train']:
            b=b.to(device); opt.zero_grad(); loss=F.cross_entropy(model(b),b.y.view(-1)); loss.backward(); opt.step()
        vm=eval_model(model,loaders['val'],device)
        if vm['mcc']>best: best=vm['mcc']; patience=0; torch.save(model.state_dict(),out/'best_model.pt')
        else: patience+=1
        if patience>=cfg['training']['early_stopping_patience']: break
    model.load_state_dict(torch.load(out/'best_model.pt',map_location=device)); tm=eval_model(model,loaders['test'],device); save_json(tm,out/'test_metrics.json'); print(tm)

if __name__=='__main__': main()
