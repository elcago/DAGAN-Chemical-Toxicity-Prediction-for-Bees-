from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data import add_descriptors, build_graph_dataset, load_apistox
from src.features import edge_feature_dim, node_feature_dim
from src.model import DAGAN
from src.utils import get_device, load_config, set_seed

class SingleGraphWrapper(nn.Module):
    def __init__(self, model, template):
        super().__init__(); self.model=model; self.template=template
    def forward(self,x,edge_index):
        d=self.template.clone(); d.x=x; d.edge_index=edge_index
        if not hasattr(d,'batch') or d.batch is None: d.batch=torch.zeros(x.size(0),dtype=torch.long,device=x.device)
        return self.model(d)

def fragment_smiles(smiles, important_edge_pairs):
    mol=Chem.MolFromSmiles(smiles)
    atoms=set()
    for i,j in important_edge_pairs: atoms.add(int(i)); atoms.add(int(j))
    if not atoms: return ''
    return Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(atoms),canonical=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--model-dir',default='results/dagan'); p.add_argument('--out',default='results/gnn_fragments.csv'); p.add_argument('--max-samples',type=int,default=100); a=p.parse_args()
    cfg=load_config(ROOT/a.config); set_seed(cfg['seed']); device=get_device(); md=ROOT/a.model_dir
    df=load_apistox({**cfg['data'],'csv_path':ROOT/cfg['data']['csv_path']}); df=add_descriptors(df,cfg['data']['descriptors'])
    scaler=joblib.load(md/'descriptor_scaler.joblib'); te=np.load(md/'split_indices.npz')['test'][:a.max_samples]; graphs=build_graph_dataset(df,te,cfg['data']['descriptors'],scaler)
    model=DAGAN(node_feature_dim(),edge_feature_dim(),len(cfg['data']['descriptors']),cfg['model']['hidden_dim'],cfg['model']['attention_heads'],cfg['model']['dropout'],cfg['model']['descriptor_hidden_dim'],cfg['model']['fusion_dim']).to(device)
    model.load_state_dict(torch.load(md/'best_model.pt',map_location=device)); model.eval(); rows=[]
    for g in graphs:
        g=g.to(device); g.batch=torch.zeros(g.num_nodes,dtype=torch.long,device=device)
        wrapper=SingleGraphWrapper(model,g).to(device)
        explainer=Explainer(model=wrapper,algorithm=GNNExplainer(epochs=200),explanation_type='model',node_mask_type='attributes',edge_mask_type='object',model_config=dict(mode='multiclass_classification',task_level='graph',return_type='raw'))
        exp=explainer(g.x,g.edge_index,index=0)
        mask=exp.edge_mask.detach().cpu().numpy(); ei=g.edge_index.detach().cpu().numpy().T
        keep=ei[mask>cfg['explainability']['edge_mask_threshold']]
        frag=fragment_smiles(g.smiles,keep)
        rows.append({'sample_index':int(g.sample_index.item()),'smiles':g.smiles,'label':int(g.y.item()),'important_fragment_smiles':frag,'num_selected_directed_edges':int(len(keep)),'max_edge_mask':float(mask.max()) if len(mask) else 0.0})
    out=ROOT/a.out; out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print(f'Saved {out}')
if __name__=='__main__': main()
