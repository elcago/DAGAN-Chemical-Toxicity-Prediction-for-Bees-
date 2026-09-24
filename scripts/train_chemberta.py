"""Fine-tune ChemBERTa on the same data split."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data import load_apistox, split_indices
from src.metrics import classification_metrics
from src.utils import load_config, save_json, set_seed

class SmilesDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer):
        self.enc=tokenizer(list(smiles),padding=True,truncation=True,max_length=256)
        self.labels=list(map(int,labels))
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        item={k:torch.tensor(v[i]) for k,v in self.enc.items()}; item['labels']=torch.tensor(self.labels[i]); return item

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--checkpoint',default='DeepChem/ChemBERTa-77M-MTR'); p.add_argument('--out',default='results/chemberta'); a=p.parse_args()
    cfg=load_config(ROOT/a.config); set_seed(cfg['seed']); df=load_apistox({**cfg['data'],'csv_path':ROOT/cfg['data']['csv_path']}); tr,va,te=split_indices(df,cfg['data'],cfg['seed'])
    tok=AutoTokenizer.from_pretrained(a.checkpoint); model=AutoModelForSequenceClassification.from_pretrained(a.checkpoint,num_labels=2,ignore_mismatched_sizes=True)
    train=SmilesDataset(df.iloc[tr].smiles,df.iloc[tr].target,tok); val=SmilesDataset(df.iloc[va].smiles,df.iloc[va].target,tok); test=SmilesDataset(df.iloc[te].smiles,df.iloc[te].target,tok)
    out=ROOT/a.out; out.mkdir(parents=True,exist_ok=True)
    args=TrainingArguments(output_dir=str(out/'hf'),learning_rate=2e-5,per_device_train_batch_size=16,per_device_eval_batch_size=32,num_train_epochs=10,weight_decay=0.01,eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='eval_loss',report_to=[])
    trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=val,tokenizer=tok); trainer.train(); pred=trainer.predict(test); prob=torch.softmax(torch.tensor(pred.predictions),1)[:,1].numpy(); metrics=classification_metrics(df.iloc[te].target.values,prob); save_json(metrics,out/'test_metrics.json'); print(metrics)
if __name__=='__main__': main()
