import numpy as np
import pandas as pd
import torch
import dgl

import os
import sys
code_path =  os.path.dirname(os.path.abspath(__file__))    # /home/user-home/yujie/0_PBCNetv2/0_PBCNET/model_code/predict
code_path = code_path.rsplit("/", 1)[0]
sys.path.append(code_path)
from utilis.utilis import Extend, pkl_load, pkl_load_no_AR

code_path = code_path.rsplit("/", 1)[0]
print(code_path)


def collate_fn(samples):
    ligand1_dir = [  s.dir_1.values[0] for s in samples]
    ligand2_dir = [  s.dir_2.values[0] for s in samples]

    graph1_list = [pkl_load(s) for s in ligand1_dir]
    graph2_list = [pkl_load(s) for s in ligand2_dir]

    g1 = dgl.batch(graph1_list)
    g2 = dgl.batch(graph2_list)

    label_list = [s.Label.values[0] for s in samples]  # delta
    label1_list = [s.Label1.values[0] for s in samples]  # validation samples' labels
    label2_list = [s.Label2.values[0] for s in samples]  # referance train samples' labels

    return g1, \
           g2, \
           torch.tensor(label_list, dtype=torch.float32), \
           torch.tensor(label1_list, dtype=torch.float32), \
           torch.tensor(label2_list, dtype=torch.float32 ), \
           None, \
           None


def collate_fn_fep(samples):
    ligand1_dir = [f'{code_path}/data/FEP/pose_graph/' + s.Ligand1.values[0] for s in samples]
    ligand2_dir = [f'{code_path}/data/FEP/pose_graph/' + s.Ligand2.values[0] for s in samples]

    graph1_list = [pkl_load(s) for s in ligand1_dir]
    graph2_list = [pkl_load(s) for s in ligand2_dir]
    
    g1 = dgl.batch(graph1_list)
    g2 = dgl.batch(graph2_list)

    label_list = [s.Lable.values[0] for s in samples]  # delta
    label1_list = [s.Lable1.values[0] for s in samples]  # validation samples' labels
    label2_list = [s.Lable2.values[0] for s in samples]  # referance train samples' labels

    return g1, \
           g2, \
           torch.tensor(label_list, dtype=torch.float32), \
           torch.tensor(label1_list, dtype=torch.float32), \
           torch.tensor(label2_list, dtype=torch.float32 ), \
           None, \
           None


def collate_fn_fep_nobond(samples, type_graph):
    ligand1_dir = [s.Ligand1.values[0].rsplit(".", 1)[0].replace('pose','pose_final') + f'_dgl_group_{type_graph}.pkl' for s in samples]
    ligand2_dir = [s.Ligand2.values[0].rsplit(".", 1)[0].replace('pose','pose_final') + f'_dgl_group_{type_graph}.pkl' for s in samples]

    graph1_list = [pkl_load(s) for s in ligand1_dir]
    graph2_list = [pkl_load(s) for s in ligand2_dir]
    
    g1 = dgl.batch(graph1_list)
    g2 = dgl.batch(graph2_list)

    label_list = [s.Lable.values[0] for s in samples]  # delta
    label1_list = [s.Lable1.values[0] for s in samples]  # validation samples' labels
    label2_list = [s.Lable2.values[0] for s in samples]  # referance train samples' labels

    return g1, \
           g2, \
           torch.tensor(label_list, dtype=torch.float32), \
           torch.tensor(label1_list, dtype=torch.float32), \
           torch.tensor(label2_list, dtype=torch.float32 ), \
           None, \
           None

def collate_fn_fep_ft(samples):
    sys_name1 = [s.Ligand1.values[0].split('/')[-2] for s in samples]
    sys_name2 = [s.Ligand2.values[0].split('/')[-2] for s in samples]

    lig_name1 = [s.Ligand1.values[0].split('/')[-1].split('.')[0] + '_dgl_group.pkl' for s in samples]
    lig_name2 = [s.Ligand2.values[0].split('/')[-1].split('.')[0] + '_dgl_group.pkl' for s in samples]
    
    ligand1_dir = []
    for sys, lig in zip(sys_name1, lig_name1):
        ligand1_dir.append(f'{code_path}/data/FEP/pose_graph/{sys}/{lig}')
    ligand2_dir = []
    for sys, lig in zip(sys_name2, lig_name2):
        ligand2_dir.append(f'{code_path}/data/FEP/pose_graph/{sys}/{lig}')

    graph1_list = [pkl_load(s) for s in ligand1_dir]
    graph2_list = [pkl_load(s) for s in ligand2_dir]
    
    g1 = dgl.batch(graph1_list)
    g2 = dgl.batch(graph2_list)

    label_list = [s.Lable.values[0] for s in samples]  # delta
    label1_list = [s.Lable1.values[0] for s in samples]  # validation samples' labels
    label2_list = [s.Lable2.values[0] for s in samples]  # referance train samples' labels

    return g1, \
           g2, \
           torch.tensor(label_list, dtype=torch.float32), \
           torch.tensor(label1_list, dtype=torch.float32), \
           torch.tensor(label2_list, dtype=torch.float32 ), \
           None, \
           None




def collate_fn_test(samples):
    ligand1_dir = [s.Ligand1.values[0] for s in samples]
    ligand2_dir = [s.Ligand2.values[0] for s in samples]

    graph1_list = [pkl_load(s) for s in ligand1_dir]
    graph2_list = [pkl_load(s) for s in ligand2_dir]
    
    g1 = dgl.batch(graph1_list)
    g2 = dgl.batch(graph2_list)

    label_list = [s.Lable.values[0] for s in samples]  # delta
    label1_list = [s.Lable1.values[0] for s in samples]  # validation samples' labels
    label2_list = [s.Lable2.values[0] for s in samples]  # referance train samples' labels

    return g1, \
           g2, \
           torch.tensor(label_list, dtype=torch.float32), \
           torch.tensor(label1_list, dtype=torch.float32), \
           torch.tensor(label2_list, dtype=torch.float32 ), \
           None, \
           None



class LeadOptDataset():
    def __init__(self, df_path, label_scalar=None):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.label_scalar = label_scalar

        if self.label_scalar == "finetune":
            label = self.df.Lable.values
            label = (np.array(label).astype(float) - 0.04191832) / 1.34086546
            self.df["Lable"] = label

        elif self.label_scalar is not None:
            label = self.df.Lable.values
            label = np.reshape(label, (-1, 1))
            self.label_scalar = self.label_scalar.fit(label)
            label = self.label_scalar.transform(label)
            self.df["Lable"] = label.flatten()

        self.df = self.df
        super(LeadOptDataset, self).__init__()

            
    def file_names_(self):
        ligand_dir = self.df.Ligand1.values
        file_names = [s.rsplit("/", 2)[1] for s in ligand_dir]
        return list(set(file_names))

        
    def __getitem__(self, idx):
        return self.df[idx:idx + 1]

    def __len__(self):
        return len(self.df)


class LeadOptDataset_retrain():
    def __init__(self, df_path, corr_path, avoid_forget=0):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)

        corr = pd.read_csv(corr_path)
        corr_small = corr[corr.spearman <= 0.5].file_name.values

        self.df["file_name"] = [i.rsplit("/",2)[1] for i in self.df.Ligand1.values]

        self.df_new = self.df[self.df["file_name"].isin(corr_small)]

        if avoid_forget == 1:
            self.df_good_part = self.df[~self.df["file_name"].isin(corr_small)]
            self.df_good_part = self.df_good_part.sample(n=len(self.df_new), replace=False, random_state=2)
            self.df_new = pd.concat([self.df_new,self.df_good_part], ignore_index=True)
            
        super(LeadOptDataset_retrain, self).__init__()
        
    def __getitem__(self, idx):
        return self.df_new[idx:idx + 1]

    def __len__(self):
        return len(self.df_new)


class LeadOptDataset_test():
    def __init__(self, df_path, label_scalar=None):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.label_scalar = label_scalar

        if self.label_scalar == "finetune":
            label = self.df.Lable.values
            label = (np.array(label).astype(float) - 0.04191832) / 1.34086546
            self.df["Lable"] = label

        elif self.label_scalar is not None:
            label = self.df.Lable.values
            label = np.reshape(label, (-1, 1))
            self.label_scalar = self.label_scalar.fit(label)
            label = self.label_scalar.transform(label)
            self.df["Lable"] = label.flatten()

        self.df = self.df[0:256]
        super(LeadOptDataset_test, self).__init__()

    def file_names_(self):
        ligand_dir = self.df.Ligand1.values
        file_names = [s.rsplit("/", 2)[1] for s in ligand_dir]
        return list(set(file_names))

    def __getitem__(self, idx):
        return self.df[idx:idx + 1]

    def __len__(self):
        return len(self.df)