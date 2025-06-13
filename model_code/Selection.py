# import
import pandas as pd
import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
import sys
import os
import random

code_path =  os.path.dirname(os.path.abspath(__file__))    # /home/user-home/yujie/0_PBCNetv2/0_PBCNET/model_code/train.py/
sys.path.append(code_path)
code_path = code_path.rsplit("/", 1)[0]   # /home/user-home/yujie/0_PBCNetv2/0_PBCNET   # data need: {code_path}/data/Selection/

from utilis.function import get_loss_func
from utilis.trick import Writer
from utilis.utilis import pkl_load
import dgl
 


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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


def collate_fn(samples):
    ligand1_dir = [f'{code_path}/data/Selection/' + s.dir_1.values[0] for s in samples]
    ligand2_dir = [f'{code_path}/data/Selection/' + s.dir_2.values[0] for s in samples]

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


# def freezen(model):
#     need_updata = ['FNN.0.weight', 'FNN.0.bias', 'FNN.2.weight', 'FNN.2.bias', 'FNN.4.weight', 'FNN.4.bias', 'FNN.6.weight', 'FNN.6.bias',
#                    'FNN2.0.weight', 'FNN2.0.bias', 'FNN2.2.weight', 'FNN2.2.bias', 'FNN2.4.weight', 'FNN2.4.bias', 'FNN2.6.weight', 'FNN2.6.bias']

#     for name, parameter in model.named_parameters():
#         if name in need_updata:
#             parameter.requires_grad = False
#         else:
#             parameter.requires_grad = True


def input_generation(file_name, ref, newly_mols):

    ddd = f"{code_path}/data/Selection/{file_name}/input_file/{ref}_reference/predict.csv"
    if os.path.exists(ddd) is False:
        os.system(f' mkdir {ddd.rsplit("/",1)[0]} ')

    ligand1_tra = []
    ligand2_tra = []
    Lable1_tra = []
    Lable2_tra = []
    Lable_tra = []
    Rank1_tra = []
    Rank2_tra = []
    Rank_last_tra = []
    ligand1_num_tra = []
    ligand2_num_tra = []
    reference_num_tra = []

    ligand1_val = []
    ligand2_val = []
    Lable1_val = []
    Lable2_val = []
    Lable_val = []
    Rank1_val = []
    Rank2_val = []
    Rank_last_val = []
    ligand1_num_val = []
    ligand2_num_val = []
    reference_num_val = []


    # load all ligand in a series
    file_name_ = f"{code_path}/data/Selection/{file_name}/"

    ligands = [i for i in os.listdir(file_name_) if
               i.rsplit(".", 1)[-1] == "pkl" and i.rsplit(".", 1)[0] != "pocket"]

    ligands = [file_name_ + i for i in ligands]

    # load refs of last epch
    pd_for_refs = pd.read_csv(f"{file_name_}/input_file/{ref-3}_reference/predict.csv")
    refs = list(set(pd_for_refs.lig2.values))
    # merge
    refs.extend(newly_mols)
    refs = [f"{file_name_}{file_name.split('_')[0]}_" + i + ".pkl" for i in refs]

    # label
    label_csv = f"{file_name_}/label.csv"
    df = pd.read_csv(label_csv)


    for i in range(len(refs)):

        reference_ligand = refs[i]

        reference_ligand_number = i + 1

        # finetune
        for query_ligand in refs:

            if query_ligand == reference_ligand:
                continue

            ligand1_val.append(query_ligand.split('Selection/')[-1])
            ligand2_val.append(reference_ligand.split('Selection/')[-1])

            ligand1_number = str(query_ligand.rsplit('.', 1)[0].split("/")[-1].split("_", 1)[1])
            ligand2_number = str(reference_ligand.rsplit('.', 1)[0].split("/")[-1].split("_", 1)[1])

            ligand1_num_val.append(ligand1_number)
            ligand2_num_val.append(ligand2_number)

            lable1 = float(df[df.name == ligand1_number].pIC50.iloc[0])
            lable2 = float(df[df.name == ligand2_number].pIC50.iloc[0])
            Lable1_val.append(lable1)
            Lable2_val.append(lable2)
            Lable_val.append(lable1 - lable2)

            index1 = ligands.index(query_ligand)
            index2 = ligands.index(reference_ligand)
            Rank1_val.append(index1)
            Rank2_val.append(index2)
            Rank_last_val.append(max(index1, index2))
            reference_num_val.append(reference_ligand_number)

        # predict
        for query_ligand in ligands:
            if query_ligand in refs:
                continue

            ligand1_tra.append(query_ligand.split('Selection/')[-1])
            ligand2_tra.append(reference_ligand.split('Selection/')[-1])

            ligand1_number = str(query_ligand.rsplit('.', 1)[0].split("/")[-1].split("_", 1)[1])
            ligand2_number = str(reference_ligand.rsplit('.', 1)[0].split("/")[-1].split("_", 1)[1])

            ligand1_num_tra.append(ligand1_number)
            ligand2_num_tra.append(ligand2_number)

            lable1 = float(df[df.name == ligand1_number].pIC50.iloc[0])
            lable2 = float(df[df.name == ligand2_number].pIC50.iloc[0])

            Lable1_tra.append(lable1)
            Lable2_tra.append(lable2)
            Lable_tra.append(lable1 - lable2)

            index1 = ligands.index(query_ligand)
            index2 = ligands.index(reference_ligand)
            Rank1_tra.append(index1)
            Rank2_tra.append(index2)
            Rank_last_tra.append(max(index1, index2))
            reference_num_tra.append(reference_ligand_number)

    df_predict = pd.DataFrame(
        {'reference_num': reference_num_tra,
         'dir_1': ligand1_tra, "dir_2": ligand2_tra, "Label": Lable_tra, \
         "Label1": Lable1_tra, "Label2": Lable2_tra, \
         "Rank1": Rank1_tra, "Rank2": Rank2_tra, "Rank": Rank_last_tra, "lig1": ligand1_num_tra, \
         "lig2": ligand2_num_tra})

    df_finetune = pd.DataFrame(
        {'reference_num': reference_num_val,
         'dir_1': ligand1_val, "dir_2": ligand2_val, "Label": Lable_val, \
         "Label1": Lable1_val, "Label2": Lable2_val, \
         "Rank1": Rank1_val, "Rank2": Rank2_val, "Rank": Rank_last_val, "lig1": ligand1_num_val, \
         "lig2": ligand2_num_val})

    new_ref_pre_csv =  f"{file_name_}/input_file/{ref}_reference/predict.csv"
    new_ref_fin_csv = f"{file_name_}/input_file/{ref}_reference/finetune.csv"

    df_predict.to_csv(new_ref_pre_csv, index=0)
    df_finetune.to_csv(new_ref_fin_csv, index=0)


def model_predict(file_name, ref, model, device, logger_writer):
    
    # load predict data
    file_name_ = f"{code_path}/data/Selection/{file_name}/"

    test_dataset = LeadOptDataset(
        f"{file_name_}/input_file/{ref}_reference/predict.csv")
    test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=20,
                                      drop_last=False, shuffle=False)

    t_pd = pd.read_csv(f"{file_name_}/input_file/{ref}_reference/predict.csv")
    names = t_pd.lig1.values

    # pre
    pres = []
    labels = []
    model.eval()
    for batch_data in test_dataloader:
        graph1, graph2, label, label1, label2, _, _ = batch_data
        # to cuda
        graph1, graph2, label, label1, label2 = graph1.to(device), graph2.to(device),label.to(device), label1.to(device), label2.to(device)
        logits,_ = model(graph1,
                         graph2)
        pre = logits.squeeze() + label2
        labels += label1.tolist()
        pres += pre.tolist()

    result_df = pd.DataFrame({"pre": pres, "label": labels, "mol_name": names})

    if ref != 1:
        std_ = result_df.groupby('mol_name')[['pre', 'label']].std().reset_index().pre.values

    result_df = result_df.groupby('mol_name')[['pre', 'label']].mean().reset_index()
    if ref != 1:
        result_df["uncertainty"] = std_

    s = result_df[["pre", "label"]].corr(method='spearman').iloc[0, 1]
    p = result_df[["pre", "label"]].corr(method='pearson').iloc[0, 1]

    mol_name = result_df.mol_name.values
    pre_ic50 = result_df.pre.values
    if ref != 1:
        uncertainty = result_df.uncertainty.values
    if ref != 1:
        ucb = pre_ic50 #- 2 * uncertainty
    else:
        ucb = pre_ic50

    ind = np.argsort(ucb)[-3:][::-1]
    # ind = np.argsort(pre_ic50)[-3:][::-1]
    ind = mol_name[ind]

    logger_writer(f"Prediction:")
    logger_writer(f"    reference number: {ref}")
    logger_writer(f"    spearman: {s}")
    logger_writer(f"    pearson: {p}")
    logger_writer(f"    selected molecule: {ind}")
    logger_writer(f" ")

    return ind


def model_finetune(file_name, ref, model, device, logger_writer):

    file_name_ = f"{code_path}/data/Selection/{file_name}/"
    train_dataset = LeadOptDataset(
        f"{file_name_}/input_file/{ref}_reference/finetune.csv")
    train_dataloader = GraphDataLoader(train_dataset, collate_fn=collate_fn, batch_size=20,
                                       drop_last=False, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=0.000001)
    loss_func = get_loss_func("mse")
    for epoch in range(3):
        model.train()
        for batch_data in train_dataloader:
            graph1, graph2, label, label1, label2, _,_ = batch_data
            # to cuda
            graph1, graph2, label, label1, label2 = graph1.to(device), graph2.to(device), label.to(device), label1.to(device), label2.to(device)

            logits,_ = model(graph1,
                             graph2)
            loss = loss_func(logits.squeeze(dim=-1).float(), label.float())

            opt.zero_grad()
            loss.backward()
            opt.step()

    logger_writer(f"Finetuned with {ref} molecules.")
    # print(f"Finetuned with {ref} molecules.")

    return model


best_mols = \
    {'FGFR2_pose': ['lig_38'],
     'BCL6_pose':  ['lig_1'],
     'HO1_pose':   ['lig_7i'],
     'LRRK2_pose': ['lig_11'],
     'sEH_pose':   ['lig_11h'],
     'CDK9_pose':  ['lig_16'],
     'WDR5_pose':  ['lig_27'],
     'AAK1_pose':  ['lig_(S)-32'],
     'PSK13_pose': ['lig_37']}

device = "cuda:1"
seed = [0,1,2,3,4,5,6,7,8,9]
setup_seed(seed[0])

select_files = [ 'FGFR2_pose',  'BCL6_pose','HO1_pose', 'LRRK2_pose', "sEH_pose",'CDK9_pose', 'WDR5_pose', 'AAK1_pose', 'PSK13_pose']



from tqdm import tqdm
for file_name in tqdm(select_files):
    print(file_name)
    # 记录
    logger_writer = Writer(f"{code_path}/results/select.txt")
    logger_writer(file_name)

    model = torch.load(f"{code_path}//PBCNet2.pth", map_location=torch.device(device),weights_only=False)

    # freezen(model)

    order = []
    for ref in range(1, 1000, 3):


        # the first step, do not finetune
        if ref == 1:
            selected_mols = model_predict(file_name, ref, model, device, logger_writer)
        else:
            input_generation(file_name, ref, newly_mols=selected_mols)

            # model = model_finetune(file_name, ref, model, device, logger_writer)
            # selected_mols = model_predict(file_name, ref, model, device, logger_writer)

            # connected
            model2 = model_finetune(file_name, ref, model, device, logger_writer)
            selected_mols = model_predict(file_name, ref, model2, device, logger_writer)

        n = 0
        for selected_mol in selected_mols:
            n += 1
            if selected_mol in best_mols[file_name]:
                best_mols[file_name].remove(selected_mol)

                order.append(ref - 1 + n)

                logger_writer(f"{selected_mol} is selected!")
                logger_writer(f"order: {ref - 1 + n}")
                logger_writer(" ")

        if not best_mols[file_name]:
            logger_writer(f"The iteration of AL is over!")
            break
    
    logger_writer(f"{file_name} each order is {order}, mean order is {np.mean(order)}")
    print(f"{file_name} each order is {order}, mean order is {np.mean(order)}")
    logger_writer(" ")
    logger_writer(" ")
    logger_writer(" ")







