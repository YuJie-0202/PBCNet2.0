import argparse
import os
import random
import time
from collections import defaultdict

code_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(code_path + '/model_code')
code_path = code_path.rsplit("/", 1)[0]     # /home/user-home/yujie/0_PBCNetv2/0_PBCNET

import numpy as np
import pandas as pd
import torch

from Dataloader.dataloader import collate_fn_fep_ft,LeadOptDataset
from torch.utils.data import DataLoader
from models.readout import PBCNetv2
from predict.predict import predict
from utilis.function import get_loss_func
from utilis.scheduler import NoamLR_shan
from utilis.trick import Writer,makedirs

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


def softmax(x):
    x_exp = np.exp(x)
    # axis=0
    x_sum = np.sum(x_exp, axis=0)
    s = x_exp / x_sum
    return s


def setup_cpu(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

Path = f'{code_path}/data/FEP/finetune_input/'

# args: device, loss_function, continue_learning, retrain,batch_size,init_lr, max_lr, final_lr
def FT(args, model, train_loader, val_loader, test_loader2, test_loader, device, num_training_samples, file_names,ref,num_again):

    train_start = time.time()
    
    opt = torch.optim.Adam(model.parameters(), lr=0.00001)
    loss_func = get_loss_func(args.loss_function)

    mae_func = torch.nn.L1Loss(reduction='sum')
    mse_func = torch.nn.MSELoss(reduction='sum')

    save_dir = os.path.join(code_path, "results", 'finetune', f"{args.finetune_filename}",f"ref{ref}_results")
    logger_writer = Writer(os.path.join(save_dir, f"{num_again}_record.txt"))

    stop_metric = 0
    not_change_epoch = 0

    # for log in a batch defined step
    batch_all = 0
    loss_for_log = 0

    #  ===============  Performance evaluation before fine tuning ===================
    # without finetuning ligands
    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels,_,_ = predict(model, test_loader, device)

    df = pd.read_csv( f"{Path}/{ref}_reference/{args.which_fep}_temp_predict.csv")
    abs_label = np.array(df.Lable.values).astype(float) + np.array(df.Lable2.values).astype(float)
    abs_predict = np.array(valid_prediction).astype(float) + np.array(df.Lable2.values).astype(float)

    # =================  ranking related indicators ====================
    Ligand1 = df.Ligand1_num.values

    _df = pd.DataFrame({"Ligand1":Ligand1, "abs_label":abs_label, "abs_predict":abs_predict})
    _df_group = _df.groupby('Ligand1')[['abs_label', 'abs_predict']].mean().reset_index()

    spearman = _df_group[["abs_label", "abs_predict"]].corr(method='spearman').iloc[0, 1]
    pearson = _df_group[["abs_label", "abs_predict"]].corr(method='pearson').iloc[0, 1]
    kendall = _df_group[["abs_label", "abs_predict"]].corr(method='kendall').iloc[0, 1]

    logger_writer(f"without ligands RMSE_G {rmse_g} Spearman {spearman} Pearson {pearson} Kendall {kendall}")


    # with fintuning ligands
    mae2, rmse2, mae_g2, rmse_g2, valid_prediction2, valid_prediction_G2, valid_labels2,_,_ = predict(model, test_loader2, device)

    df2 = pd.read_csv( f"{Path}/{ref}_reference/{args.which_fep}_temp_predict_withtuneligs.csv")
    abs_label2 = np.array(df2.Lable.values).astype(float) + np.array(df2.Lable2.values).astype(float)
    abs_predict2 = np.array(valid_prediction2).astype(float) + np.array(df2.Lable2.values).astype(float)

    Ligand1_2 = df2.Ligand1_num.values

    _df2 = pd.DataFrame({"Ligand1":Ligand1_2, "abs_label":abs_label2, "abs_predict":abs_predict2})
    _df_group2 = _df2.groupby('Ligand1')[['abs_label', 'abs_predict']].mean().reset_index()

    spearman2 = _df_group2[["abs_label", "abs_predict"]].corr(method='spearman').iloc[0, 1]
    pearson2 = _df_group2[["abs_label", "abs_predict"]].corr(method='pearson').iloc[0, 1]
    kendall2 = _df_group2[["abs_label", "abs_predict"]].corr(method='kendall').iloc[0, 1]


    logger_writer(f"with ligands RMSE_G {rmse_g2} Spearman {spearman2} Pearson {pearson2} Kendall {kendall2}")
    logger_writer(" ")


    # ====================  Finetune ======================

    for epoch in range(args.finetune_epoch):
        model.train()
        start = time.time()
        training_mae = 0
        training_mse = 0

        for batch_data in train_loader:
            graph1, graph2, label, label1, label2, rank1, file_name = batch_data
            # to cuda
            graph1, graph2, label, label1, label2 = (graph1.to(device),
                                                     graph2.to(device),
                                                     label.to(device),
                                                     label1.to(device),
                                                     label2.to(device))

            logits, logits_net = model(graph1,graph2) 

            loss = loss_func(logits.squeeze(dim=-1).float(), label.float())
            train_mae_ = mae_func(logits.squeeze(dim=-1).float(), label.float())
            train_mse_ = mse_func(logits.squeeze(dim=-1).float(), label.float())

            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_all += 1
            loss_for_log += loss.item()
            training_mae += train_mae_.item()
            training_mse += train_mse_.item()

        _loss = loss_for_log / num_training_samples  # mean loss for each batch with size of log_frequency.

        logger_writer("    ")
        # print(f"Epoch {epoch}  Batch {batch_all}  Loss {_loss}")
        logger_writer(f"Epoch {epoch}  Loss {_loss}")
        loss_for_log = 0

        # ============= test [FEP1 + FEP2] ==============
        train_time = time.time()
        training_rmse = (training_mse / num_training_samples) ** 0.5
        training_mae = training_mae / num_training_samples

        # without finetuning ligands
        mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels,_,_ = predict(model, test_loader, device)

        if epoch == 0:
            prediction_of_file = pd.DataFrame({f'prediction_ic50_{epoch}': valid_prediction,
                                               f'prediction_G_{epoch}': valid_prediction_G,
                                               f"label_ic50_{epoch}": valid_labels})
        else:
            prediction_of_file_ = pd.DataFrame({f'prediction_ic50_{epoch}': valid_prediction,
                                                f'prediction_G_{epoch}': valid_prediction_G,
                                                f"label_ic50_{epoch}": valid_labels})
            prediction_of_file = pd.merge(prediction_of_file, prediction_of_file_, how="outer",
                                          right_index=True, left_index=True)

        df = pd.read_csv( f"{Path}/{ref}_reference/{args.which_fep}_temp_predict.csv")
        abs_label = np.array(df.Lable.values).astype(float) + np.array(df.Lable2.values).astype(float)
        abs_predict = np.array(valid_prediction).astype(float) + np.array(df.Lable2.values).astype(float)

        # ================= ranking related indicators ====================
        Ligand1 = df.Ligand1_num.values

        _df = pd.DataFrame({"Ligand1":Ligand1, "abs_label":abs_label, "abs_predict":abs_predict})
        _df_group = _df.groupby('Ligand1')[['abs_label', 'abs_predict']].mean().reset_index()

        spearman = _df_group[["abs_label", "abs_predict"]].corr(method='spearman').iloc[0, 1]
        pearson = _df_group[["abs_label", "abs_predict"]].corr(method='pearson').iloc[0, 1]
        kendall = _df_group[["abs_label", "abs_predict"]].corr(method='kendall').iloc[0, 1]

        logger_writer(f"Training Set mae {training_mae}")
        logger_writer(f"Training Set rmse {training_rmse}")
        logger_writer(f"Epoch {epoch}")
        logger_writer(f"without ligands RMSE_G {rmse_g} Spearman {spearman} Pearson {pearson} Kendall {kendall}")


        # with finetuning ligands
        mae2, rmse2, mae_g2, rmse_g2, valid_prediction2, valid_prediction_G2, valid_labels2,_,_ = predict(model, test_loader2, device)

        if epoch == 0:
            prediction_of_file2 = pd.DataFrame({f'prediction_ic50_{epoch}': valid_prediction2,
                                               f'prediction_G_{epoch}': valid_prediction_G2,
                                               f"label_ic50_{epoch}": valid_labels2})
        else:
            prediction_of_file_2 = pd.DataFrame({f'prediction_ic50_{epoch}': valid_prediction2,
                                                f'prediction_G_{epoch}': valid_prediction_G2,
                                                f"label_ic50_{epoch}": valid_labels2})
            prediction_of_file2 = pd.merge(prediction_of_file2, prediction_of_file_2, how="outer",
                                          right_index=True, left_index=True)

        df2 = pd.read_csv( f"{Path}/{ref}_reference/{args.which_fep}_temp_predict_withtuneligs.csv")
        abs_label2 = np.array(df2.Lable.values).astype(float) + np.array(df2.Lable2.values).astype(float)
        abs_predict2 = np.array(valid_prediction2).astype(float) + np.array(df2.Lable2.values).astype(float)

        # ================= ranking related indicators ====================
        Ligand1_2 = df2.Ligand1_num.values

        _df2 = pd.DataFrame({"Ligand1":Ligand1_2, "abs_label":abs_label2, "abs_predict":abs_predict2})
        _df_group2 = _df2.groupby('Ligand1')[['abs_label', 'abs_predict']].mean().reset_index()

        spearman2 = _df_group2[["abs_label", "abs_predict"]].corr(method='spearman').iloc[0, 1]
        pearson2 = _df_group2[["abs_label", "abs_predict"]].corr(method='pearson').iloc[0, 1]
        kendall2 = _df_group2[["abs_label", "abs_predict"]].corr(method='kendall').iloc[0, 1]

        logger_writer(f"with ligands RMSE_G {rmse_g2} Spearman {spearman2} Pearson {pearson2} Kendall {kendall2}")
        logger_writer(" ")





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=1, help="The number of device")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--which_fep', type=str, default="fep2")
    parser.add_argument('--seed', type=int, default=2,help='the random seed' )
    parser.add_argument('--loss_function', type=str, default="mse")
    parser.add_argument('--cpu_num', type=int, default=10,help='the number of cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--finetune_epoch', type=int, default=10)
    parser.add_argument('--finetune_filename', type=str, default="pfkfb3")
    

    args = parser.parse_args()

    setup_cpu(args.cpu_num)
    setup_seed(args.seed)

    cuda = "cuda:" + str(args.device)
    # cuda = 'cpu'

    fep1 = ['PTP1B', 'Thrombin', 'Tyk2', 'CDK2', 'Jnk1', 'Bace', 'MCL1', 'p38']
    fep2 = ['syk', 'shp2','pfkfb3',  'eg5', 'cdk8', 'cmet', 'tnks2', 'hif2a']

    if args.which_fep == "fep1":
        fep = fep1
        num_ = [2,3,4,5,6,7,8,9]
    else:
        fep = fep2
        num_ = [2,3,4,5,6,7,8,9,10]

    for finetune_filename in fep:

        args.finetune_filename = finetune_filename
        print(args.finetune_filename)

        for ref in num_:
            print(ref)

            df_finetune_all = pd.read_csv(f"{Path}/{ref}_reference/{args.which_fep}_finetune.csv")
            df_prediction_all = pd.read_csv(f"{Path}/{ref}_reference/{args.which_fep}_predict.csv")
            df_prediction2_all = pd.read_csv(f"{Path}/{ref}_reference/{args.which_fep}_predict_with_tuneligs.csv")

            df_finetune_all = df_finetune_all[df_finetune_all.file_name == args.finetune_filename]
            df_prediction_all = df_prediction_all[df_prediction_all.file_name == args.finetune_filename]
            df_prediction2_all = df_prediction2_all[df_prediction2_all.file_name == args.finetune_filename]

            df_finetune_all_group = df_finetune_all.groupby("again_number")
            df_prediction_all_group = df_prediction_all.groupby("again_number")
            df_prediction2_all_group = df_prediction2_all.groupby("again_number")

            for again_num, df_prediction in tqdm(df_prediction_all_group):

                model = torch.load(f"{code_path}//PBCNet2.pth", map_location=torch.device('cpu'), weights_only=False)
                model.to(cuda)

                df_prediction.to_csv( f"{Path}/{ref}_reference/{args.which_fep}_temp_predict.csv",index=0)

                # finetune
                for _again_num, df_finetune in df_finetune_all_group:
                    if _again_num == again_num:
                        df_finetune.to_csv(f"{Path}/{ref}_reference/{args.which_fep}_temp_finetune.csv",index=0)

                # prediction with tune ligands
                for _again_num, df_finetune in df_prediction2_all_group:
                    if _again_num == again_num:
                        df_finetune.to_csv(f"{Path}/{ref}_reference/{args.which_fep}_temp_predict_withtuneligs.csv",index=0)

                finetune_dataset = LeadOptDataset(f"{Path}/{ref}_reference/{args.which_fep}_temp_finetune.csv")

                num_training_samples = len(finetune_dataset)
                finetune_dataloader = DataLoader(finetune_dataset, collate_fn=collate_fn_fep_ft, batch_size=args.batch_size,
                                                      drop_last=False, shuffle=True,
                                                      num_workers=args.num_workers, pin_memory=True)

                prediction_dataset = LeadOptDataset(f"{Path}/{ref}_reference/{args.which_fep}_temp_predict.csv")
                prediction_dataloader = DataLoader(prediction_dataset, collate_fn=collate_fn_fep_ft,
                                                        batch_size=args.batch_size,
                                                        drop_last=False, shuffle=False,
                                                        num_workers=args.num_workers, pin_memory=True)

                prediction_dataset2 = LeadOptDataset(f"{Path}/{ref}_reference/{args.which_fep}_temp_predict_withtuneligs.csv")
                prediction_dataloader2 = DataLoader(prediction_dataset2, collate_fn=collate_fn_fep_ft,
                                                        batch_size=args.batch_size,
                                                        drop_last=False, shuffle=False,
                                                        num_workers=args.num_workers, pin_memory=True)

                FT(args=args, model=model, train_loader=finetune_dataloader, test_loader2=prediction_dataloader2,
                      test_loader=prediction_dataloader, device=cuda, num_training_samples=num_training_samples,
                      ref=ref, num_again=again_num,val_loader=None,file_names=None)
