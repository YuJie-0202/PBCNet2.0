from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import numpy as np
import pandas as pd
import os
import sys
from functools import partial

code_path =  os.path.dirname(os.path.abspath(__file__))    # /home/user-home/yujie/0_PBCNetv2/0_PBCNET/model_code/predict
code_path = code_path.rsplit("/", 1)[0]
sys.path.append(code_path)
code_path = code_path.rsplit("/", 1)[0]

from Dataloader.dataloader import collate_fn_fep, LeadOptDataset,collate_fn_test, collate_fn_fep_nobond
from torch.utils.data import DataLoader

@torch.no_grad()
def predict(model, loader, device,bb=False):
    model.eval()

    valid_prediction = []
    valid_labels = []
    ref_1_labels = []  # ref
    valid_2_labels = []  # query

    att__1 = []
    att__2 = []

    for batch_data in loader:
        graph1, graph2, label, label1, label2, rank1, file_name = batch_data
        # to cuda
        graph1, graph2,  label, label1, label2 = (graph1.to(device), graph2.to(device), label.to(device), label1.to(
            device), label2.to(device))

        logits,_ = model(graph1,
                       graph2,bb)

        valid_prediction += logits.tolist()
        valid_labels += label.tolist()
        ref_1_labels += label1.tolist()
        valid_2_labels += label2.tolist()


    mae = mean_absolute_error(valid_labels, valid_prediction)
    rmse = mean_squared_error(valid_labels, valid_prediction) ** 0.5

    # ======== to 'kcal/mol' unit =======
    valid_labels_G = np.log(np.power(10, -np.array(valid_labels).astype(float)))*297*1.9872*1e-3
    valid_prediction_G = np.log(np.power(10, -np.array(valid_prediction).astype(float)))*297*1.9872*1e-3

    mae_g = mean_absolute_error(valid_labels_G, valid_prediction_G)
    rmse_g = mean_squared_error(valid_labels_G, valid_prediction_G) ** 0.5

    valid_prediction = np.array(valid_prediction).flatten()
    valid_prediction_G = np.array(valid_prediction_G).flatten()

    return mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G,np.array(valid_labels),np.array(ref_1_labels),np.array(valid_2_labels)


def test_fep(fep,logger_writer,model,device,code_path,batch_size):
    # fep: FEP1 or FEP2
    rmse_gs = []
    spearmans = []
    pearsons = []
    kendalls = []

    spearmans_var = []
    pearsons_var = []
    kendalls_var = []

    spearmans_min = []
    pearsons_min = []
    kendalls_min = []

    spearmans_max = []
    pearsons_max = []
    kendalls_max = []

    spearmans_10 = []
    pearson_10 = []

    if fep == 'FEP1':
        test_file_name_FEP = ['CDK2', 'Tyk2', 'Bace', 'Jnk1', 'PTP1B', 'MCL1', 'p38', 'Thrombin']
       
    if fep == 'FEP2':
        test_file_name_FEP = ['pfkfb3', 'shp2', 'eg5', 'hif2a', 'cdk8', 'syk', 'cmet', 'tnks2']
        
    for file_name in test_file_name_FEP:
        # load the test data 
        # for one system
        df_file = pd.read_csv(f"{code_path}/data/FEP/direct_input/{file_name}.csv")
        test_dataset = LeadOptDataset(f"{code_path}/data/FEP/direct_input/{file_name}.csv")
        test_dataloader = DataLoader(test_dataset,
                                     collate_fn=collate_fn_fep,
                                     batch_size=batch_size,
                                     drop_last=False,
                                     shuffle=False,
                                     pin_memory=False)

        mae,rmse,mae_g,rmse_g,valid_prediction,valid_prediction_G,valid_labels,ref_1_label,val_2_label= predict(model, test_dataloader, device)
        
        # rmsd
        rmse_gs.append(rmse_g)
        # save rmsd
        if file_name == test_file_name_FEP[0]:
            prediction_of_FEP = pd.DataFrame({f'prediction_ic50_{file_name}': valid_prediction,
                                               f'prediction_G_{file_name}': valid_prediction_G,
                                               f"label_ic50_{file_name}": valid_labels})
        else:
            prediction_of_FEP_ = pd.DataFrame({f'prediction_ic50_{file_name}': valid_prediction,
                                                f'prediction_G_{file_name}': valid_prediction_G,
                                                f"label_ic50_{file_name}": valid_labels})
            prediction_of_FEP = pd.merge(prediction_of_FEP, prediction_of_FEP_, how="outer",right_index=True,left_index=True)

        # coor
        abs_label = val_2_label
        abs_predict = np.array(ref_1_label).astype(float) - np.array(valid_prediction).astype(float)

        df_file["abs_label_p"] = abs_label
        df_file["abs_predict_p"] = abs_predict

        reference_num = df_file.reference_num.values
        ligand1_num = df_file.Ligand1_num.values
        _df = pd.DataFrame({"reference_num": reference_num, f"abs_label_{file_name}": abs_label, f"abs_predict_{file_name}": abs_predict, f"ligand1_num_{file_name}": ligand1_num})

        # save coor
        if file_name == test_file_name_FEP[0]:
            corr_of_FEP = _df.groupby(f'ligand1_num_{file_name}')[[f'abs_label_{file_name}', f'abs_predict_{file_name}']].mean().reset_index()
        else:
            corr_of_FEP_ = _df.groupby(f'ligand1_num_{file_name}')[[f'abs_label_{file_name}', f'abs_predict_{file_name}']].mean().reset_index()
            corr_of_FEP = pd.merge(corr_of_FEP, corr_of_FEP_, how="outer",right_index=True,left_index=True)

        # 
        _df_group = _df.groupby('reference_num')

        spearman_ = []
        pearson_ = []

        for _, _df_onegroup in _df_group:
            spearman = _df_onegroup[[f"abs_label_{file_name}", f"abs_predict_{file_name}"]].corr(method='spearman').iloc[0, 1]
            pearson = _df_onegroup[[f"abs_label_{file_name}", f"abs_predict_{file_name}"]].corr(method='pearson').iloc[0, 1]
            spearman_.append(spearman)
            pearson_.append(pearson)

        s_,p_ = np.mean(spearman_),np.mean(pearson_)
        spearmans.append(np.mean(spearman_))
        pearsons.append(np.mean(pearson_))

        s_var_,p_var_ = np.var(spearman_),np.var(pearson_)
        spearmans_var.append(np.var(spearman_))
        pearsons_var.append(np.var(pearson_))

        s_min_,p_min_ = np.min(spearman_),np.min(pearson_)
        spearmans_min.append(np.min(spearman_))
        pearsons_min.append(np.min(pearson_))

        s_max_,p_max_ = np.max(spearman_),np.max(pearson_)
        spearmans_max.append(np.max(spearman_))
        pearsons_max.append(np.max(pearson_))

        if logger_writer:
            logger_writer(f"{file_name},RMSE:{rmse_g},spearman:{s_},spearman_var:{s_var_},spearmans_min:{s_min_},spearmans_max:{s_max_},\
                        pearson:{p_}, pearsons_var:{p_var_},pearson_min:{p_min_},pearsons_max:{p_max_}")
        print(f"{file_name},RMSE:{rmse_g},spearman:{s_},spearman_var:{s_var_},spearmans_min:{s_min_},spearmans_max:{s_max_},\
                        pearson:{p_}, pearsons_var:{p_var_},pearson_min:{p_min_},pearsons_max:{p_max_}")
    if logger_writer:
        logger_writer(f"{fep},RMSE:{np.mean(rmse_gs)}, RMSE_g:{np.mean(rmse_gs)},spearman:{np.mean(spearmans)},pearson:{np.mean(pearsons)}")

    print(f"{fep},RMSE:{np.mean(rmse_gs)}, RMSE_g:{np.mean(rmse_gs)},spearman:{np.mean(spearmans)},pearson:{np.mean(pearsons)}")
    
    return prediction_of_FEP, corr_of_FEP, np.mean(spearmans)



def test_fep_nobond(fep,logger_writer,model,device,code_path,batch_size,test_file_name_FEP,type_graph, bb=False):
    # fep: FEP1 or FEP2
    rmse_gs = []
    spearmans = []
    pearsons = []
    kendalls = []

    spearmans_var = []
    pearsons_var = []
    kendalls_var = []

    spearmans_min = []
    pearsons_min = []
    kendalls_min = []

    spearmans_max = []
    pearsons_max = []
    kendalls_max = []

    spearmans_10 = []
    pearson_10 = []


    for file_name in test_file_name_FEP:

        df_file = pd.read_csv(f"{code_path}/data/FEP/direct_input/{file_name}.csv")
        test_dataset = LeadOptDataset(f"{code_path}/data/FEP/direct_input/{file_name}.csv")
        test_dataloader = DataLoader(test_dataset,
                                          collate_fn=partial(collate_fn_fep_nobond, type_graph=type_graph),
                                          batch_size=batch_size,
                                          drop_last=False,
                                          shuffle=False,
                                          pin_memory=False)

        mae,rmse,mae_g,rmse_g,valid_prediction,valid_prediction_G,valid_labels,ref_1_label,val_2_label= predict(model, test_dataloader, device, bb)
        
        # rmsd
        rmse_gs.append(rmse_g)
  
        if file_name == test_file_name_FEP[0]:
            prediction_of_FEP = pd.DataFrame({f'prediction_ic50_{file_name}': valid_prediction,
                                               f'prediction_G_{file_name}': valid_prediction_G,
                                               f"label_ic50_{file_name}": valid_labels})
        else:
            prediction_of_FEP_ = pd.DataFrame({f'prediction_ic50_{file_name}': valid_prediction,
                                                f'prediction_G_{file_name}': valid_prediction_G,
                                                f"label_ic50_{file_name}": valid_labels})
            prediction_of_FEP = pd.merge(prediction_of_FEP, prediction_of_FEP_, how="outer",right_index=True,left_index=True)


        # cor
        abs_label = val_2_label
        abs_predict = np.array(ref_1_label).astype(float) - np.array(valid_prediction).astype(float)

        df_file["abs_label_p"] = abs_label
        df_file["abs_predict_p"] = abs_predict

        reference_num = df_file.reference_num.values
        ligand1_num = df_file.Ligand1_num.values
        _df = pd.DataFrame({"reference_num": reference_num, f"abs_label_{file_name}": abs_label, f"abs_predict_{file_name}": abs_predict, f"ligand1_num_{file_name}": ligand1_num})

        if file_name == test_file_name_FEP[0]:
            corr_of_FEP = _df.groupby(f'ligand1_num_{file_name}')[[f'abs_label_{file_name}', f'abs_predict_{file_name}']].mean().reset_index()
        else:
            corr_of_FEP_ = _df.groupby(f'ligand1_num_{file_name}')[[f'abs_label_{file_name}', f'abs_predict_{file_name}']].mean().reset_index()
            corr_of_FEP = pd.merge(corr_of_FEP, corr_of_FEP_, how="outer",right_index=True,left_index=True)

        _df_group = _df.groupby('reference_num')

        spearman_ = []
        pearson_ = []

        for _, _df_onegroup in _df_group:
            spearman = _df_onegroup[[f"abs_label_{file_name}", f"abs_predict_{file_name}"]].corr(method='spearman').iloc[0, 1]
            pearson = _df_onegroup[[f"abs_label_{file_name}", f"abs_predict_{file_name}"]].corr(method='pearson').iloc[0, 1]
            spearman_.append(spearman)
            pearson_.append(pearson)

        s_,p_ = np.mean(spearman_),np.mean(pearson_)
        spearmans.append(np.mean(spearman_))
        pearsons.append(np.mean(pearson_))

        s_var_,p_var_ = np.var(spearman_),np.var(pearson_)
        spearmans_var.append(np.var(spearman_))
        pearsons_var.append(np.var(pearson_))

        s_min_,p_min_ = np.min(spearman_),np.min(pearson_)
        spearmans_min.append(np.min(spearman_))
        pearsons_min.append(np.min(pearson_))

        s_max_,p_max_ = np.max(spearman_),np.max(pearson_)
        spearmans_max.append(np.max(spearman_))
        pearsons_max.append(np.max(pearson_))

        if logger_writer:
            logger_writer(f"{file_name},RMSE:{rmse_g},spearman:{s_},spearman_var:{s_var_},spearmans_min:{s_min_},spearmans_max:{s_max_},\
                        pearson:{p_}, pearsons_var:{p_var_},pearson_min:{p_min_},pearsons_max:{p_max_}")
        print(f"{file_name},RMSE:{rmse_g},spearman:{s_},spearman_var:{s_var_},spearmans_min:{s_min_},spearmans_max:{s_max_},\
                        pearson:{p_}, pearsons_var:{p_var_},pearson_min:{p_min_},pearsons_max:{p_max_}")
    if logger_writer:
        logger_writer(f"{fep},RMSE:{np.mean(rmse_gs)}, RMSE_g:{np.mean(rmse_gs)},spearman:{np.mean(spearmans)},pearson:{np.mean(pearsons)}")
    print(f"{fep},RMSE:{np.mean(rmse_gs)}, RMSE_g:{np.mean(rmse_gs)},spearman:{np.mean(spearmans)},pearson:{np.mean(pearsons)}")
    
    return None




