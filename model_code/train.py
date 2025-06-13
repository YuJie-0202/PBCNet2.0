import argparse
import os
import random
import time
import sys
code_path =  os.path.dirname(os.path.abspath(__file__))    # /home/user-home/yujie/0_PBCNetv2/0_PBCNET/model_code/train.py/
sys.path.append(code_path)
code_path = code_path.rsplit("/", 1)[0]

import numpy as np
import pandas as pd
import torch
from Dataloader.dataloader import collate_fn, LeadOptDataset
from torch.utils.data import DataLoader
from models.readout import PBCNetv2
from predict.predict import predict, test_fep, test_fep_nobond
from utilis.function import get_loss_func
from utilis.scheduler import NoamLR_shan
from utilis.trick import Writer
from utilis.utilis import param_count
from utilis.initial import initialize_weights


def softmax(x):
    x_exp = np.exp(x)
    # colunm axis=0
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

def train(args,
          model, 
          train_loader, 
          device, 
          num_training_samples):

    train_start = time.time()
    # learning rate decay and optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.000001)
    epoch_step = len(train_loader)
    scheduler = NoamLR_shan(opt,
                            warmup_epochs=[1],
                            decay_epochs=[2],
                            final_epochs=[3],
                            steps_per_epoch=epoch_step,
                            init_lr=[args.init_lr],
                            max_lr=[args.max_lr],
                            final_lr=[args.final_lr])

    # training loss
    loss_func = get_loss_func(args.loss_function)

    # for computing training mae and rmse
    mae_func = torch.nn.L1Loss(reduction='sum')
    mse_func = torch.nn.MSELoss(reduction='sum')

    # for log
    save_dir = os.path.join(code_path, "result")
    logger_writer = Writer(os.path.join(save_dir, "record_1.txt"))
    # for log in a batch defined step
    batch_all = 0
    loss_for_log = 0

    model.train()
    for epoch in range(3):
        loss_mse = []
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
            label_neg = torch.neg(label)
            logits,logits_neg = model(graph1,graph2) 
            loss = loss_func(logits.squeeze().float(), label.float()) + loss_func(logits_neg.squeeze().float(), label_neg.float())

            train_mae_ = mae_func(logits.squeeze().float(), label.float())
            train_mse_ = mse_func(logits.squeeze().float(), label.float())
            loss_mse.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            batch_all += 1
            loss_for_log += loss.item()
            training_mae += train_mae_.item()
            training_mse += train_mse_.item()

            if batch_all % args.log_frequency == 0:
                _loss = loss_for_log / args.log_frequency  # mean loss for each batch with size of log_frequency.
                print(f"Epoch {epoch}  Batch {batch_all}  Loss {_loss}")
                logger_writer(f"Epoch {epoch}  Batch {batch_all}  Loss {_loss}")
                print(f"Epoch {epoch}  Batch {batch_all}  Learning rate {scheduler.get_lr()[0]}")
                logger_writer(f"Epoch {epoch}  Batch {batch_all}  Learning rate {scheduler.get_lr()[0]}")
                loss_for_log = 0

            if batch_all % 5 * args.log_frequency == 0:
                rmsd_file_fep1, corr_file_fep1, s_fep1 = test_fep('FEP1',logger_writer,model,device,code_path,args.batch_size)
                rmsd_file_fep2, corr_file_fep2, s_fep2 = test_fep('FEP2',logger_writer,model,device,code_path,args.batch_size)

            # stop training
            if batch_all % 15000 == 0:
                train_time = time.time()

                training_rmse = (training_mse / num_training_samples) ** 0.5
                training_mae = training_mae / num_training_samples

                logger_writer("  ")
                logger_writer(f"Epoch {epoch}_{batch_all}")
                logger_writer(f"Training time {train_time - start}")
                logger_writer(f"Training Set mae {training_mae}")
                logger_writer(f"Training Set rmse {training_rmse}")

                test_time = time.time()
                logger_writer(f"test time {test_time - train_time}")
                Path = os.path.join(save_dir, f"model_{epoch}_{batch_all}_{args.seed}.pth")
                torch.save(model, Path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument('--log_frequency', type=int, default=1, help='Number of batches reporting performance once' )
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--loss_function', type=str, default="mse")
    parser.add_argument("--device", type=int, default=0,help="The number of device")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2,help='the random seed' )
    parser.add_argument('--cpu_num', type=int, default=4,help='the number of cpu')
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.0001)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--final_lr', type=float, default=0.0001)
    parser.add_argument('--retrain', type=int, default=0,help='Whether to continue training with an incomplete training model (Finetune)')
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--fold', type=str,default="0.1")

    args = parser.parse_args()
    setup_cpu(args.cpu_num)
    setup_seed(args.seed)
    cuda = "cuda:" + str(args.device)
    # cuda = 'cpu'

    # define model
    if args.retrain == 1:
        model = torch.load('/code/PBCNet.pth')
        model.to(cuda)
    else:
        model = PBCNetv2(hidden_channels = args.hidden_dim,
                         num_layer = args.radius,
                         num_rbf = 32,
                         max_z = 128,
                         equivariance_invariance_group = "O(3)",
                         activation = 'silu',
                         dtype=torch.float32).to(cuda)
    para_num = param_count(model)
    print(para_num)

    train_dataset = LeadOptDataset(args.train_path)
    num_training_samples = len(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                  collate_fn=collate_fn, 
                                  batch_size=args.batch_size,
                                  drop_last=False, 
                                  shuffle=True,
                                  num_workers=args.num_workers, 
                                  pin_memory=False)
    
    train(args, 
          model, 
          train_dataloader,
          cuda, 
          num_training_samples)
