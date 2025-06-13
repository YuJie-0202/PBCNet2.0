#!/bin/bash

script_dir=$(cd $(dirname $0);pwd)
dir=$(dirname $script_dir)

device=1
batch_size=128
loss_function="mse"
hidden_dim=96
train_path=$dir"/data/training_clip_862W.csv" 

log_frequnency=500
init_lr=0.000001
max_lr=0.000001
final_lr=0.000001
fold="mse_lr1e-6_862w_attwithatom"

seed=1
radius=3 
retrain=0

python $script_dir/train.py \
--device ${device} \
--batch_size ${batch_size} \
--loss_function ${loss_function} \
--hidden_dim  ${hidden_dim} \
--train_path ${train_path} \
--log_frequency ${log_frequnency} \
--max_lr ${max_lr} \
--init_lr ${init_lr} \
--final_lr ${final_lr} \
--retrain ${retrain} \
--fold ${fold} \
--seed ${seed} \
--radius ${radius} \