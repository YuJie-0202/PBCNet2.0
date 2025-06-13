#!/bin/bash

script_dir=$(cd $(dirname $0);pwd)
dir=$(dirname $script_dir)

loss_function="mse"
fep="fep1"
device=1
batch_size=128
seed=1

python $script_dir/Finetune.py \
--device ${device} \
--batch_size ${batch_size} \
--loss_function ${loss_function} \
--seed ${seed} \
--which_fep ${fep}
