#!/bin/bash
DATA="amazon-digital-music"
GPU=0

for LR in 1e-2
do
for LAM in 1e-2
do
for lamP in 2.0 1.7 1.5 1.3 1.0
do
for lamN in 1.0 0.9 0.8 0.7 0.6
do
for MIX in 0.7
do
python main.py --mix --mix_ratio $MIX --dataset $DATA --gpu_id $GPU --lr $LR --lambd $LAM --lamb_p $lamP --lamb_n $lamN --lambp 6.5 --lambn -0.5
done
done
done
done
done