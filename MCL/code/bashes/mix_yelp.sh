#!/bin/bash
DATA="yelp"
GPU=4

for LR in 1e-3 1e-4 1e-2
do
for LAM in 50e-1 10e-1 1e-2
do
for lamP in 1.0 1.2 1.5
do
for lamN in 0.5 0.9 1.0
do
for MIX in 0.7 0.5 0.3
do
python main.py --mix --mix_ratio $MIX --dataset $DATA --gpu_id $GPU --lr $LR --lambd $LAM --lamb_p $lamP --lamb_n $lamN --lambp 8 --lambn -1 --multicore 0
done
done
done
done
done