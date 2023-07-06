#!/bin/bash
DATA="amazon-digital-music"
GPU=1

for LR in 1e-2 1e-4 1e-6
do
for LAM in 1e-2 10e-1 50e-1
do
for lamP in 1.5 1.2 1.0
do
for lamN in 1.0 0.9 0.5
do
python main.py --gfml --dataset $DATA --gpu_id $GPU --lr $LR --lambd $LAM --lamb_p $lamP --lamb_n $lamN
done
done
done
done