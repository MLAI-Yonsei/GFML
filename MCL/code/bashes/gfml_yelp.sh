#!/bin/bash
DATA="yelp"
GPU=4

for LR in 1e-2 1e-3 1e-4 1e-5 1e-6
do
for LAM in 1e-2 1e-1 10e-1 30e-1 50e-1
do
python main.py --gfml --dataset $DATA --gpu_id $GPU --lr $LR --lambd $LAM --multicore 0
done
done