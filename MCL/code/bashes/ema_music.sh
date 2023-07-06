#!/bin/bash
DATA="amazon-digital-music"
GPU=1

for ED in 0.995 0.999 0.990
do
python main.py --mix --mix_ratio 0.5 --dataset amazon-digital-music --gpu_id 0 --lr 1e-4 --lambd 10e-1 --lamb_p 1.5 --lamb_n 1.0 --lambp 6.5 --lambn -0.5 --ema --ema_deca $ED
y
done