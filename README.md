# GFML : Gravity Function for Metric Learning
**Official GitHub Repository:**
This repository provides the code implementation for the EAAI journal paper "GFML: Gravity Function for Metric Learning". You can explore and reproduce the results for various tasks, including recommender systems and vision tasks.

---

**Abstract:**
Diverse machine learning algorithms rely on the distance metric to compare and aggregate the information. A metric learning algorithm that captures the relevance between two vectors plays a critical role in machine learning. Metric learning may become biased toward the major classes and not be robust to the minor ones, i.e., metric learning may be vulnerable in an imbalanced dataset. We propose a gravity function-based metric learning (GFML) that captures the relationship between vectors based on the gravity function. We formulate GFML with two terms: 1) mass of the given vectors and 2) distance between the query and key vector. Mass learns the importance of the object itself, enabling robust metric learning on imbalanced datasets. GFML is simple and scalable; therefore, it can be adopted in diverse tasks. We validate that GFML improves the recommender system and image classification.

**Recommender System Task:**
To run the recommender system task, use the following command:
```bash
python gfml-rs/codes/main.py
```

**MCL Baseline Task:**
You can run the MCL baseline task with the GFML integration as follows:
```bash
python MCL/code/main.py --gfml --dataset $DATA --gpu_id $GPU --lr $LR --lambd $LAM --lamb_p $lamP --lamb_n $lamN
```
Additionally, you can perform parameter sweeps using the scripts available in `MCL/code/bashes`.

**Vision Task:**
To execute a vision task, such as training a vision model, you can use the following command:
```bash
python gfml-vis/main.py --model-name vit_gra --max-epochs 200 --batch-size 128 --weight-decay 5e-5 --lr 1e-3 --min-lr 1e-5 --warmup-epoch 5 --dropout 0.0 --head 12 --num-layers 7 --hidden 384 --mlp-hidden 384 --autoaugment --label-smoothing --lamb_gra 3.0 --lamb_mix 0.5 --mix_on 1 --mix_mode split --dist_mode l2 --mass_pos tail --num_split 6 --split_mix 1 --dataset cimb10 --num-classes 10 --project-name 221104-cimb10
``` 

This repository offers complete support to experiment and validate GFML across different domains.

---
- gfml-rs : These codes are based on https://github.com/gusye1234/LightGCN-PyTorch .
- gfml-vis : These codes are based on https://github.com/omihub777/ViT-CIFAR .
- MCL : This These codes are based on https://github.com/layer6ai-labs/MCL . 
