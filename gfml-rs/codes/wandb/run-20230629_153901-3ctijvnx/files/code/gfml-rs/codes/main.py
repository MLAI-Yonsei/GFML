import world
import utils
from world import cprint
import torch
import numpy as np
# from tensorboardX import SummaryWriter
import time
import Procedure
from torch_ema import ExponentialMovingAverage

from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

try:
    if world.mass_mode in ["user", "item"]:
        world.ema = ExponentialMovingAverage(Recmodel.get_submodule('mass').parameters(), decay=world.lam_d)
    elif world.mass_mode == "both":
        world.ema = ExponentialMovingAverage(list(Recmodel.get_submodule('mass_u').parameters()) +
                                             list(Recmodel.get_submodule('mass_i').parameters()), decay=world.lam_d)
except:
    pass

bpr = utils.BPRLoss(Recmodel, world.config)

print(f"**********************************")
print(Recmodel)
print(f"This model is {world.model_name}")

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
with open(f"../results/exp_{world.exp_name}_HyPara.txt", "a") as f:
    print(f"{world.args}", file=f)
# if world.LOAD:
#     try:
#         Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
#         world.cprint(f"loaded model weights from {weight_file}")
#     except FileNotFoundError:
#         print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
# if world.tensorboard:
#     w : SummaryWriter = SummaryWriter(
#                                     join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
#                                     )
# else:
#     w = None
#     world.cprint("not enable tensorflowboard")
world.wandb.watch(Recmodel)
w = None
try:
    # Early stopping
    last_score = -1e-1
    patienc = 30
    stop_count = 0

    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            current_ndcg = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

            if world.ES:
                if current_ndcg < last_score:
                    stop_count += 1
                    print(f"STOP_COUNT : [{stop_count}|{patienc}]")
                    if stop_count >= patienc:
                        print(f"Ealy stop!!!")
                        break
                elif current_ndcg >= last_score:
                    last_score = current_ndcg
                    stop_count = 0

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        # torch.save(Recmodel.state_dict(), weight_file)
    world.wandb.finish()
finally:
    if world.tensorboard:
        w.close()