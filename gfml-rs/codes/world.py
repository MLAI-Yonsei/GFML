'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing
import wandb

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "../light-gcn"
CODE_PATH = join(ROOT_PATH, 'codes')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'dist_test', 'ml1m', 'amazon-music', 'amazon-baby']
all_models  = ['mf', 'lgn', 'cml']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['wd'] = args.weight_decay
config['ema_on'] = args.ema_on
config['loss_mode'] = args.loss_mode
config['mass_mode'] = args.mass_mode
config['lr_decay'] = args.lr_decay
config['Early_Stopping'] = args.ES
config['infer_mode'] = args.infer_mode
config['recdim'] = args.recdim

#------ CML hy-para ------
config['features'] = args.features
config['margin'] = args.margin
config['master_learning_rate'] = args.master_learning_rate
config['clip_norm'] = args.clip_norm
config['hidden_layer_dim'] = args.hidden_layer_dim
config['dropout_rate'] = args.dropout_rate
config['feature_l2_reg'] = args.feature_l2_reg
config['feature_projection_scaling_factor'] = args.feature_projection_scaling_factor
config['use_rank_weight'] = args.use_rank_weight
config['use_cov_loss'] = args.use_cov_loss
config['cov_loss_weight'] = args.cov_loss_weight

GPU = torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu_id}' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model

# gravity hyper para & a few customs
lamb = args.lamb
loss_mode = args.loss_mode
mix_ratio = args.mix_ratio
exp_name = f'1018_{args.dataset}_{args.model}_{args.loss_mode}_{args.dist_mode}_{args.mass_mode}_lamb_{args.lamb}_emb_norm_{str(args.emb_norm)}'
emb_norm = args.emb_norm
mass_mode = args.mass_mode
ES = args.ES
lam_d = args.lam_d

ema = None

wandb.login()
wandb.init(project='GFML_Revision',
           name=f'{dataset}_{model_name}_{loss_mode}_{args.dist_mode}_{mix_ratio}',
           config=args)
wandb.config.update(args)

BEST_NDCG = 1e-10

dist_mode = args.dist_mode
if dist_mode == 'kl':
    kl = torch.nn.KLDivLoss(reduction='none')

if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")


TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
