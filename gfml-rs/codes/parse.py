'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton, bpr reg")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='ml1m',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10, 20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='mf', help='rec-model, support [mf, lgn]')
    parser.add_argument('--lamb', type=float, default=1.0, help='lambda of dist in gravity func.')
    parser.add_argument('--mix_ratio', type=float, default=0.5)
    parser.add_argument('--loss_mode', type=str, default='gravity', help='[dot, bpr, gravity, mix]')
    parser.add_argument('--dist_mode', type=str, default='l2', help='[l1, l2, kl, emd]')
    parser.add_argument('--exp_name', type=str, default='Apple', help='Name of this experiment')
    parser.add_argument('--ema_on', type=int, default=1)
    parser.add_argument('--emb_norm', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay of gravity reg')
    parser.add_argument('--lr_decay', type=float, default=1e-2, help='Weight decay of opti')
    parser.add_argument('--mass_mode', type=str, default='both', help='[user, item, both]')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--ES', type=int, default=1)
    parser.add_argument('--infer_mode', type=str, default='mat')
    parser.add_argument('--lam_d', type=float, default=0.995)

    # CML hypara
    parser.add_argument('--features', default=None)
    parser.add_argument('--margin', type=float, default=1.9)
    parser.add_argument('--master_learning_rate', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=int, default=1)
    parser.add_argument('--hidden_layer_dim', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--feature_l2_reg', type=float, default=0.1)
    parser.add_argument('--feature_projection_scaling_factor', type=float, default=0.5)
    parser.add_argument('--use_rank_weight', action='store_true')
    parser.add_argument('--use_cov_loss', action='store_true')
    parser.add_argument('--cov_loss_weight', type=float, default=1.0)
    return parser.parse_args()
