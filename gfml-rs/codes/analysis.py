import numpy as np
import torch
import world
import register
from register import dataset
from model import PureMF, LightGCN
from parse import parse_args
args = parse_args()

PATH = '../results/exp_name_1018_amazon-music_mf_gravity_l2_both_lamb_3.9550444822926103_emb_norm_0_ndcg_0.1279.pt'

# exp_name_1018_amazon-music_mf_mix_bpr_l1_both_lamb_3.959706837325762_emb_norm_1_ndcg_0.1352_mix_05.pt - mix_bpr
# ../results/exp_name_1018_amazon-music_mf_bpr_l1_both_lamb_3.959706837325762_emb_norm_1_ndcg_0.1341.pt - bpr
# ../results_arxiv/exp_name_1018_amazon-music_mf_dot_l2_both_lamb_1.9573556583343616_emb_norm_1_ndcg_0.1038.pt - dot
# exp_name_1018_amazon-music_mf_mix_l2_both_lamb_3.994534262637375_emb_norm_1_ndcg_0.1235.pt - mix
# exp_name_1018_amazon-music_mf_gravity_l2_both_lamb_3.9550444822926103_emb_norm_0_ndcg_0.1279.pt -gravity
recmodel = register.MODELS['mf'](world.config, dataset)
recmodel.load_state_dict(torch.load(PATH))

# mass_u_vector = recmodel.mass_u.weight.detach().numpy()
# mass_i_vector = recmodel.mass_i.weight.detach().numpy()
embed_u = recmodel.embedding_user.weight.detach().numpy()
embed_i = recmodel.embedding_item.weight.detach().numpy()

# np.save('../matrix/mass_u_baby_mf_bpr_last', mass_u_vector)
# np.save('../matrix/mass_i_baby_mf_bpr_last', mass_i_vector)
torch.save(embed_u, '../matrix/embed_u_baby_mf_bpr_last.pt')
torch.save(embed_i, '../matrix/embed_i_baby_mf_bpr_last.pt')

# PATH = '../results/exp_name_1018_ml1m_mf_mix_bpr_l2_both_lamb_3.0824860422368356_emb_norm_1_ndcg_0.3877.pt'

# recmodel = register.MODELS['mf'](world.config, dataset)
# recmodel.load_state_dict(torch.load(PATH))

# mass_u_vector = recmodel.mass_u.weight.detach().numpy()
# mass_i_vector = recmodel.mass_i.weight.detach().numpy()
# embed_u = recmodel.embedding_user.weight.detach().numpy()
# embed_i = recmodel.embedding_item.weight.detach().numpy()

# np.save('../matrix/mass_u_ml1m_mf_mix_bpr_last', mass_u_vector)
# np.save('../matrix/mass_i_ml1m_mf_mix_bpr_last', mass_i_vector)
# torch.save(embed_u, '../matrix/embed_u_ml1m_mix_bpr_last.pt')
# torch.save(embed_i, '../matrix/embed_i_ml1m_mix_bpr_last.pt')