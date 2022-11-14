"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

        self.act = torch.nn.Sigmoid()
        self.bce = torch.nn.BCELoss()

    def getUsersRating(self, users):
        raise NotImplementedError

    def l2_dist(self, x, y):
        mat = x-y
        dist = torch.log(torch.sum((mat * mat), dim=1) + 0.01)
        dist = self.lam * dist
        return dist

    def l1_dist(self, x, y):
        dist = torch.log(torch.sum(torch.abs(x - y), dim=-1) ** 2 + 0.01)
        # dist = torch.nan_to_num(dist, nan=0.0, neginf=-1e-3)
        dist = self.lam * dist
        return dist

    def kl_dist(self, x, y):
        x = self.act(x)
        y = self.act(y)
        dist = world.kl(x, y)
        dist = torch.mean(dist, dim=-1) ** 2
        dist = torch.log(dist + 0.01)
        # dist = torch.nan_to_num(dist, nan=0.0, neginf=-1e-3)
        return self.lam * dist

    def emd(self, x, y):
        dist = torch.mean(torch.square(torch.cumsum(x, dim=-1) - torch.cumsum(y, dim=-1)), dim=-1)
        dist = torch.log(dist ** 2 + 0.01)
        # dist = torch.nan_to_num(dist, nan=0.0, neginf=-1e-3)
        return self.lam * dist

    def gravity(self, u_emb, i_emb, item_idx):
        if world.emb_norm:
            u_emb = torch.nn.functional.normalize(u_emb)
            i_emb = torch.nn.functional.normalize(i_emb)
        if world.mass_mode == "both":
            users, items = item_idx[0].long(), item_idx[1].long()
            mass = torch.log(self.relu(self.mass_u(users)) + 1) * torch.log(self.relu(self.mass_i(items)) + 1)
            if world.dist_mode == 'l2':
                dist = self.l2_dist(u_emb, i_emb)
                g = mass - dist.view(-1, 1)
            elif world.dist_mode == 'l1':
                dist = self.l1_dist(u_emb, i_emb)
                g = mass - dist.view(-1, 1)
            elif world.dist_mode == 'kl':
                dist = self.kl_dist(u_emb, i_emb)
                g = mass - dist.view(-1, 1)
            elif world.dist_mode == 'emd':
                dist = self.emd(u_emb, i_emb)
                g = mass - dist.view(-1, 1)
        else:
            mass = torch.log(self.relu(self.mass(item_idx.long())) + 1)
            if world.dist_mode == 'l2':
                dist = self.l2_dist(u_emb, i_emb)
                g = mass(item_idx.long()) - dist.view(-1, 1)
            elif world.dist_mode == 'l1':
                dist = self.l1_dist(u_emb, i_emb)
                g = mass(item_idx.long()) - dist.view(-1, 1)
            elif world.dist_mode == 'kl':
                dist = self.kl_dist(u_emb, i_emb)
                g = mass(item_idx.long()) - dist.view(-1, 1)
            elif world.dist_mode == 'emd':
                dist = self.emd(u_emb, i_emb)
                g = mass(item_idx.long()) - dist.view(-1, 1)
        if world.loss_mode == 'brp_gra':
            return g
        return self.act(g)

    def gravity_loss_func(self, users, pos, neg, users_emb, pos_emb, neg_emb):
        if self.mass_mode == 'item':
            mass_pos = self.mass(pos.long())
            mass_neg = self.mass(neg.long())
            pos_scores = self.gravity(users_emb, pos_emb, pos)
            neg_scores = self.gravity(users_emb, neg_emb, neg)
            reg_loss = (1 / 2) * (mass_pos.norm(2).pow(2) + mass_neg.norm(2).pow(2) / float(len(users)))
        elif self.mass_mode == 'user':
            mass_user = self.mass(users.long())
            pos_scores = self.gravity(users_emb, pos_emb, users)
            neg_scores = self.gravity(users_emb, neg_emb, users)
            reg_loss = (1 / 2) * (mass_user.norm(2).pow(2) / float(len(users)))
        elif self.mass_mode == 'both':
            mass_pos = self.mass_i(pos.long())
            mass_neg = self.mass_i(neg.long())
            mass_user = self.mass_u(users.long())
            pos_scores = self.gravity(users_emb, pos_emb, (users, pos))
            neg_scores = self.gravity(users_emb, neg_emb, (users, neg))
            reg_loss = (1/2) * (mass_user.norm(2).pow(2) +
                                mass_pos.norm(2).pow(2) +
                                mass_neg.norm(2).pow(2) / float(len(users)))
        return pos_scores, neg_scores, reg_loss

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset,):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.mass_mode = config['mass_mode']
        self.decay = config['decay']
        self.wd = config['wd']

        self.lam = world.lamb
        if world.loss_mode in ['gravity', 'mix', 'bpr_gra', 'mix_bpr']:
            if config['mass_mode'] == 'user':
                self.mass = nn.Embedding(self.num_users, 1)
                nn.init.xavier_normal_(self.mass.weight)
            elif config['mass_mode'] == 'item':
                self.mass = nn.Embedding(self.num_items, 1)
                nn.init.xavier_normal_(self.mass.weight)
            elif config['mass_mode'] == 'both':
                self.mass_u = nn.Embedding(self.num_users, 1)
                self.mass_i = nn.Embedding(self.num_items, 1)
                nn.init.xavier_normal_(self.mass_u.weight)
                nn.init.xavier_normal_(self.mass_i.weight)

        self.act_fun = nn.ELU()
        self.loss_mode = world.loss_mode
        self.mix_ratio = world.mix_ratio
        self.relu = nn.ReLU()

        print(f"{world.model_name} : {world.loss_mode}_act_fun_elu_lambda_{self.lam}")
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        if world.loss_mode in ['gravity']:
            if world.dist_mode == 'l2':
                users = users.long()
                user_emb = self.embedding_user(users)
                items_emb = self.embedding_item.weight
                user_emb = user_emb.view(user_emb.shape[0], -1, user_emb.shape[1]).expand(user_emb.shape[0],
                                                                                          items_emb.shape[0],
                                                                                          user_emb.shape[1])
                mat = user_emb - items_emb
                dist = self.lam * torch.log(torch.sum((mat * mat), dim=-1) + 0.01)

            elif world.dist_mode == 'emd':
                users = users.long()
                user_emb = self.embedding_user(users)
                items_emb = self.embedding_item.weight
                user_emb = user_emb.view(user_emb.shape[0], -1, user_emb.shape[1]).expand(user_emb.shape[0],
                                                                                          items_emb.shape[0],
                                                                                          user_emb.shape[1])

                dist = torch.mean(torch.square(torch.cumsum(user_emb, dim=-1) - torch.cumsum(items_emb, dim=-1)), dim=-1)
                dist = self.lam * torch.log(dist ** 2 + 0.01)

            elif world.dist_mode == 'l1':
                users = users.long()
                user_emb = self.embedding_user(users)
                items_emb = self.embedding_item.weight
                user_emb = user_emb.view(user_emb.shape[0], -1, user_emb.shape[1]).expand(user_emb.shape[0],
                                                                                         items_emb.shape[0],
                                                                                         user_emb.shape[1])
                dist = self.lam * torch.log(torch.sum(torch.abs(user_emb - items_emb), dim=-1) ** 2 + 0.01)

            elif world.dist_mode == 'kl':
                users = users.long()
                user_emb = self.embedding_user(users)
                items_emb = self.embedding_item.weight
                user_emb = user_emb.view(user_emb.shape[0], -1, user_emb.shape[1]).expand(user_emb.shape[0],
                                                                                          items_emb.shape[0],
                                                                                          user_emb.shape[1])
                dist = self.lam * torch.log(torch.mean(world.kl(user_emb, items_emb), dim=-1) ** 2 + 0.01)

            if world.mass_mode == 'both':
                mass_u = self.mass_u(users.long()).expand(len(users), self.num_items)
                mass = torch.log(self.relu(mass_u) + 1) * torch.log(self.relu(self.mass_i.weight.t()) + 1)
                scores = mass - dist
            else:
                scores = torch.log(self.relu(self.mass.weight.t())+1) - dist
        else:
            users = users.long()
            users_emb = self.embedding_user(users)
            items_emb = self.embedding_item.weight
            scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())

        if self.loss_mode == 'dot':
            pos_scores = self.act(torch.sum(users_emb*pos_emb, dim=1))
            neg_scores = self.act(torch.sum(users_emb*neg_emb, dim=1))
            pred = torch.cat((pos_scores, neg_scores), dim=0)
            target = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), dim=0)

            loss = self.bce(pred, target)
            return loss

        elif self.loss_mode == 'gravity':
            if self.mass_mode == 'item':
                mass_pos = self.mass(pos.long())
                mass_neg = self.mass(neg.long())
                pos_scores = self.gravity(users_emb, pos_emb, pos)
                neg_scores = self.gravity(users_emb, neg_emb, neg)
                reg_loss = (1 / 2) * (mass_pos.norm(2).pow(2) + mass_neg.norm(2).pow(2) / float(len(users)))
            elif self.mass_mode == 'user':
                mass_user = self.mass(users.long())
                pos_scores = self.gravity(users_emb, pos_emb, users)
                neg_scores = self.gravity(users_emb, neg_emb, users)
                reg_loss = (1 / 2) * (mass_user.norm(2).pow(2) / float(len(users)))
            elif self.mass_mode == 'both':
                mass_pos = self.mass_i(pos.long())
                mass_neg = self.mass_i(neg.long())
                mass_user = self.mass_u(users.long())
                pos_scores = self.gravity(users_emb, pos_emb, (users, pos))
                neg_scores = self.gravity(users_emb, neg_emb, (users, neg))
                reg_loss = (1/2) * (mass_user.norm(2).pow(2) +
                                    mass_pos.norm(2).pow(2) +
                                    mass_neg.norm(2).pow(2) / float(len(users)))

            pred = torch.cat((pos_scores, neg_scores), dim=0)
            target = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), dim=0)

            loss = self.bce(pred, target) + (self.wd * reg_loss)
            return loss

        elif self.loss_mode == 'bpr':
            pos_scores = torch.sum(users_emb*pos_emb, dim=1)
            neg_scores = torch.sum(users_emb*neg_emb, dim=1)

            reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                                  pos_emb.norm(2).pow(2) +
                                  neg_emb.norm(2).pow(2)) / float(len(users))
            loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
            return loss, reg_loss

        # elif self.loss_mode == 'mix':
        #     pos_scores_d = torch.sum(users_emb * pos_emb, dim=1)
        #     neg_scores_d = torch.sum(users_emb * neg_emb, dim=1)
        #     pos_scores_g = self.gravity(users_emb, pos_emb)
        #     neg_scores_g = self.gravity(users_emb, neg_emb)
        #
        #     pos_scores = pos_scores_d + self.mix_ratio*pos_scores_g
        #     neg_scores = neg_scores_d + self.mix_ratio*neg_scores_g
        #
        #     reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
        #                           pos_emb.norm(2).pow(2) +
        #                           neg_emb.norm(2).pow(2)) / float(len(users))
        elif self.loss_mode == 'mix':
            pos_scores_dot = self.act(torch.sum(users_emb * pos_emb, dim=1))
            neg_scores_dot = self.act(torch.sum(users_emb * neg_emb, dim=1))
            pred_dot = torch.cat((pos_scores_dot, neg_scores_dot), dim=0)

            pos_scores_gra, neg_scores_gra, reg_loss_gra = self.gravity_loss_func(users, pos, neg, users_emb, pos_emb,
                                                                                  neg_emb)
            pos_scores_gra = self.act(pos_scores_gra)
            neg_scores_gra = self.act(neg_scores_gra)
            pred_gra = torch.cat((pos_scores_gra, neg_scores_gra), dim=0).squeeze()

            target = torch.cat((torch.ones_like(pos_scores_dot), torch.zeros_like(neg_scores_dot)), dim=0)
            loss_dot = self.bce(pred_dot, target)
            loss_gra = self.bce(pred_gra, target)

            loss = ((1-self.mix_ratio)*loss_dot + self.mix_ratio * loss_gra) + (self.wd * reg_loss_gra)
            return loss

        elif self.loss_mode == 'bpr_gra':
            pos_scores, neg_scores, reg_loss_mass = self.gravity_loss_func(users, pos, neg, users_emb, pos_emb, neg_emb)
            reg_loss_bpr = (1 / 2) * (
                        users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / float(len(users))
            reg_loss = (1/2) * (self.wd * reg_loss_mass + self.decay * reg_loss_bpr)
            loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
            loss += reg_loss
            return loss

        elif self.loss_mode == "mix_bpr":
            pos_scores_bpr = torch.sum(users_emb * pos_emb, dim=1)
            neg_scores_bpr = torch.sum(users_emb * neg_emb, dim=1)

            reg_loss_bpr = (1 / 2) * (users_emb.norm(2).pow(2) +
                                  pos_emb.norm(2).pow(2) +
                                  neg_emb.norm(2).pow(2)) / float(len(users))
            loss_bpr = torch.mean(nn.functional.softplus(neg_scores_bpr - pos_scores_bpr))

            loss_bpr = loss_bpr + (self.decay * reg_loss_bpr)

            pos_scores_gra, neg_scores_gra, reg_loss_gra = self.gravity_loss_func(users, pos, neg, users_emb, pos_emb,
                                                                                  neg_emb)

            pos_scores_gra = self.act(pos_scores_gra).squeeze()
            neg_scores_gra = self.act(neg_scores_gra).squeeze()
            pred_gra = torch.cat((pos_scores_gra, neg_scores_gra), dim=0).squeeze()

            target = torch.cat((torch.ones_like(pos_scores_gra), torch.zeros_like(neg_scores_gra)), dim=0)
            loss_gra = self.bce(pred_gra, target) + (self.wd * reg_loss_gra)

            loss = ((1-self.mix_ratio) * loss_bpr) + (self.mix_ratio * loss_gra)
            return loss


    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self,
                 config:dict,
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : BasicDataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items

        self.lam = world.lamb
        self.mass = nn.Embedding(self.num_users, 1)
        self.act_fun = nn.ELU()
        self.loss_mode = world.loss_mode
        self.mix_ratio = world.mix_ratio
        self.relu = nn.ReLU()

        print(f"{world.model_name} : {world.loss_mode}_act_fun_elu_lambda_{self.lam}")
        self.__init_weight()

    def __init_weight(self):
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
        nn.init.xavier_uniform_(self.mass.weight)

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        self.users = users
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        if self.loss_mode == 'bpr':
            pos_scores = torch.sum(users_emb * pos_emb, dim=1)
            neg_scores = torch.sum(users_emb * neg_emb, dim=1)

            reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                                  posEmb0.norm(2).pow(2) +
                                  negEmb0.norm(2).pow(2)) / float(len(users))

        elif self.loss_mode == 'gravity':
            pos_scores = self.gravity(users_emb, pos_emb)
            neg_scores = self.gravity(users_emb, neg_emb)

            reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                                  posEmb0.norm(2).pow(2) +
                                  negEmb0.norm(2).pow(2)) / float(len(users))

        elif self.loss_mode == 'mix':
            pos_scores_d = torch.sum(users_emb * pos_emb, dim=1)
            neg_scores_d = torch.sum(users_emb * neg_emb, dim=1)
            pos_scores_g = self.gravity(users_emb, pos_emb)
            neg_scores_g = self.gravity(users_emb, neg_emb)

            pos_scores = pos_scores_d + self.mix_ratio * pos_scores_g
            neg_scores = neg_scores_d + self.mix_ratio * neg_scores_g

            reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                                  posEmb0.norm(2).pow(2) +
                                  negEmb0.norm(2).pow(2)) / float(len(users))

        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma