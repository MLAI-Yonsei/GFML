import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import wormhole

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0., num_tokens=0, qk_mode=None, mass_q=None, mass_k=None):
        super(TransformerEncoder, self).__init__()
        num_tokens = num_tokens
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout, num_tokens=num_tokens, qk_mode=qk_mode, mass_q=mass_q, mass_k=mass_k)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0., num_tokens=0, qk_mode=None, mass_q=None, mass_k=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

        self.qk_mode = qk_mode

        if self.qk_mode == 'vit_gra':
            self.mass_q = mass_q
            self.mass_k = mass_k

            if not self.mass_q and not mass_k:
                self.qk_mode = 'vit'

            if wormhole.mass_pos == "full":
                self.mass_q = nn.Embedding(num_tokens, 1)
                self.mass_k = nn.Embedding(num_tokens, 1)
                nn.init.uniform_(self.mass_q.weight)
                nn.init.uniform_(self.mass_k.weight)

            self.relu = nn.ReLU()
            self.gelu = nn.GELU()
            self.lamb_gra = wormhole.lamb_gra
            self.mix_on = wormhole.mix_on
            self.lamb_mix = wormhole.lamb_mix
            self.score_norm = wormhole.score_norm
            self.dist_mode = wormhole.dist_mode
            self.mix_mode = wormhole.mix_mode
            self.num_split = wormhole.num_split
            self.split_mix = wormhole.split_mix

    def qk_dot(self, q, k):
        score = torch.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d # (b,h,n,n)
        return score

    def qk_gra(self, q, k):
        dist, mass = 0, 0
        if self.dist_mode == 'l2':
            dist = torch.log1p(torch.cdist(q, k, p=2) ** 2)
        elif self.dist_mode == 'l1':
            dist = torch.log1p(torch.cdist(q, k, p=1) ** 2)

        if not self.mass_q and not self.mass_k:
            mass = self.mass_q.weight + self.mass_k.weight.T

        score = (mass - self.lamb_gra * dist) / self.sqrt_d
        return score

    def forward(self, x):
        self.b, n, f = x.size()
        q = self.q(x).view(self.b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(self.b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(self.b, n, self.head, self.feats//self.head).transpose(1,2)
        score = None

        if self.qk_mode == 'vit':
            score = self.qk_dot(q,k)

        elif (self.qk_mode == 'vit_gra') and (not self.mix_on):
            score = self.qk_gra(q, k)

        elif self.qk_mode == 'vit_gra' and self.mix_on:
            if self.mix_mode == 'split':

                q_dot_list, q_gra_list = [], []
                for i, r in enumerate(torch.tensor_split(q, self.head, dim=1)):
                    if i <= self.num_split -1:
                        q_gra_list.append(r)
                    else:
                        q_dot_list.append(r)

                q_dot = torch.cat(q_dot_list, dim=1)
                q_gra = torch.cat(q_gra_list, dim=1)

                k_dot_list, k_gra_list = [], []
                for i, s in enumerate(torch.tensor_split(k, self.head, dim=1)):
                    if i <= self.num_split -1:
                        k_gra_list.append(s)
                    else:
                        k_dot_list.append(s)
                k_dot = torch.cat(k_dot_list, dim=1)
                k_gra = torch.cat(k_gra_list, dim=1)

                if self.split_mix:
                    score_dot = self.qk_dot(q_dot, k_dot)
                    score_gra = self.qk_gra(q_gra, k_gra)
                    score_dot_mix = self.qk_dot(q_gra, k_gra)
                    score_mix = (1-self.lamb_mix) * score_dot_mix + (self.lamb_mix * score_gra)
                    score = torch.cat((score_dot, score_mix), dim=1)
                else:
                    score_dot = self.qk_dot(q_dot, k_dot)
                    score_gra = self.qk_gra(q_gra, k_gra)
                    score = torch.cat((score_dot, score_gra), dim=1)

            elif self.mix_mode == 'normal':
                score_gra = self.qk_gra(q, k)
                score_dot = self.qk_dot(q, k)
                score = (1-self.lamb_mix) * score_dot + (self.lamb_mix * score_gra)

        score = F.softmax(score, dim=-1)
        attn = torch.einsum("bhij, bhjf->bihf", score, v)   #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))

        return o

class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):
        
        ...

if __name__=="__main__":
    b,n,f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)