import torch
import numpy
import functools

import torch.nn.functional as F

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


class Multiply(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        x = torch.mul(x, self.alpha)
        return x

class CML(torch.nn.Module):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 features=None,
                 margin=1.5,
                 master_learning_rate=0.1,
                 clip_norm=1.0,
                 hidden_layer_dim=128,
                 dropout_rate=0.2,
                 feature_l2_reg=0.1,
                 feature_projection_scaling_factor=0.5,
                 use_rank_weight=True,
                 use_cov_loss=True,
                 cov_loss_weight=0.1
                 ):
        super(CML, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.clip_norm = clip_norm
        self.margin = margin
        if features is not None:
            self.features = features
        else:
            self.features = None

        self.master_learning_rate = master_learning_rate
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.feature_l2_reg = feature_l2_reg
        self.feature_projection_scaling_factor = feature_projection_scaling_factor
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight

        self.user_positive_items_pairs = None
        self.negative_samples = None
        self.score_user_ids = None

        self.user_embeddings = torch.nn.Embedding(self.n_users, self.embed_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_items, self.embed_dim)

        self.feat_extract_model = torch.nn.Sequential(
            torch.nn.Linear(self.features, self.hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_rate),
            torch.nn.Linear(self.hidden_layer_dim, self.embed_dim),
        )

    def feature_loss(self):
        loss = 0.0
        if self.features is not None:
            self.feat_extarc_model.add_module(name='feature_projection',
                                              module=Multiply(self.feature_projection_scaling_factor))
            self.feature_projection = self.feat_extract_model(self.features)

            feature_distance = torch.sum(F.mse_loss(self.item_embeddings, self.feature_projection, reduction='none'),
                                         dim=1)

            loss += torch.sum(feature_distance) * self.feature_l2_reg

        return loss

    def covariance_loss(self):
        X = torch.concat((self.item_embeddings, self.user_embeddings), dim=0)
        n_rows = X.shape[0].to(torch.float32)
        X = X - torch.mean(X, dim=0)
        cov = torch.matmul(X, X.T) / n_rows

        return cov.fill_diagonal_(0).sum() * self.cov_loss_weight

    def clip_by_norm_op(self):
        pass

    def emb_distance(self, user_emb, item_emb):
        return torch.sum(F.mse_loss(user_emb, item_emb, reduction='none'), dim=1)

    def neg_emb_dist(self, user_emb, item_emb):
        tensor_range = range(len(user_emb))
        diff = user_emb.unsqueeze(dim=1) - item_emb.unsqueeze(dim=1)
        sq_diff = diff.pow(2)
        sum_sq_diff = sq_diff.sum(-1)[tensor_range, tensor_range, :]
        return sum_sq_diff

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.user_embeddings(users)
        items_emb = self.item_embeddings(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)


    def loss(self, user, pos, neg):
        loss_value = 0.0
        user_emb = self.user_embeddings(user) # (B, D)
        pos_emb = self.item_embeddings(pos) # (B, D)
        neg_emb = self.item_embeddings(neg) # (B, N, D)

        pos_dist = self.emb_distance(user_emb, pos_emb) # (B, )
        neg_dist = self.neg_emb_distance(user_emb, neg_emb) # (B, N)
        min_neg_dist = torch.min(neg_dist, dim=1).values

        loss_per_pair = torch.maximum(pos_dist - min_neg_dist + self.margin, torch.tensor([0.]))

        if self.use_rank_weight:
            imposters = (pos_dist.view((-1,1)) - neg_dist + self.margin) > 0
            rank = imposters.to(torch.float32).mean(1) * self.n_items
            loss_per_pair *= torch.log(rank+1)

        self.loss_e = torch.sum(loss_per_pair)
        self.loss_f = self.feature_loss()

        loss_value = self.loss_e + self.loss_f
        if self.use_cov_loss:
            loss_value += self.covariance_loss()

        return loss_value














