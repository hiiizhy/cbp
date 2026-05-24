import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, config, numUsers, numItems):
        super(Model, self).__init__()

        # self
        self.config = config
        self.dim = self.config.dim

        self.h1 = self.config.h1
        self.h2 = self.config.h2

        self.alpha = self.config.alpha
        self.beta = self.config.beta

        self.nnrelu = nn.Sequential(
            nn.ReLU(),
            nn.ReLU(),
        )

        self.nitems = numItems
        self.nusers = numUsers

        self.layer_norm = nn.LayerNorm(self.dim, eps = 1e-5)

        self.userEmb = nn.Embedding(numUsers + 1, self.dim)
        self.itemEmb = nn.Embedding(numItems + 1, self.dim, padding_idx=config.padIdx)


        self.out = nn.Linear(self.dim, numItems)
        self.his_embds = nn.Linear(numItems, self.dim)
        self.gate_his  = nn.Linear(self.dim, 1)
        self.gate_trans= nn.Linear(self.dim, 1)

        self.asp_d_h1 = nn.Linear(self.dim, self.h1)
        self.asp_h1_h2 = nn.Linear(self.h1, self.h2)

        self.rnn = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.dim,
            # bidirectional=True,
            dropout=0.5,
            num_layers=2,
            batch_first=True,
        )
        self.rnn.flatten_parameters()
        self.dropout = nn.Dropout(0.5)  # self.drop_ratio


    def forward(self, user, seq, decay, offset, uhis, ihis, his, userPOP, itemPOP, device):
        batch = seq.shape[0]  # batch

        embs = self.itemEmb(seq)

        embs3d = decay * embs.sum(2)
        embs2d = torch.tanh(embs3d.sum(1))

        scores_trans = self.out(embs2d)
        scores_trans = F.softmax(scores_trans, dim=-1)

        itemPOP_his = self.his_embds(itemPOP.repeat(batch,1))

        Iembs = torch.tanh(self.itemEmb(seq).sum(2).sum(1))
        itemBias = self.out(Iembs * itemPOP_his)
        itemBias = F.softmax(itemBias, dim=-1)

        inter_scores = scores_trans
        user_scores = uhis
        item_scores = itemBias

        scores = F.softmax((inter_scores - self.alpha * item_scores + self.beta * user_scores), dim=-1)

        return scores