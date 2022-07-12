import torch.nn as nn
import torch.nn.functional as F
import torch
from Transfomer import TransformerBlock
from rightTransfomer import rightTransformerBlock
from Embedding import Embedding
from Multihead_Attention import MultiHeadedAttention
from postionEmbedding import PositionalEmbedding
from LayerNorm import LayerNorm
from SubLayerConnection import *
from DenseLayer import DenseLayer
import numpy as np

class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.token_embedding = nn.Embedding(args.Nl_Vocsize, self.embedding_size-1)
        self.token_embedding1 = nn.Embedding(args.Nl_Vocsize, self.embedding_size)

        self.text_embedding = nn.Embedding(20, self.embedding_size)
        self.transformerBlocksTree = nn.ModuleList(
            [rightTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.resLinear = nn.Linear(self.embedding_size, 2)
        self.pos = PositionalEmbedding(self.embedding_size)
        self.loss = nn.CrossEntropyLoss()
        self.norm = LayerNorm(self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size // 2, int(self.embedding_size / 4), batch_first=True, bidirectional=True)
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))
        self.resLinear2 = nn.Linear(self.embedding_size, 1)
    def forward(self, input_node, inputtype, inputad, res, inputtext, linenode, linetype, linemus):
        nlmask = torch.gt(input_node, 0)
        resmask = torch.eq(input_node, 2)#torch.gt(res, 0)
        inputad = inputad.float()
        nodeem = self.token_embedding(input_node)
        nodeem = torch.cat([nodeem, inputtext.unsqueeze(-1).float()], dim=-1)
        x = nodeem
        lineem = self.token_embedding1(linenode)
        x = torch.cat([x, lineem], dim=1)
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, inputad)
        x = x[:,:input_node.size(1)]
        resSoftmax = F.softmax(self.resLinear2(x).squeeze(-1).masked_fill(resmask==0, -1e9), dim=-1)
        loss = -torch.log(resSoftmax.clamp(min=1e-10, max=1)) * res
        loss = loss.sum(dim=-1)
        return loss, resSoftmax, x

       




