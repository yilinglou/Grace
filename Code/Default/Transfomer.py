import torch.nn as nn
from gcnn import GCNN
from Multihead_Attention import MultiHeadedAttention
from SubLayerConnection import SublayerConnection
from DenseLayer import DenseLayer
from LayerNorm import LayerNorm

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.Tconv_forward = GCNN(dmodel=hidden)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(hidden)

    def forward(self, x, mask, inputP):
        #x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, _x, _x, mask=mask))
        #x = self.sublayer2(x, lambda _x: self.combination.forward(_x, _x, pos))
        #x = self.sublayer3(x, lambda _x: self.combination2.forward(_x, _x, charem))
        #print(x.size())
        x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, None, inputP))
        x = self.norm(x)
        return self.dropout(x)
