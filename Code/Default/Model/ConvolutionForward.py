import torch.nn as nn
from gelu import GELU
class ConvolutionLayer(nn.Module):
    def __init__(self, dmodel, layernum, kernelsize=3, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.conv1 = nn.Conv1d(dmodel, layernum, kernelsize, padding=(kernelsize-1)//2)
        self.conv2 = nn.Conv1d(dmodel, layernum, kernelsize, padding=(kernelsize-1)//2)
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).repeat(1, 1, x.size(2))
        x = x.masked_fill(mask==0, 0)
        convx = self.conv1(x.permute(0, 2, 1))
        convx = self.dropout(self.activation(convx))
        out = self.conv2(convx).permute(0, 2, 1)
        return out#self.dropout(self.activation(self.conv1(self.conv2(x))))
