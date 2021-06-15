import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=512, Bert=False):
        super(TokenEmbedding, self).__init__()
        if Bert:
            self.em = nn.Embedding(vocab_size, 768, padding_idx=0)
            self.Linear = nn.Linear(768, embed_size)
        else:
            self.em = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            #super().__init__(vocab_size, embed_size, padding_idx=0)
        self.useBert = Bert
    def forward(self, inputtokens):
        out = self.em(inputtokens)
        if self.useBert:
            out = self.Linear(out)
        return out
        
