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
class TreeAttEncoder(nn.Module):
    def __init__(self, args):
        super(TreeAttEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.token_embedding = Embedding(args.Code_Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])
        self.transformerBlocksTree = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])


    def forward(self, input_code, input_codechar, inputAd):
        codemask = torch.gt(input_code, 0)
        charEm = self.char_embedding(input_codechar)
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        #print(charEm.shape)
        x = self.token_embedding(input_code.long())
        for trans in self.transformerBlocksTree:
            x = trans.forward(x, codemask, charEm, inputAd, True)
        for trans in self.transformerBlocks:
            x = trans.forward(x, codemask, charEm)
        return x
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
        self.token_embedding = nn.Embedding(args.Nl_Vocsize, self.embedding_size - 1)
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
        self.criterion = nn.BCEWithLogitsLoss(reduction = 'none')
    def forward(self, input_node, inputtype, inputad, res, inputtext, linenode, linetype, linemus):
        nlmask = torch.gt(input_node, 0)
        resmask = torch.eq(input_node, 2)#torch.gt(res, 0)
        #linemask = torch.eq(input_node, 4)
        inputad = inputad.float()
        nodeem = self.token_embedding(input_node)
        #nodeem = torch.cat([nodeem, inputtype.unsqueeze(-1).float(), inputtext.unsqueeze(-1).float()], dim=-1)
        nodeem = torch.cat([nodeem, inputtext.unsqueeze(-1).float()], dim=-1)
        x = nodeem#self.norm(nodeem)
        #childEm = self.text_embedding(inputtext)
        #childEm = self.conv(childEm.permute(0, 3, 1, 2))
        #childEm = childEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        '''text = self.text_embedding(inputtext)
        text = text.view(-1, 10, self.embedding_size // 2)
        length = torch.gt(inputtext.view(-1, 10), 0).sum(dim=-1)
        _, idx_sort = torch.sort(length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        text = text.index_select(0, idx_sort)
        length = length.masked_fill(length == 0, 1)
        padedseq = nn.utils.rnn.pack_padded_sequence(text, length[idx_sort].cpu(), batch_first=True)
        packed_wenc, hc_out = self.lstm(padedseq)
        wenc, _l = nn.utils.rnn.pad_packed_sequence(packed_wenc, batch_first=True)
        text = wenc[tuple(range(text.size(0))), length[idx_sort] - 1]
        text = text[idx_unsort]
        text = text.view(-1, inputtext.size(1), self.embedding_size // 2)
        x = torch.cat([x, text], dim=-1)'''
        #x = x + childEm
        #for trans in self.transformerBlocksTree:
        #    x = trans(x, nlmask, childEm)
        #nlmask2 = torch.gt(linenode)
        lineem = self.token_embedding1(linenode)
        # lineem = torch.cat([linetype.unsqueeze(-1).float(), lineem], dim=-1)
        x = torch.cat([x, lineem], dim=1)
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, inputad)
        #x = torch.cat([nodeem, inputtype.unsqueeze(-1).float()], dim=-1)
        x = x[:,:input_node.size(1)]
        #x = torch.cat([x, inputtext.unsqueeze(dim=-1).float()], dim=-1)
        xx = self.resLinear2(x).squeeze(-1)
        resSoftmax = F.softmax(xx.masked_fill(resmask==0, -1e9), dim=-1)
        # loss = -torch.log(resSoftmax.clamp(min=1e-10, max=1)) * res
        loss = self.criterion(xx, res.float())
        loss = loss.masked_fill(resmask == 0, 0)
        loss = loss.sum() / resmask.sum()
        return loss, resSoftmax, x

        resSoftmax = F.softmax(self.resLinear(x), dim=-1)
        #print(resSoftmax.size())
        #res_one = F.one_hot(res, 2)
        #smooth = 0.0
        #mas = 1 - res_one
        #res_one = res_one * (1 - smooth) + smooth * mas
        #loss = - (res_one * torch.log((resSoftmax).clamp(min=1e-10, max=1))).sum(dim=-1) 
        loss = -torch.log(torch.gather(resSoftmax.clamp(min=1e-10, max=1.0), -1, res.unsqueeze(-1))).squeeze(-1)
        #print(loss.sum())
        #loss2 = loss.masked_fill(linemask == 0, 0.0)
        loss = loss.masked_fill(resmask == 0, 0.0)
        posmask = torch.eq(res, 1)
        poloss = loss.masked_fill(posmask == 0, 0)
        loss1 = torch.sum(poloss, dim=-1) / posmask.sum(dim=-1).float()
        resTruelen = torch.sum(resmask, dim=-1).float()
        #print('ppp', input_node.data.cpu().numpy())
        #print('ppp', resmask.data.cpu().numpy())
        totalloss = torch.sum(loss, dim=-1) / resTruelen 
        #print(totalloss)
        #print(poloss.size(), totalloss.size())
        totalloss = totalloss + 0.2 * loss1#+ loss2
        #w = torch.from_numpy(np.array([0.2, 0.8])).cuda()
        #totalloss = self.loss(ress.permute(0, 2, 1), res)
        return totalloss, resSoftmax
class CodeEncoder(nn.Module):
    def __init__(self, args):
        super(CodeEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(20)])
        self.token_embedding = Embedding(args.Code_Vocsize, self.embedding_size)
        '''self.transformerBlocksTree = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])'''


    def forward(self, input_nl, input_nlchar):
        nlmask = torch.gt(input_nl, 0)
        charEm = self.char_embedding(input_nlchar.long())
        #print(input_nlchar.size())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        x = self.token_embedding(input_nl.long())
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, charEm)
        print(x.size())
        return x, nlmask
class CopyNet(nn.Module):
    def __init__(self, args):
        super(CopyNet, self).__init__()
        self.embedding_size = args.embedding_size
        self.LinearSource = nn.Linear(self.embedding_size, self.embedding_size)
        self.LinearTarget = nn.Linear(self.embedding_size, self.embedding_size)
        self.LinearRes = nn.Linear(self.embedding_size, 1)
        self.LinearProb = nn.Linear(self.embedding_size, 2)
    def forward(self, source, traget):
        sourceLinear = self.LinearSource(source)
        targetLinear = self.LinearTarget(traget)
        genP = self.LinearRes(F.tanh(sourceLinear.unsqueeze(1) + targetLinear.unsqueeze(2))).squeeze(-1)
        prob = F.softmax(self.LinearProb(traget), dim=-1)#.squeeze(-1))
        return genP, prob
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.word_len = args.WoLen
        self.nl_len = args.NlLen
        self.classnum = args.Nl_Vocsize + args.CodeLen*args.SentenceLen
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.resLen = args.Nl_Vocsize
        self.code_len = args.CodeLen
        self.transformerBlocks = nn.ModuleList(
            [rightTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.encoder = CodeEncoder(args)#LineEncoder(args)
        self.finalLinear = nn.Linear(self.embedding_size, 1024)
        self.resLinear = nn.Linear(1024, self.resLen)
        self.token_embedding = Embedding(args.Nl_Vocsize, self.embedding_size)
        #self.copy = CopyNet(args)
    def getBleu(self, losses, ngram):
            bleuloss = []
            for i in range(len(losses) - ngram + 1):
                tmp = []
                for j in range(ngram): 
                    tmp.append(losses[i + j].squeeze(1))
                tmpLoss = torch.max(torch.stack(tmp, dim=-1), -1)[0]
                bleuloss.append(tmpLoss)
            bleuloss = torch.mean(torch.stack(bleuloss, 1)) 
            return bleuloss
    def getNlencode(self, input_nl, input_nlchar):
        nlmask = torch.gt(input_nl, 0)
        charEm = self.encoder.char_embedding(input_nlchar.long())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        x = self.token_embedding(input_nl.long())
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, None, None, charEm, False)
        return x
    def getCodeencode(self, inputcode, inputcodechar):
        return self.encoder(inputcode, inputcodechar)[0] 
    def forward(self, input_nl, input_nlchar, input_code, input_codechar, inputres, antimask):
        nlmask = antimask#.unsqueeze(0).repeat(input_nl.size(0), 1, 1).unsqueeze(1)
        resmask = torch.gt(inputres, 0)
        charEm = self.encoder.char_embedding(input_nlchar.long())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        encodes, codemask = self.encoder(input_code, input_codechar)
        #print(encodes.size())
        #codemask = torch.gt(torch.sum(encodes, -1), 0)
        x = self.token_embedding(input_nl.long())
        #print(nlmask.shape)
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, encodes, codemask, charEm)
        hstate = x
        x = self.finalLinear(x)
        x = self.resLinear(x)
        res = F.softmax(x, dim=-1)
        loss = -torch.log(torch.gather(res.clamp(min=1e-10, max=1.0), -1, inputres.unsqueeze(-1))).squeeze(-1)
        loss = loss.masked_fill(resmask == 0, 0.0)
        #losses = torch.split(loss, 1 , dim=1)
        resTruelen = torch.sum(resmask, dim=-1).float()
        totalloss = torch.mean(loss, dim=-1) * self.nl_len / resTruelen
        #totalloss = torch.mean(totalloss)#torch.mean(totalloss)
        return totalloss, res
class LineEncoder(nn.Module):
    def __init__(self, args):
        super(LineEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.codeEncoder = CodeEncoder(args)
        self.Attention = MultiHeadedAttention(8, self.embedding_size, 0.1)
        self.poolConvnl = nn.Conv1d(self.embedding_size, self.embedding_size, 3, padding=1)
        self.maxPoolnl = nn.MaxPool1d(args.CodeLen)
        self.att = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.lineConv = nn.Conv1d(self.embedding_size, self.embedding_size, 3, padding=1)
        self.lineEmbedding = PositionalEmbedding(self.embedding_size)
        self.Attention = MultiHeadedAttention(8, self.embedding_size, 0.1)
        self.feed_forward = DenseLayer(d_model=self.embedding_size, d_ff=4*self.embedding_size, dropout=0.1)
        self.sublayer = SublayerConnection(size=self.embedding_size, dropout=0.1)

    def forward(self, inputCode, inputCodeChar):
        lineEm = self.lineEmbedding(inputCode)
        lineEms = torch.split(lineEm, 1, dim=1)
        codelines = torch.split(inputCode, 1, dim=1)
        codelineschar = torch.split(inputCodeChar, 1, dim=1)
        codelinesEncoding = []
        codelinesmask = []
        linemask = torch.gt(torch.sum(inputCode, dim=-1), 0)
        for i in range(len(codelines)):
            codelinesEncoding.append(self.codeEncoder(codelines[i].squeeze(1), codelineschar[i].squeeze(1), lineEms[i].squeeze(1)))
            codelinesmask.append(torch.gt(codelines[i].squeeze(1), 0))
        finalOutputs = []
        for i in range(len(codelinesEncoding)):
            output = []
            for j in range(len(codelinesEncoding)):
                output.append(self.Attention(codelinesEncoding[i], codelinesEncoding[j], codelinesEncoding[j], codelinesmask[j]))
            outputs = torch.stack(output, 1)
            outputs = torch.sum(linemask.unsqueeze(-1).unsqueeze(-1).float() * outputs, 1)
            output = self.sublayer(outputs, self.feed_forward)
            output = output + codelinesEncoding[i]
            finalOutputs.append(output)
        finalOutput = torch.stack(finalOutputs, -2)
        finalOutput = finalOutput.flatten(start_dim=1, end_dim=2)
        mask = torch.cat(codelinesmask, -1)
        return finalOutput, mask#, finalOutput[:,0:self.codelen,:], codelinesmask[0]

class JointEmbber(nn.Module):
    def __init__(self, args):
        super(JointEmbber, self).__init__()
        self.embedding_size = args.embedding_size
        self.codeEncoder = LineEncoder(args)
        self.margin = args.margin
        self.nlEncoder = NlEncoder(args)
        self.poolConvnl = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.maxPoolnl = nn.MaxPool1d(args.NlLen)
        self.codeLinear = nn.Linear(args.CodeLen * args.SentenceLen * self.embedding_size, self.embedding_size)
    def scoring(self, qt_repr, cand_repr):
        sim = F.cosine_similarity(qt_repr, cand_repr)
        return sim
    def nlencoding(self, inputnl, inputnlchar):
        nl = self.nlEncoder(inputnl, inputnlchar)
        nl = self.maxPoolnl(self.poolConvnl(nl.permute(0, 2, 1))).squeeze(-1)
        return nl
    def codeencoding(self, inputcode, inputcodechar):
        code = self.codeEncoder(inputcode, inputcodechar)
        code = self.codeLinear(code[0].flatten(start_dim=1))
        return code
    def forward(self, inputnl, inputnlchar, inputcode, inputcodechar, inputcodeneg, inputcodenegchar):
        #print(inputcode.size(), inputcodechar.size())
        nl = self.nlencoding(inputnl, inputnlchar)
        code = self.codeencoding(inputcode, inputcodechar)
        codeneg = self.codeencoding(inputcodeneg, inputcodenegchar)
        good_score = self.scoring(nl, code)
        bad_score = self.scoring(nl, codeneg)
        loss = (self.margin - good_score + bad_score).clamp(min=1e-6)#.mean()
        return loss, good_score, bad_score

class CombineModel(nn.Module):
    def __init__(self, args):
        super(CombineModel, self).__init__()
        self.Decoder = Decoder(args)
        self.embedding_size = args.embedding_size
        #self.encoder = CodeEncoder(args)
        self.poolConvnl = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.maxPoolnl = nn.MaxPool1d(args.NlLen)
        self.poolConvcode = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.maxPoolcode = nn.MaxPool1d(args.CodeLen)
        self.margin = args.margin
    def nlencoding(self, inputnl, inputnlchar):
        nl = self.Decoder.getNlencode(inputnl, inputnlchar)
        nl = self.maxPoolnl(self.poolConvnl(nl.permute(0, 2, 1))).squeeze(-1)
        return nl
    def codeencoding(self, inputcode, inputcodechar):
        code = self.Decoder.getCodeencode(inputcode, inputcodechar)
        code = self.maxPoolcode(self.poolConvcode(code.permute(0, 2, 1))).squeeze(-1)
        return code
    def scoring(self, qt_repr, cand_repr):
        sim = F.cosine_similarity(qt_repr, cand_repr)
        return sim
    def decode(self, inputnl, inputnlchar, inputcode, inputcodechar, inputres, antimask):
        return self.Decoder(inputnl, inputnlchar, inputcode, inputcodechar, inputres, antimask)
    def forward(self, inputnl, inputnlchar, inputcode, inputcodechar, inputres, inputcodeneg, inputcodenegchar, antimask):
        loss1, _ = self.Decoder(inputnl, inputnlchar, inputcode, inputcodechar, inputres , antimask)
        nl = self.nlencoding(inputnl, inputnlchar)
        code = self.codeencoding(inputcode, inputcodechar)
        codeneg = self.codeencoding(inputcodeneg, inputcodenegchar)
        good_score = self.scoring(nl, code)
        bad_score = self.scoring(nl, codeneg)
        loss = (self.margin - good_score + bad_score).clamp(min=1e-6).mean()
        return loss1 + 5 * loss

        








