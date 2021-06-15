import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
from tqdm import tqdm
from scipy import sparse
import math
from transformers import AutoTokenizer
import json
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dmap = {
class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train", proj="Math", testid=0, lst=[]):
        self.train_path = proj + ".pkl"
        self.val_path = "ndev.txt"  # "validD.txt"
        self.test_path = "ntest.txt"
        self.proj = proj
        self.SentenceLen = config.SentenceLen
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Voc['Method'] = len(self.Nl_Voc)
        self.Nl_Voc['Test'] = len(self.Nl_Voc)
        self.Nl_Voc['Line'] = len(self.Nl_Voc)
        self.Nl_Voc['RTest'] = len(self.Nl_Voc)
        self.Nl_Len = config.NlLen
        # self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.ids = []
        self.Nls = []
        if os.path.exists("nl_voc.pkl"):
        #    self.init_dic()
            self.Load_Voc()
        else:
            self.init_dic()
        print(self.Nl_Voc)
        if not os.path.exists(self.proj + 'data.pkl'):
            data = self.preProcessData(open(self.train_path, "rb"))
        else:
            data = pickle.load(open(self.proj + 'data.pkl', 'rb'))
        self.data = []
        if dataName == "train":
            for i in range(len(data)):
                #if testid == 0:
                #    self.data.append(data[i][testid + 1:])
                #elif testid == len(data[i]) - 1:
                #    self.data.append(data[i][0:testid] + data[i][testid + 1:])
                #else:
                tmp = []
                for j in range(len(data[i])):
                    if j in lst:
                        continue
                    tmp.append(data[i][j])#self.data.append(data[i][0:testid] + data[i][testid + 1:])
                self.data.append(tmp)
            #for i in range(len(data)):
            #    self.data = data[0:testid] + data[testid + 1:]
        elif dataName == 'test':
            #self.data = self.preProcessData(open('Lang.pkl', 'rb'))
            #self.ids = []
            testnum = 0#int(0.05 * len(data[0]))
            ids = []
            while len(ids) < testnum:
                rid = random.randint(0, len(data[0]) - 1)
                if rid == testid or rid in ids or rid == 51:#if rid >= testid * testnum and rid < testid * testnum + testnum or rid in ids:
                    continue
                ids.append(rid)
            self.ids = ids
            for i in range(len(data)):
                tmp = []
                for x in self.ids:
                    tmp.append(data[i][x])
                self.data.append(tmp)
        else:
            testnum = 1#int(0.1 * len(data[0]))
            ids = []
            for i in range(len(data)): 
                tmp = []
                for x in range(testnum * testid, testnum * testid + testnum):
                    if x < len(data[i]):
                        if i == 0:
                            ids.append(x)
                        tmp.append(data[i][x])
                self.data.append(tmp)
            self.ids = ids

    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))
    def splitCamel(self, token):
        ans = []
        tmp = ""
        for i, x in enumerate(token):
            if i != 0 and x.isupper() and token[i - 1].islower() or x in '$.' or token[i - 1] in '.$':
                ans.append(tmp)
                tmp = x.lower()
            else:
                tmp += x.lower()
        ans.append(tmp)
        return ans
    def init_dic(self):
        print("initVoc")
        f = open(self.p + '.pkl', 'rb')
        data = pickle.load(f)
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for x in data:
            for s in x['methods']:
                s = s[:s.index('(')]
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
                print(Codes[-1])
            for s in x['ftest']:
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
        code_voc = VocabEntry.from_corpus(Codes, size=50000, freq_cutoff = 0)
        self.Code_Voc = code_voc.word2id
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def tokenize_for_bleu_eval(self, code):
        code = re.sub(r'([^A-Za-z0-9])', r' \1 ', code)
        #code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = code.replace('"', '`')
        code = code.replace('\'', '`')
        tokens = [t for t in code.split(' ') if t]
        return tokens
    def getoverlap(self, a, b):
        ans = []
        for x in a:
            maxl = 0
            for y in b:
                tmp = 0
                for xm in x:
                    if xm in y:
                        tmp += 1
                maxl = max(maxl, tmp)
            ans.append(int(100 * maxl / len(x)) + 1)
        return ans
    def getRes(self, codetoken, nltoken):
        ans = []
        for x in nltoken:
            if x == "<pad>":
                continue
            if x in codetoken and codetoken.index(x) < self.Code_Len and x != "(" and x != ")":
                ans.append(len(self.Nl_Voc) + codetoken.index(x))
            else:
                if x in self.Nl_Voc:
                    ans.append(self.Nl_Voc[x])
                else:
                    ans.append(1)
        for x in ans:
            if x >= len(self.Nl_Voc) + self.Code_Len:
                print(codetoken, nltoken)
                exit(0)
        return ans
    def preProcessData(self, dataFile):
        lines = pickle.load(dataFile)#dataFile.readlines()
        Nodes = []
        LineNodes = []
        LineMus = []
        Res = []
        inputText = []
        inputNlad = []
        maxl = 0
        maxl2 = 0
        error = 0
        error1 = 0
        error2 = 0
        correct = 0
        LineNum = []
        
        for k in range(len(lines)):
            x = lines[k]
            
                    
            nodes = []
            types = []
            res = []
            nladrow = []# = np.zeros([3200, 3200])
            nladcol = []
            nladval = []
            texta = []
            textb = []
            linenodes = []
            linetypes = []
            methodnum = len(x['methods'])
            rrdict = {}
            for s in x['methods']:
                rrdict[x['methods'][s]] = s[:s.index('(')]
            for i in range(methodnum):
                nodes.append('Method')
                if len(rrdict[i].split(":")) > 1:
                    tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:] + [rrdict[i].split(":")[1]]) 
                else:
                    tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:]) 
                #print(tokens, self.splitCamel(tokens))
                #tmpids = self.Get_Em(self.splitCamel(tokens), self.Code_Voc)#tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokens))#print(rrdict[i])
                ans = self.splitCamel(tokens)
                ans.remove('.')
                texta.append(ans)#(self.pad_seq(tmpids, 10))
                #print(tmpids)
                #assert(0)
                if i not in x['correctnum']:
                    types.append(1)
                else:
                    types.append(x['correctnum'][i] + 1)
                if i in x['ans']:
                    res.append(1)
                else:
                    res.append(0)
            rrdic = {}
            for s in x['ftest']:
                rrdic[x['ftest'][s]] = s
            for i in range(len(x['ftest'])):
                nodes.append('Test')
                if len(rrdic[i].split(":")) > 1:
                    tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:] + [rrdic[i].split(":")[1]])
                else:
                    tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:])
                #print(tokens, self.splitCamel(tokens))
                #tmpids = self.Get_Em(self.splitCamel(tokens), self.Code_Voc)#tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokens))#print(rrdict[i])
                #print(tmpids)
                ans = self.splitCamel(tokens)
                ans.remove('.')
                #print(ans)
                #print(self.splitCamel(tokens), self.splitCamel(tokens).remove('.'))
                #assert(0)
                textb.append(ans)#(self.pad_seq(tmpids, 10))
            
            if type(x['rtest']) == int:
                x['rtest'] = [0] * x['rtest']

            for i in range(len(x['rtest'])):
                nodes.append('RTest')

            rrdic = {}
            #for s in x['rtest']:
            #    rrdic[x['rtest'][s]] = s
            #textc = []

            mus = []
            for i in range(len(x['lines'])):
                if i not in x['ltype']:
                    x['ltype'][i] = 'Empty'
                if x['ltype'][i] not in self.Nl_Voc:
                    self.Nl_Voc[x['ltype'][i]] = len(self.Nl_Voc)
                linenodes.append(x['ltype'][i])
                if i in x['lcorrectnum']:
                    linetypes.append(x['lcorrectnum'][i] + 1)
                else:
                    linetypes.append(1)
            '''for i in range(len(x['mutation'])):
                if x['mutation'][i] not in self.Nl_Voc:
                    self.Nl_Voc[x['mutation'][i]] = len(self.Nl_Voc)
                nodes.append(x['mutation'][i])
                types.append(0)'''
            maxl = max(maxl, len(nodes))
            maxl2 = max(maxl2, len(linenodes))
            ed = {}

            line2method = {}
            line_num = [0] * methodnum
            
            for e in x['edge2']:
                # e[0]: method no
                # e[1]: line no
                assert e[1] not in line2method
                line2method[e[1]] = e[0]
                line_num[e[0]] += 1
                
            ed = {}
            for e in x['edge10']:
                # e[0]: lineno
                # e[1]: pass test no
                if e[0] not in line2method:
                    continue
                method_no = line2method[e[0]]
                if (method_no, e[1]) in ed:
                    continue
                ed[(method_no, e[1])] = 1
                a = method_no
                b = e[1] + methodnum + len(x['ftest'])
                
                nladrow.append(a)
                nladcol.append(b)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            ed = {}
            for e in x['edge']:
                # e[0]: line no
                # e[1]: fail test no
                if e[0] not in line2method:
                    continue
                method_no = line2method[e[0]]
                if (method_no, e[1]) in ed:
                    continue
                ed[(method_no, e[1])] = 1

                a = method_no
                b = e[1] + methodnum
                
                nladrow.append(a)
                nladcol.append(b)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            
            line_num = [x + 1 for x in line_num]

            overlap = self.getoverlap(texta, textb)
            '''for i in range(len(texta)):
                for j in range(len(textb)):
                    t = 0
                    for xs in texta[i]:
                        if xs in textb[j]:
                            t += 1
                    if t / len(texta[i]) > 0.4:
                        nladrow.append(i)
                        nladcol.append(self.Nl_Len + j)
                        nladval.append(t / len(texta[i]))
                for j in range(len(textc)):
                    t = 0
                    for xs in texta[i]:
                        if xs in textc[j]:
                            t += 1
                    if t / len(texta[i]) > 0.4:
                        nladrow.append(i)
                        nladcol.append(methodnum + len(x['ftest']) + j)
                        nladval.append(t / len(texta[i]))'''
            # for i in range(len(l_pass_num)):
            #     if i not in line2method:
            #         continue
            #     assert pass_num[line2method[i]] >= l_pass_num[i]
            #     assert fail_num[line2method[i]] >= l_fail_num[i]
            # print(len(types), len(pass_num))
            # print(fail_num)
            # print(l_fail_num)
            Nodes.append(self.pad_seq(self.Get_Em(nodes, self.Nl_Voc), self.Nl_Len))
            Res.append(self.pad_seq(res, self.Nl_Len))
            inputText.append(self.pad_seq(overlap, self.Nl_Len))
            # LineNodes.append(self.pad_seq(self.Get_Em(linenodes, self.Nl_Voc), self.Code_Len))
            LineNum.append(self.pad_seq(line_num, self.Nl_Len))

            row = {}
            col = {}
            for i  in range(len(nladrow)):
                if nladrow[i] not in row:
                    row[nladrow[i]] = 0
                row[nladrow[i]] += 1
                if nladcol[i] not in col:
                    col[nladcol[i]] = 0
                col[nladcol[i]] += 1
            for i in range(len(nladrow)):
                nladval[i] = 1 / math.sqrt(row[nladrow[i]]) * 1 / math.sqrt(col[nladcol[i]])
            nlad = sparse.coo_matrix((nladval, (nladrow, nladcol)), shape=(self.Nl_Len, self.Nl_Len))
            inputNlad.append(nlad)
        print("max1: %d max2: %d"%(maxl, maxl2))

        #assert(0)#assert(0)
        batchs = [Nodes, LineNum, inputNlad, Res, inputText]
        self.data = batchs
        open(self.proj + "data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
        #open('nl_voc.pkl', 'wb').write(pickle.dumps(self.Nl_Voc))
        return batchs

    def __getitem__(self, offset):
        ans = []
        if True:
            for i in range(len(self.data)):
                if i == 2:
                    #torch.FloatTensor(np.array([self.data[i][offset].row, self.data[i][offset].col])).float()
                    #torch.FloatTensor(self.data[i][offset].data)
                    #torch.FloatTensor(self.data[i][offset].data)
                    #ans.append(self.data[i][offset])
                    #ans.append(torch.sparse.FloatTensor(torch.LongTensor(np.array([self.data[i][offset].row, self.data[i][offset].col])), torch.FloatTensor(self.data[i][offset].data).float(), torch.Size([self.Nl_Len,self.Nl_Len])))
                    #open('tmp.pkl', 'wb').write(pickle.dumps(self.data[i][offset]))
                    #assert(0)
                    ans.append(self.data[i][offset].toarray())
                    #print(self.data[i][offset].toarray()[0, 2545])
                    #assert(0)
                else:
                    ans.append(np.array(self.data[i][offset]))
        else:
            for i in range(len(self.data)):
                if i == 4:
                    continue
                ans.append(np.array(self.data[i][offset]))
            negoffset = random.randint(0, len(self.data[0]) - 1)
            while negoffset == offset:
                negoffset = random.randint(0, len(self.data[0]) - 1)
            if self.dataName == "train":
                ans.append(np.array(self.data[2][negoffset]))
                ans.append(np.array(self.data[3][negoffset]))
        return ans
    def __len__(self):
        return len(self.data[0])
    def Get_Train(self, batch_size):
        data = self.data
        loaddata = data
        batch_nums = int(len(data[0]) / batch_size)
        if True:
            if self.dataName == 'train':
                shuffle = np.random.permutation(range(len(loaddata[0])))
            else:
                shuffle = np.arange(len(loaddata[0]))
            for i in range(batch_nums):
                ans = []
                for j in range(len(data)):
                    if j != 2:
                        tmpd = np.array(data[j])[shuffle[batch_size * i: batch_size * (i + 1)]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * i, batch_size * (i + 1)):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * i, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([batch_size, self.Nl_Len, self.Nl_Len])))
                yield ans
            if batch_nums * batch_size < len(data[0]):
                ans = []
                for j in range(len(data)):
                    if j != 2:
                        tmpd = np.array(data[j])[shuffle[batch_nums * batch_size: ]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * batch_nums, len(data[0])):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * batch_nums, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([len(data[0]) - batch_size * batch_nums, self.Nl_Len, self.Nl_Len])))
                yield ans
            
class node:
    def __init__(self, name):
        self.name = name
        self.father = None
        self.child = []
        self.id = -1
