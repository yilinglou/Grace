import torch
from torch import optim
from Dataset_noline import SumDataset
import os
from tqdm import tqdm
from Model_noline import *
import numpy as np
#from annoy import AnnoyIndex
from nltk import word_tokenize
import pickle
from ScheduledOptim import ScheduledOptim
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import random
import adamod
import time
import sys
#import wandb
#wandb.init(project="codesum")
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

NlLen_map = {"Time":3900, "Math":4500, "Lang":280, "Chart": 2350, "Mockito":1780, "unknown":2200, "Closure":6000}
batch_size_map = {"Chart":60, 'Lang':60, 'Time':60, "Math":60, "Mockito":60, "Closure":20}
args = dotdict({
    'NlLen':NlLen_map[sys.argv[2]],
    'SentenceLen':10,
    'embedding_size':32,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'lr':0.01
})
os.environ['PYTHONHASHSEED'] = str(args.seed)
#os.environ["CUDA_VISIBLE_DEVICES"]="4, 1"
def save_model(model, dirs = "checkpointcodeSearch"):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + '/best_model.ckpt')


def load_model(model, dirs="checkpointcodeSearch"):
    assert os.path.exists(dirs + '/best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + '/best_model.ckpt'))
def evalscore(trans, ground):
    score = 0
    recall = 0
    for i in range(len(trans)):
        ori = ground[i][0]
        pre = 0
        lll = []
        for key in trans[i]:
            if key not in  ["Unknown", "unknown"]:
                lll.append(key)
        trans[i] = lll
        for t in range(len(trans[i])):
            word = trans[i][t]
            if word in ori:
                pre += 1
        score += float(pre) / max(1, len(trans[i]))
        pre = 0 
        for t in range(len(ori)):
            word = ori[t]
            if word in trans[i]:
                pre += 1
        recall += float(pre) / len(ori) 
    score /= len(trans)
    recall /= len(trans)
    return score, recall
use_cuda = torch.cuda.is_available()
def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = gVar(data[i])
        tensor = data
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor
def getAntiMask(size):
  ans = np.zeros([size, size])
  for i in range(size):
    for j in range(0, i + 1):
      ans[i, j] = 1.0
  return ans
def getAdMask(size):
  ans = np.zeros([size, size])
  for i in range(size - 1):
      ans[i, i + 1] = 1.0
  return ans
def getAhMask(size):
  ans = np.zeros([size, size])
  for i in range(size - 1):
      ans[i + 1, i] = 1.0
  return ans
def train(t = 5, p='Math'):
    # print(args.seed)
    # print(args.lr)
    # print(args.CodeLen)
    # print(args.NlLen)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    random.seed(args.seed + t)
    #torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 

    #np.random.seed(args.seed)  
    #random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev_set = SumDataset(args, "test", p, testid=t)
    val_set = SumDataset(args, "val", p, testid=t)
    data1 = pickle.load(open(p + '.pkl', 'rb'))

    #dev_data = pickle.load(open(p + '.pkl', 'rb'))
    train_set = SumDataset(args, "train", testid=t, proj=p, lst=dev_set.ids + val_set.ids)
    #val_set = SumDataset(args, 'val', testid=t)
    numt = len(train_set.data[0])
    #print(numt)
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)

    #data = data[t]
    print(dev_set.ids)
    model = NlEncoder(args)
    if use_cuda:
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
        #model = nn.DataParallel(model, device_ids=[0, 1])
    #load_model(model)
    #wandb.watch(model)
    #model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1,  2, 3, 4, 5, 6, 7])
    #nlem = pickle.load(open("embedding.pkl", "rb"))
    #codeem = pickle.load(open("code_embedding.pkl", "rb"))
    #model.encoder.token_embedding.token.em.weight.data.copy_(gVar(nlem))
    #model.token_embedding.token.em.weight.data.copy_(gVar(codeem))
    #model.token_embedding.token.em.weight.data.copy_(gVar(codeem))
    maxl = 1e9
    #optimizer = ScheduledOptim(adamod.AdaMod(model.parameters(), lr=1e-3, beta3=0.999), args.embedding_size, 4000)
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)
    maxAcc = 0
    minloss = 1e9
    rdic = {}
    brest = []
    bans = []
    batchn = []
    each_epoch_pred = {}
    for x in dev_set.Nl_Voc:
      rdic[dev_set.Nl_Voc[x]] = x
    for epoch in range(11):
        index = 0
        for dBatch in tqdm(train_set.Get_Train(args.batch_size)):
            if index == 0:
                accs = []
                loss = []
                model = model.eval()
                '''for k, devBatch in tqdm(enumerate(train_set.Get_Train(args.batch_size))):
                  for i in range(len(devBatch)):
                    devBatch[i] = gVar(devBatch[i])
                  with torch.no_grad():
                    l, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3])
                    pred = pre.argmax(dim=-1)
                    loss.append(l.sum().item())'''
                '''alst = []
                for k, devBatch in tqdm(enumerate(dev_set.Get_Train(len(dev_set)))):
                  for i in range(len(devBatch)):
                    devBatch[i] = gVar(devBatch[i])
                  with torch.no_grad():
                    l, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[5], devBatch[6], devBatch[7])
                    resmask = torch.eq(devBatch[0], 2)
                    s = -pre#-pre[:, :, 1]
                    s = s.masked_fill(resmask == 0, 1e9)
                    #print(s)
                    pred = s.argsort(dim=-1)
                    pred = pred.data.cpu().numpy()
                    for l in range(len(pred)):
                        datat = dev_data[dev_set.ids[l]]
                        lst = pred[l].tolist()[:resmask.sum(dim=-1)[l].item()]
                        maxn = 1e9
                        for x in datat['ans']:
                            i = lst.index(x)
                            maxn = min(maxn, i)
                        alst.append(maxn)
                score = np.mean(alst)'''
                #score = []
                #print(numt)
                score2 = []
                for k, devBatch in tqdm(enumerate(val_set.Get_Train(len(val_set)))):
                        for i in range(len(devBatch)):
                            devBatch[i] = gVar(devBatch[i])
                        with torch.no_grad():
                            l, pre, xx = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4])
                            resmask = torch.eq(devBatch[0], 2)
                            s = -pre#-pre[:, :, 1]
                            s = s.masked_fill(resmask == 0, 1e9)
                            pred = s.argsort(dim=-1)
                            pred = pred.data.cpu().numpy()
                            alst = []

                            for k in range(len(pred)): 
                                datat = data1[val_set.ids[k]]
                                maxn = 1e9
                                lst = pred[k].tolist()[:resmask.sum(dim=-1)[k].item()]#score = np.sum(loss) / numt
                                #bans = lst
                                all_bug = []
                                for x in datat['ans']:
                                    i = lst.index(x)
                                    all_bug.append(i)
                                    maxn = min(maxn, i)
                                print('all bug res', all_bug)
                                score2.append(maxn)

                        # 记录sus score(closure)
                        if epoch == 10:
                            assert len(pre) == 1
                            pre = pre.masked_fill(resmask == 0, 0)
                            pre_num = pre[0].cpu().numpy().tolist()[:resmask[0].sum().item()]
                            assert val_set.ids[0] == t
                            datat = data1[val_set.ids[0]]
                            methods_map = datat['methods']
                            methods_rmap = {}
                            for method_name in methods_map:
                                methods_rmap[methods_map[method_name]] = method_name
                            assert len(methods_map) == len(methods_rmap) == len(pre_num)
                            score_path = os.path.join('spectrum_scores_noline_final', p, "%s_%s_%s"%(args.lr, args.seed, args.batch_size))
                            if not os.path.exists(score_path):
                                os.makedirs(score_path)
                            with open(score_path + '/' + str(t),'w') as f:
                                for index in range(len(pre_num)):
                                    f.write(methods_rmap[index] + ',' + str(pre_num[index]) + '\n')

                        # 记录34维特征
                        # assert len(xx) == 1
                        # # x: 1 * len * 32
                        # xx = xx.masked_fill(resmask.unsqueeze(-1) == 0, 0)
                        # x_num = xx[0].cpu().numpy().tolist()[:resmask[0].sum().item()]
                        # # print(sum(pre_num))
                        # assert val_set.ids[0] == t
                        # datat = data1[val_set.ids[0]]
                        # methods_map = datat['methods']
                        # methods_rmap = {}
                        # for method_name in methods_map:
                        #     methods_rmap[methods_map[method_name]] = method_name
                        # assert len(methods_map) == len(methods_rmap) == len(x_num)
                        # score_path = os.path.join('spectrum_scores_34', p, str(epoch))
                        # if not os.path.exists(score_path):
                        #     os.makedirs(score_path)
                        # with open(score_path + '/' + str(t),'w') as f:
                        #     for index in range(len(x_num)):
                        #         f.write(methods_rmap[index] + ' ' + ','.join([str(x) for x in x_num[index]]) + '\n')

                each_epoch_pred[epoch] = lst
                score = score2[0]
                print('curr accuracy is ' + str(score) + "," + str(score2))
                if score2[0] == 0:
                    batchn.append(epoch)
                    

                if  maxl >= score:
                    brest = score2
                    bans = lst
                    maxl = score
                    print("find better score " + str(score) + "," + str(score2))
                    #save_model(model)
                    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
                model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4])
            print(loss.mean().item())
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            #print(loss.item())
            #if loss.item() < maxl:
            #    maxl = loss.item()
            #    save_model(model)
            optimizer.step_and_update_lr()
            index += 1
    return brest, bans, batchn, each_epoch_pred
class SearchNode:
  def __init__(self, state, prob, ans):
    self.state = state
    self.prob = prob
    self.ans = ans
def BeamSearch(inputCode, vds, model, beamsize, batch_size, k):
    with torch.no_grad():
        rdic = {}
        for x in vds.Nl_Voc:
            rdic[vds.Nl_Voc[x]] = x
        beams = {}
        for i in range(batch_size):
            beams[i] = [SearchNode([vds.Nl_Voc['<start>']], 1, ['<start>'])]
        index = 0
        antimask = gVar(getAntiMask(args.NlLen))
        antimask = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)
        while True:
            tmpFed = []
            tmpChars = []
            endnum = {}
            for i in range(batch_size):
                for x in beams[i]:
                    if x.ans[-1] == "<end>" or len(x.ans) >= args.NlLen:
                        ##print(x.prob)
                        endnum[i] = 1
                    tmpFed.append(vds.pad_seq(x.state, args.NlLen))
                    tmpChar = vds.Get_Char_Em(x.ans)
                    for j in range(len(tmpChar)):
                        tmpChar[j] = vds.pad_seq(tmpChar[j], args.WoLen)
                    tmpChar = vds.pad_list(tmpChar, args.NlLen, args.WoLen)
                    tmpChars.append(tmpChar)
            if len(endnum) == batch_size:
                break
            tmpFed = np.array(tmpFed)
            tmpChars = np.array(tmpChars)
            _, result = model(gVar(tmpFed), gVar(tmpChars), gVar(inputCode[0]), gVar(inputCode[1]), gVar(inputCode[2]), antimask)
            results = torch.argmax(result, dim=-1)#result.data.cpu().numpy()
            results = results.data.cpu().numpy()
                #print(result, inputCode)
            for j in range(batch_size):
                tmpbeam = []
                for t, x in enumerate(beams[j]):
                    #result = np.negative(results[j * beamsize + t, index])
                    #cresult = np.negative(result)
                    indexs = [results[j * beamsize + t, index]]#np.argsort(result)
                    
                    for i in range(beamsize):
                        if indexs[i] >= args.Nl_Vocsize:
                            if vds.code[args.batch_size * k + j][indexs[i] - args.Nl_Vocsize] in vds.Nl_Voc:
                                newState = x.state + [vds.Nl_Voc[vds.code[args.batch_size * k + j][indexs[i] - args.Nl_Vocsize]]]
                            else:
                                newState = x.state + [1]
                        else:
                            newState = x.state + [indexs[i]]
                        newprob = x.prob * 1#* cresult[indexs[i]]
                        if indexs[i] >= args.Nl_Vocsize:
                            newans = x.ans + [vds.code[args.batch_size * k + j][indexs[i] - args.Nl_Vocsize]]
                        else:              
                            newans = x.ans + [rdic[indexs[i]]]
                        tmpbeam.append(SearchNode(newState, newprob, newans))
                tmpbeam = sorted(tmpbeam, key = lambda x: x.prob, reverse=True)
                   #print(tmpbeam[0].prob)
                beams[j] = tmpbeam[:beamsize]
            index += 1
        return beams
def test():
    dev_set = SumDataset(args, "val")
    print(len(dev_set))
    args.Nl_Vocsize = len(dev_set.Nl_Voc)
    args.Code_Vocsize = len(dev_set.Code_Voc)
    args.Vocsize = len(dev_set.Char_Voc)
    rdic = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    #print(dev_set.Nl_Voc)
    model = Decoder(args)
    if torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=0)
    model = model.eval()
    load_model(model)
    f = open("outval.txt", "w")
    index = 0 
    for x in tqdm(devloader):
        ans = BeamSearch((x[2], x[3], x[4]), dev_set, model, 1, args.batch_size, index)
        index += 1
        for i in range(args.batch_size):
            beam = ans[i]
            for x in beam[0].ans:
                if x == "<start>":
                    continue
                if x == "<end>":
                    break
                f.write(x + " ")
            f.write("\n")
        #f.write(" ".join(ans.ans[1:-1]))
        #f.write("\n")
        #f.flush()#print(ans)s

def trainsearch():
    train_set = SumDataset(args, "train", "search")
    print(len(train_set.data[0]))
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    dev_set = SumDataset(args, "val", "search")
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=True, drop_last=True, num_workers=0)
    model = JointEmbber(args)
    #load_model(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    maxMrr = 0
    if torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
    load_model(model)
    model = nn.DataParallel(model, device_ids=[0, 1])
    #load_model(model)
    #save_model(model.module)
    for epoch in range(100000):
        model = model.train()
        for dBatch in tqdm(data_loader):
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[5])
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        mrr = evalMrr(model.module, dev_set)
        wandb.log({"mrr":mrr})
        if maxMrr < mrr:
            print("find better accuracy " + str(mrr))
            save_model(model.module)
            maxMrr = mrr
def evalMrr(model, dataset):
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.poolsize,
                                              shuffle=False, num_workers=0, drop_last=True)
    model = model.eval()
    mrr = []
    for dBatch in tqdm(data_loader):
        for i in range(len(dBatch)):
            dBatch[i] = gVar(dBatch[i])
        with torch.no_grad():
            nl_encoding = model.nlencoding(dBatch[0], dBatch[1])
            code_encoding = model.codeencoding(dBatch[2], dBatch[3])
            for i in range(args.poolsize):
                qts_repr = nl_encoding[i].expand(args.poolsize, -1)
                scores = model.scoring(qts_repr, code_encoding).data.cpu().numpy()
                neg_scores = np.negative(scores)
                predict = np.argsort(neg_scores)
                predict = [int(k) for k in predict]
                mrr.append(1 / (predict.index(i) + 1))
    return np.mean(mrr)

        
def combinetrain():
    train_set = SumDataset(args, "train")
    print(len(train_set.data[0]))
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    dev_set = SumDataset(args, "test")
    print(len(dev_set.data[3]))
    antimask = gVar(getAntiMask(args.NlLen))
    antimask = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=True, drop_last=True, num_workers=1)
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                              shuffle=False, drop_last=True, num_workers=1)
    model = CombineModel(args)
    if torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
    #load_model(model)
    #wandb.watch(model)
    #model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1,  2, 3, 4, 5, 6, 7])
    #nlem = pickle.load(open("embedding.pkl", "rb"))
    #codeem = pickle.load(open("code_embedding.pkl", "rb"))
    #model.encoder.token_embedding.token.em.weight.data.copy_(gVar(nlem))
    #model.token_embedding.token.em.weight.data.copy_(gVar(codeem))
    #model.token_embedding.token.em.weight.data.copy_(gVar(codeem))
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=1e-4), args.embedding_size, 4000)
    maxAcc = 0
    minloss = 1e9
    rdic = {}
    for x in dev_set.Nl_Voc:
      rdic[dev_set.Nl_Voc[x]] = x
    for epoch in range(100000):
        model = model.train()
        index = 0
        for dBatch in tqdm(data_loader):
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[5], dBatch[6], antimask)
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step_and_update_lr()
            if index == 100:
                trans = []
                ground = []
                losses = []
                for k, devBatch in tqdm(enumerate(devloader)):
                  for i in range(len(devBatch)):
                    devBatch[i] = gVar(devBatch[i])
                  with torch.no_grad():
                    l, pre = model.decode(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], antimask)
                    losses.append(l.mean().item())
                    pred = pre.argmax(dim=-1)
                    pred = pred.data.cpu().numpy()
                    for p, x in enumerate(pred):
                      tmp = []
                      for y in x:
                        if y >= args.Nl_Vocsize:
                          tmp.append(dev_set.code[k * args.batch_size + p][y - args.Nl_Vocsize])
                        else:
                          tmp.append(rdic[y])
                      tmpturn = []
                      for x in tmp:
                        if x == "<start>":
                          continue
                        if x == "<end>":
                          break
                        tmpturn.append(x)
                      trans.append(tmpturn)
                      ground.append([dev_set.nls[k * args.batch_size + p]])
                    #trans.extend(tmptrans)
                    
                #print(len(trans), ground)
                score = corpus_bleu(ground, trans)
                l = np.mean(losses)
                wandb.log({"Test Accuracy": score, "Test Loss": l})
                if  maxAcc < score:
                    maxAcc = score
                    print("find better score " + str(score) + "and loss is " + str(l))
                    save_model(model)
                    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
                model = model.train()
            index += 1

if __name__ == "__main__":
    #args.lr = float(sys.argv[3])
    #args.seed = int(sys.argv[4])
    
    np.set_printoptions(threshold=sys.maxsize)
    res = {}    
    p = sys.argv[2]
    res[int(sys.argv[1])] = train(int(sys.argv[1]), p)

    open('%sres%d_noline_final.pkl'%(p, int(sys.argv[1])), 'wb').write(pickle.dumps(res))
    '''res = {}
    for i in range(100):
        a = train(i)
        res[i] = a 
        open('res.pkl', 'wb').write(pickle.dumps(res))'''
    #test()
    #trainsearch()
    #combinetrain()



