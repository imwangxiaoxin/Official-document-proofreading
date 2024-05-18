import torch
import torch.nn as nn
import numpy as np
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import Tool
import gensim
import time
import matplotlib.pyplot as plt
import jieba
import seaborn as sns
from Net import Net
import Downloader



LR = 0.001
EPOCH = 10
testlength = 50
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(lines):
    sens=Tool.gettext(lines)
    net = Net()
    model = gensim.models.Word2Vec.load('model/cmb.txt.model')
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    batchsize=50;
    for step in range(EPOCH):#轮次
        lr = LR - step / EPOCH / 2 * LR
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        minloss = 0
        bs=(int)(len(sens)/batchsize)
        for b in range(bs): #批次
            traindata=sens[b*batchsize:(b+1)*batchsize]
            if b % 1000 == 0:
                torch.save(net, "model/step" + str(step) + "-batch" + str(b) + '.pkl')
            sourcedata, sourcelabel = Tool.eraseone(traindata)  # 随机抹掉一个词后的文本原数据
            sdata, slable = Tool.data2vecs(model, sourcedata, sourcelabel)  # 词向量原数据
            data = sdata
            label = slable
            data = torch.Tensor(np.asarray(data))
            data = data.to(device)  # 转换为cuda类型
            label = torch.Tensor(np.asarray(label))
            label = label.to(device)
            pre_label = net(data)
            pre_label = torch.squeeze(pre_label)
            loss = criterion(torch.tanh(pre_label * 2), torch.tanh(label * 2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossnum=loss.item()
            if lossnum < minloss:
                minloss = lossnum

            print ("训练进度:step" + str(step) + "/" + str(EPOCH) + "-batch" + str(b) + "/" + str(bs) + '————train-loss:' + str(round(lossnum, 2)) + "，minloss：" + str(round(minloss, 2)))
        #end = time.perf_counter()
        #print("trained time:" + str(round((end - trainstart) / 60, 1)) + "min, file time:" + str(round((end - filestart) / 60, 1)) + "min")





def check(netmodel):
    net = torch.load("model/" + netmodel)
    net.to(device)
    model = gensim.models.Word2Vec.load('model/cmb.txt.model')
    print("请输入要检查的文本")
    text = input()
    sentenses = Tool.gettext([text])
    modeltensor = torch.tensor(model.wv.vectors)
    modeltensor=modeltensor.to(device)

    d, l = Tool.eraseone(sentenses)
    sdata, slable = Tool.data2vecs(model, d, l)
    tdata = torch.Tensor(np.asarray(sdata))
    tdata = tdata.to(device)
    pre_label = net(tdata)
    for i in range(len(d)):
        if l[i] not in model.wv:  #未知字符暂不处理
            continue
        labletensor = torch.tensor(model.wv[l[i]])
        labletensor=labletensor.to(device)
        betterrate = Tool.tensorbetter( modeltensor, pre_label[i], labletensor)
        print(l[i].ljust(6,"　")+"\t\t："+">".rjust(int(betterrate*100),">")+str(round(betterrate * 100)) + "%")





def betterhist(lines, netmodel):
    traindata = Tool.gettext(lines,200)  # 拆词后的句子数组
    sourcedata, sourcelabel = Tool.eraseone(traindata)  # 随机抹掉一个词后的文本原数据
    model = gensim.models.Word2Vec.load('model/cmb.txt.model')
    sdata, slable = Tool.data2vecs(model, sourcedata, sourcelabel)  # 词向量原数据
    net = torch.load("model/" + netmodel)
    net.to(device)
    data = torch.Tensor(np.asarray(sdata))
    data = data.to(device)
    pre_label = net(data)#.tolist()
    betterlist = []
    modeltensor = torch.tensor(model.wv.vectors)
    modeltensor=modeltensor.to(device)
    for i in range(len(sourcelabel)):
        if sourcelabel[i] not in model.wv:
            continue
        rtlable = torch.tensor(model.wv[sourcelabel[i]])
        rtlable = rtlable.to(device)
        betters = Tool.tensorbetter(modeltensor, pre_label[i], rtlable)
        betterlist.append(betters)
        print("sentece" + str(i) + "/" + str(len(sourcelabel)) + "，"
              + "/".join(sourcedata[i]) + "：" + sourcelabel[i]
              + "\t\t排名：" + str(round(betters * 100, 1)) + "%")
    sns.distplot(betterlist, bins=20, kde=True, color='green')
    plt.title('Histogram with Density Plot')
    plt.xlabel('Shadow')
    plt.ylabel('Frequency')
    plt.show()


def word2vectorready(lines):
    traindata = Tool.gettext(lines)
    # 创建Word2Vec模型
    model = gensim.models.Word2Vec(traindata, vector_size=100, window=5, min_count=1, workers=16)

    # 训练模型
    start = time.perf_counter()
    print("start to train word2vec model")
    model.train(traindata, total_examples=len(traindata), epochs=100)
    end = time.perf_counter()
    print("model is ready, time:" + str(end - start) + "s")
    model.save("model/cmb.model")
    return traindata


if __name__ == '__main__':
    random.seed(int(time.time()))
    Downloader.download()
    with open("cmb.txt", encoding="utf-8") as f:
        lines = f.readlines();
        word2vectorready(lines)
        train(lines)
        betterhist(lines,  "step3-batch0.pkl")
        while True:
            check( "step3-batch0.pkl")

