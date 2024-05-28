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

def train(lines): #lines 一行是一段：xxxx,xxxx。xxxx,xxxx。
    #net = Net()
    net = torch.load("model/step0-batch37000.pkl")
    model = gensim.models.Word2Vec.load('model/cmb.model')
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    batchsize=250;#根据显卡性能决定，12G显存大概是100
    sens=Tool.lines2sens(lines) #把每段文字拆成一句一句的:xxxx,
    for step in range(EPOCH):#轮次
        lr = LR - step / EPOCH / 2 * LR
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        minloss = 0
        batchcount=(int)(len(sens)/batchsize)
        for b in range(batchcount): #批次
            batchsens=sens[b*batchsize:(b+1)*batchsize]
            traindata=Tool.sens2words(batchsens)#xx|xx|xx|xx|。
            if b % 1000 == 0:
                torch.save(net, "model/step" + str(step) + "-batch" + str(b) + '.pkl')
            sourcedata, sourcelabel = Tool.eraseone(traindata)  # 随机抹掉一个词后的文本原数据
            sourcedata=sourcedata[0:batchsize*10]#截断多余数据，防止cuda out of memory
            sourcelabel=sourcelabel[0:batchsize*10]
            print("datalength:"+str(len(sourcelabel)))
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
            print ("训练进度:step" + str(step) + "/" + str(EPOCH) + "-batch" + str(b) + "/" + str(batchcount) + '————train-loss:' + str(round(lossnum, 2)) + "，minloss：" + str(round(minloss, 2)))
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
        print("/".join(d[i])+"\t|"+ l[i].ljust(6,"　")+"\t\t排名："+">".rjust(int(betterrate*100),">")+str(round(betterrate * 100,1)) + "%")


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
    #Downloader.jfjb()
    #Downloader.xi()
    #Downloader.rmrb()
    with open("text/xi.txt", encoding="gbk") as f:
        xi = f.readlines();
    with open("text/jfjb.txt", encoding="gbk") as f:
        jfjb = f.readlines();
    with open("text/rmrb.txt", encoding="gbk") as f:
        rmrb = f.readlines();
    cmb= xi+jfjb+rmrb
    #word2vectorready(cmb)
    train(cmb)
    #betterhist(cmb,  "step3-batch0.pkl")
    #while True:
    #    check( "step3-batch0.pkl")

