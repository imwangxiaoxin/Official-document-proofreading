import csv
import math
import gensim
import json
import jieba
import time
import random
import torch


def tensorbetter(model,prelable,right):
    refer=torch.dot(prelable,right).tolist()#/torch.sqrt(torch.dot(prelable,prelable)) #参考投影值
    allcos=torch.matmul(model,prelable).tolist()
    count=0
    for i in range(len(allcos)):
        if allcos[i]>refer:
            count=count+1#参考投影值
    return count/len(allcos)

def gettext(lines,maxlen=10000000000):
    start = time.perf_counter()
    sentences = []
    i = 0
    for l in lines:
        l = l.replace("\\\n", "\n")
        l = l.replace("，", "，\n")
        l = l.replace("。", "。\n")
        l = l.replace("？", "？\n")
        l = l.replace("！", "！\n")
        ls = l.split("\n")
        for ll in ls:
            if len(ll) < 10:
                continue
            sen = jieba.cut(ll, cut_all=False)
            sentences.append(list(sen))
        i = i + 1
        if i % 10000 == 0:
            print("dealing text:" + str(i) + "/" + str(len(lines)))
        if i>maxlen :
            break
    end = time.perf_counter()
    print("text done, time:" + str(end - start) + "s，sentents length:" + str(len(sentences)))
    return sentences




def eraseline(sen): #  ***** -> [**_**,*]
    ds = []
    ts = []
    for bi in range(len(sen)):
        d = [" ", " ", " ", " ", " ", ]
        tgt = sen[bi];
        for i in range(5):
            si = bi - 2 + i
            if si < 0 or si >= len(sen):
                continue
            d[i] = sen[si]
        d[2] = "_"
        ds.append(d)
        ts.append(tgt)
    return ds, ts


def eraseone(sens):  # 把句子中的某个词剪切出来
    data = []
    lable = []
    for sen in sens:
        if len(sen) < 3:
            continue
        d, tgt = eraseline(sen)
        data = data + d
        lable = lable + tgt
    return data, lable


def data2vecs(model, data, lable):
    datavectors = []
    lablevectors = []
    for i in range(0, len(data)):
        d = []
        for ci in data[i]:
            if ci in model.wv:
                d.append(model.wv[ci])
            else:
                d.append(model.wv["、"])
        datavectors.append(d)
        if lable[i] not in model.wv:
            lablevectors.append((model.wv[" "]))
        else:
            lablevectors.append(model.wv[lable[i]])
    return datavectors, lablevectors