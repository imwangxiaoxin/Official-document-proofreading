import time
from datetime import datetime, timedelta
from threading import Thread,Lock
import requests
from lxml import etree
import json
import os

def xi():
    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = {'User-Agent': user_agent}
    with open("xi.txt", "w", encoding='gbk', errors='ignore') as f:
        for page in range(1367):
            print("开始处理第"+str(page)+"页")
            url="http://jhsjk.people.cn/testnew/result?keywords=&isFuzzy=0&searchArea=0&year=0&form=0&type=0&page="\
                +str(page)\
                +"&origin=全部&source=2"
            #url="http://jhsjk.people.cn/result?form=701&else =501"
            response=requests.get(url,headers=headers)
            jobj=json.loads(response.text)
            for a in jobj["list"]:
                url="http://jhsjk.people.cn/article/"+a["article_id"]
                response=requests.get(url,headers=headers)
                et=etree.HTML(response.content)
                body=et.xpath("//p/text()")
                filename=a["article_id"]+"-"+a["input_date"]+"-"+a["title"]
                filename=filename.replace("/"," ")
                f.write(a["title"]+"\n")
                for l in body:
                    f.write(l+"\n")
                print("完成文件："+filename)

def jfjb(): #速度很快
    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = {'User-Agent': user_agent}
    with (open("jfjb.txt", "w", encoding='gbk', errors='ignore') as f):
        start_date = datetime(2018, 1, 1)
        today=datetime.now()
        delta=0
        while True:
            delta=delta+1
            curday = start_date + timedelta(days=delta)
            if curday>today:
                break
            print("开始处理" + str(curday) )
            url = "http://www.81.cn/_szb/jfjb/"+curday.strftime("%Y/%m/%d")+"/index.json"
            response = requests.get(url, headers=headers)
            jobj = json.loads(response.text)
            for paperobj in jobj["paperInfo"]:
                for articleobj in paperobj["xyList"]:
                    content=articleobj["content"]
                    f.write(articleobj["title"] + "\n")
                    if "title2" in articleobj :
                        f.write(articleobj["title2"] + "\n")
                    et = etree.HTML(content)
                    body = et.xpath("//p/text()")
                    for l in body:
                        if len(l)<10 :
                            continue
                        f.write( l + "\n" )
                    print("完成文件：" + articleobj["title"])

user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
headers = {'User-Agent': user_agent}
lock=Lock()
def rmrb():
    lines=[]
    start_date = datetime(2021, 1, 1)
    today=datetime.now()
    delta=0
    while True:
        delta=delta+1
        curday = start_date + timedelta(days=delta)
        if curday>today:
            break
        p = Thread(target=downaday,args=(curday,lines))
        p.start()
        time.sleep(0.5)
        if delta%100 == 0:#lines太大占用内存，定期写入磁盘
            print("开始等待写入")
            time.sleep(10)
            lock.acquire()
            print("正在写入，共"+str(len(lines))+"行")
            with (open("rmrb.txt", "a", encoding='gbk', errors='ignore') as f):#需提前新建一个空的文件，让它来追加
                for l in lines :
                    if len(l)<10:
                        continue
                    f.write(l+"\n")
            lines=[]
            print("写入结束，继续下载")
            lock.release()
def downaday(curday,lines):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>开始处理" + str(curday) )
    url = "http://paper.people.com.cn/rmrb/html/"+curday.strftime("%Y-%m/%d")+"/nbs.D110000renmrb_01.htm"
    response = requests.get(url, headers=headers)
    dayobj = etree.HTML(response.content)
    paperlinks = dayobj.xpath("//div[@class='swiper-container']/div[@class='swiper-slide']/a/@href")
    for apaper in paperlinks :
        url = "http://paper.people.com.cn/rmrb/html/"+curday.strftime("%Y-%m/%d")+"/"+apaper
        response = requests.get(url, headers=headers)
        newsobj = etree.HTML(response.content)
        newslink = newsobj.xpath("//div[@class='news']/ul[@class='news-list']/li/a/@href")
        for anews in newslink :
            url = "http://paper.people.com.cn/rmrb/html/"+curday.strftime("%Y-%m/%d")+"/"+anews
            response = requests.get(url, headers=headers)
            articleobj = etree.HTML(response.content)
            title = articleobj.xpath("//div[@class='article']/h1/text()")
            lock.acquire()
            print("lines count :"+str(len(lines))+ "正在处理：",end="")
            if len(title) > 0 :
                lines.append(title[0] )
                print(title[0])
            paragraghs = articleobj.xpath("//div[@class='article']/div[@id='ozoom']/p/text()")
            lines.extend(paragraghs)
            lock.release()
