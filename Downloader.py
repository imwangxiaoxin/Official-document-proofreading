import requests
from lxml import etree
import json
import os

def download():
    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = {'User-Agent': user_agent}
    with open("cmb.txt", "w") as f:
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
