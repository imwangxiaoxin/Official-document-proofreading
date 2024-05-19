# Official-document-proofreading 公文校对工具

使用领导人讲话作为训练素材，训练一个卷积神经网络模型，实现对公文的识别校对
### 1.训练样本
从人民网的“习近平系列重要讲话数据库”下载文字素材，共1.3万篇，约4千万字。整理好的[文本（2024.4月下载）](https://pan.baidu.com/s/1AyvUY6tOTP3QMfd_RkGlzw),提取码：pqo5
### 2.分词工作
使用jieba这个工具进行自动分词，效果非常好，基本符合汉语的说话习惯
### 3.词袋处理
按照5个字为1个词袋，第3个字挖掉用来预测，***** -> **_**|*
### 4.词向量化
使用word2vec模型对汉语词汇进行向量化，以上素材共统计出8万个词，向量设置为100维足够了。训练好的[词向量模型](https://pan.baidu.com/s/1p6BMuWYajBgeIM-IvzpJ9Q),提取码：3wht
### 5.神经网络
设计3层全连接卷积神经网络，均设1000个通道
### 6.训练过程
将训练材料分批进行训练，使用i7 13700kf + rtx 4070super台式机进行训练约半小时1轮，1轮就有非常好的效果，建议10轮。我训练了2轮，训练好的[模型](https://pan.baidu.com/s/1ddAaKqBr3QB2kvcZPozcbw),提取码：xowd
### 7.预测过程
将预测出的词向量与所有词向量进行比对，按照重合度（向量投影）进行排序，看所给的词所在排名，预测速率约每秒1000个字
### 8.预测效果
从结果上看，约95%以上的预测结果排名在前1%以上，基本可以找出所有低级语法错误，但是仍有部分低频次的词语被误伤
```cmd
请输入要检查的文本
习近平总书记2023年10月在江西考察时指出：“要找准定位、明确方向，整合资源、精准发力，加快传统产业改造升级，加快战略性新兴产业发展壮大，积极部署未来产业，努力构建体现江西特色和优势的现代化产业体系。”
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\Admin\AppData\Local\Temp\jieba.cache
Loading model cost 0.319 seconds.
Prefix dict has been built successfully.
text done, time:0.32001339999987977s，sentents length:5
 / /_/总书记/2023	|习近平　　　		排名：>0.0%
 /习近平/_/2023/年	|总书记　　　		排名：>0.0%
习近平/总书记/_/年/10	|2023　　		排名：>0.0%
总书记/2023/_/10/月	|年　　　　　		排名：>0.0%
2023/年/_/月/在	|10　　　　		排名：>0.0%
年/10/_/在/江西	|月　　　　　		排名：>0.0%
10/月/_/江西/考察	|在　　　　　		排名：>0.0%
月/在/_/考察/时	|江西　　　　		排名：>0.1%
在/江西/_/时/指出	|考察　　　　		排名：>0.0%
江西/考察/_/指出/：	|时　　　　　		排名：>0.0%
考察/时/_/：/“	|指出　　　　		排名：>0.0%
时/指出/_/“/要	|：　　　　　		排名：>0.0%
指出/：/_/要/找准	|“　　　　　		排名：>0.0%
：/“/_/找准/定位	|要　　　　　		排名：>0.0%
“/要/_/定位/、	|找准　　　　		排名：>0.0%
要/找准/_/、/明确	|定位　　　　		排名：>0.1%
找准/定位/_/明确/方向	|、　　　　　		排名：>0.0%
定位/、/_/方向/，	|明确　　　　		排名：>0.4%
、/明确/_/，/ 	|方向　　　　		排名：>0.0%
 / /_/资源/、	|整合　　　　		排名：>0.1%
 /整合/_/、/精准	|资源　　　　		排名：>0.0%
整合/资源/_/精准/发力	|、　　　　　		排名：>0.0%
资源/、/_/发力/，	|精准　　　　		排名：>0.0%
、/精准/_/，/ 	|发力　　　　		排名：>0.0%
 / /_/传统产业/改造	|加快　　　　		排名：>0.0%
 /加快/_/改造/升级	|传统产业　　		排名：>0.7%
加快/传统产业/_/升级/，	|改造　　　　		排名：>0.7%
传统产业/改造/_/，/ 	|升级　　　　		排名：>0.3%
 / /_/战略性/新兴产业	|加快　　　　		排名：>0.0%
 /加快/_/新兴产业/发展壮大	|战略性　　　		排名：>0.1%
加快/战略性/_/发展壮大/，	|新兴产业　　		排名：>0.0%
战略性/新兴产业/_/，/ 	|发展壮大　　		排名：>0.6%
 / /_/构建/体现	|努力　　　　		排名：>0.1%
 /努力/_/体现/江西	|构建　　　　		排名：>0.0%
努力/构建/_/江西/特色	|体现　　　　		排名：>0.5%
构建/体现/_/特色/和	|江西　　　　		排名：>>>>>5.9%
体现/江西/_/和/优势	|特色　　　　		排名：>0.0%
江西/特色/_/优势/的	|和　　　　　		排名：>0.0%
特色/和/_/的/现代化	|优势　　　　		排名：>0.0%
和/优势/_/现代化/产业	|的　　　　　		排名：>0.0%
优势/的/_/产业/体系	|现代化　　　		排名：>0.0%
的/现代化/_/体系/。	|产业　　　　		排名：>0.0%
现代化/产业/_/。/ 	|体系　　　　		排名：>0.0%
```
