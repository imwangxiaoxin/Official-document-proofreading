使用领导人讲话作为训练素材，训练一个卷积神经网络模型，实现对公文的识别校对
1.训练样本：从人民网的“习近平系列重要讲话数据库”下载文字素材，共1.3万篇，约4千万字
2.分词工作：jieba
3.词袋处理：按照5个字为1个词袋，第3个字挖掉用来预测，***** -> **_**|*
4.词向量化：word2vec，向量共100维，以上素材共统计出8万个词
5.神经网络：卷积神经网络，共3层，均设1000个通道
6.训练过程：使用i7 13700kf + rtx 4070super台式机进行训练约半小时1轮
7.预测过程：约每秒1000个字
