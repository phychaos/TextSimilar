# TextSimilar
短文本相似度
### 孪生网络
[Learning Text Similarity with Siamese Recurrent Networks](http://www.aclweb.org/anthology/W/W16/W16-1617.pdf)  
loss函数  
<img src="logdir/graph/siamese.png">  
---
### match pyramid
[Text Matching as Image Recognition](https://arxiv.org/abs/1602.06359)    
<img src="logdir/graph/match_pyramid">  
---
数据来源于[蚂蚁金融NLP之问题相似度计算](https://dc.cloud.alipay.com/index#/topic/intro?id=8)  
>问题相似度计算，即给定客服里用户描述的两句话，用算法来判断是否表示了相同的语义。
>
>示例：
>
>1. “花呗如何还款” --“花呗怎么还款”：同义问句
>
>2. “花呗如何还款” -- “我怎么还我的花被呢”：同义问句
>
>3. “花呗分期后逾期了如何还款”-- “花呗分期后逾期了哪里还款”：非同义问句
>
>对于例子a，比较简单的方法就可以判定同义；对于例子b，包含了错别字、同义词、词序变换等问题，两个句子乍一看并不类似，想正确判断比较有挑战；对于例子c，两句话很类似，仅仅有一处细微的差别 “如何”和“哪里”，就导致语义不一致。

数据预处理python3 run.py, 在data目录得到data.pkl和vocab.pkl。  
```python
if __name__ == "__main__":
	preprocessor(True)
	network = 'rnn'  # network = [rnn match_pyramid cnn]
	run(network)
```
---
#### siamese 结果
>| network   |  f1    |  
>|-----------|--------|  
>| word-GRU  | 0.5116 |  
>| char-GRU  | 0.4734 |  