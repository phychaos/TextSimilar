# TextSimilar
短文本相似度
### 孪生网络
[Learning Text Similarity with Siamese Recurrent Networks](http://www.aclweb.org/anthology/W/W16/W16-1617.pdf)  
loss函数  
<img src="logdir/graph/siamese.png">  
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
def run(is_preprocessor=False):
	train_l_x, val_l_x, train_l_len, val_l_len, train_r_x, val_r_x, train_r_len, val_r_len, train_y, val_y, max_len, vocab = load_train_data(
		is_preprocessor)
	vocab_size = len(vocab.word2idx)
	
	model = SiameseNetwork(vocab_size, hp.embedding_size, max_len, hp.batch_size, is_training=True, seg=hp.seg)
	sv = tf.train.Supervisor(graph=model.graph, logdir=checkpoint_dir, save_model_secs=200)
	with sv.managed_session() as sess:
		print("start training...\n")
		for epoch in range(1, hp.num_epochs + 1):
			if sv.should_stop():
				break
			train_loss = []
			
			for feed_dict, _ in get_feed_dict(model, train_l_x, train_r_x, train_l_len, train_r_len, train_y,
											  hp.batch_size):
				loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)
				train_loss.append(loss)
			dev_loss = []
			predicts = []
			for feed_dict, start_batch in get_feed_dict(model, val_l_x, val_r_x, val_l_len, val_r_len, val_y,
														hp.batch_size, False):
				loss, gs, pre_y = sess.run([model.loss, model.global_step, model.pre_y], feed_dict=feed_dict)
				dev_loss.append(loss)
				predicts.extend(pre_y[start_batch:])
			
			print_info(epoch, gs, train_loss, dev_loss, val_y, predicts)
			
if __name__ == "__main__":
	run(True)
```
---
#### siamese 结果
>| network   | precision | recall |  f1    |  
>|-----------|-----------|--------|--------|  
>| char-LSTM |	0.3748   | 0.7201 | 0.493  |  
>| char-GRU  |	0.3552   | 0.7094 | 0.4734 |  