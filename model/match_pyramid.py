#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 19-1-25 上午11:17
# @Author  : 林利芳
# @File    : match_pyramid.py
import tensorflow as tf
from config.hyperparams import MatchPyramidParams as hp
from model.module.modules import embedding, positional_encoding, multihead_attention, feedforward, layer_normalize


class MatchPyramidNetwork(object):
	def __init__(self, vocab_size, embedding_size, max_len, batch_size, is_training=True):
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.max_len = max_len
		self.is_training = is_training
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.left_x = tf.placeholder(tf.int32, shape=(batch_size, max_len), name="left_x")
			self.right_x = tf.placeholder(tf.int32, shape=(batch_size, max_len), name="right_x")
			self.y = tf.placeholder(tf.int32, shape=(batch_size,), name="target")
			self.left_seq_lens = tf.placeholder(dtype=tf.int32, shape=[batch_size])
			self.right_seq_lens = tf.placeholder(dtype=tf.int32, shape=[batch_size])
			self.global_step = tf.train.create_global_step()
			
			outputs = self.match_pyramid()
			outputs, self.pre_y = self.multi_dense_layer(outputs)
			self.acc = self.predict()
			self.loss = self.loss_layer(outputs)
			self.train_op = self.optimize()
	
	def match_pyramid(self):
		"""
		pyramid
		:return:
		"""
		left_embed = embedding(self.left_x, vocab_size=self.vocab_size, num_units=self.embedding_size, scale=True,
							   scope="left_embed")
		right_embed = embedding(self.right_x, vocab_size=self.vocab_size, num_units=self.embedding_size, scale=True,
								scope="right_embed")
		outputs = self.match_text(left_embed, right_embed)
		outputs = self.cnn_layer(outputs, 1)
		outputs = self.cnn_layer(outputs, 2)
		return outputs
	
	@staticmethod
	def match_text(left_embed, right_embed):
		"""
		文本匹配 cosine dot binary
		:param left_embed: 词嵌入 batch * T * D
		:param right_embed: 词嵌入 batch * T * D
		:return:
		"""
		with tf.variable_scope("match-text"):
			dot_output = tf.matmul(left_embed, tf.transpose(right_embed, [0, 2, 1]))  # batch * T * T
			left_norm = tf.sqrt(tf.matmul(left_embed, tf.transpose(left_embed, [0, 2, 1]))+hp.eps)
			right_norm = tf.sqrt(tf.matmul(right_embed, tf.transpose(right_embed, [0, 2, 1]))+hp.eps)
			cosine_outputs = tf.div(dot_output, left_norm * right_norm)
			binary_outputs = tf.cast(tf.equal(cosine_outputs, 1), tf.float32)
			dot_output = tf.expand_dims(dot_output, axis=-1)
			cosine_outputs = tf.expand_dims(cosine_outputs, axis=-1)
			binary_outputs = tf.expand_dims(binary_outputs, axis=-1)
			
			outputs = tf.concat([dot_output, cosine_outputs, binary_outputs], axis=-1)
		print(outputs.get_shape().as_list())
		return dot_output
	
	@staticmethod
	def cnn_layer(inputs, layer=1):
		"""
		卷积层 卷积核2,3,4,5 激活层relu 池化层 size=2
		:param inputs: batch T * T
		:param layer: batch T * T
		:return:
		"""
		outputs = []
		channel = inputs.get_shape().as_list()[-1]
		for ii, width in enumerate(hp.kernel):
			with tf.variable_scope("cnn_{}_{}_layer".format(layer, ii + 1)):
				weight = tf.Variable(tf.truncated_normal([width, width, channel, hp.channel], stddev=0.1, name='w'))
				bias = tf.get_variable('bias', [hp.channel], initializer=tf.constant_initializer(0.0))
				output = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME')  # batch T T channel
				output = tf.nn.relu(tf.nn.bias_add(output, bias, data_format="NHWC"))
				pool = tf.nn.max_pool(output, ksize=[1, hp.pool_size, hp.pool_size, 1], strides=[1, 1, 1, 1],
									  padding='VALID')
				outputs.append(pool)
		outputs = tf.concat(outputs, axis=-1)
		return outputs
	
	@staticmethod
	def multi_dense_layer(inputs):
		"""
		多层感知机 T*T*channel -> dense_size ->2
		:param inputs: batch T T channel
		:return:
		"""
		_, width, height, channel = inputs.get_shape().as_list()
		size = width * height * channel
		inputs = tf.reshape(inputs, shape=[-1, size])
		with tf.variable_scope("dense_layer"):
			w = tf.get_variable(name='w', dtype=tf.float32, shape=[size, hp.dense_size])
			b = tf.get_variable(name='b', dtype=tf.float32, shape=[hp.dense_size])
			outputs = layer_normalize(tf.matmul(inputs, w) + b, )
		
		with tf.variable_scope("logit_layer"):
			w = tf.get_variable(name='w', dtype=tf.float32, shape=[hp.dense_size, 2])
			b = tf.get_variable(name='b', dtype=tf.float32, shape=[2])
			outputs = tf.nn.softmax(tf.matmul(outputs, w) + b, axis=-1)
		pre_y = tf.cast(tf.argmax(outputs, axis=-1), dtype=tf.int32)
		return outputs, pre_y
	
	def rnn_layer(self, inputs, seq_lens, seg=hp.seg):
		"""
		创建双向RNN层
		:param inputs:
		:param seq_lens:
		:param seg: LSTM GRU F-LSTM, IndRNN
		:return:
		"""
		if seg == 'LSTM':
			fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hp.num_units)
			bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hp.num_units)
		
		elif seg == 'GRU':
			fw_cell = tf.nn.rnn_cell.GRUCell(num_units=hp.num_units)
			bw_cell = tf.nn.rnn_cell.GRUCell(num_units=hp.num_units)
		else:
			fw_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hp.num_units)
			bw_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hp.num_units)
		# 双向rnn
		(fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(
			fw_cell, bw_cell, inputs, sequence_length=seq_lens, dtype=tf.float32)
		# 合并双向rnn的output batch_size * max_seq * (hidden_dim*2)
		output = tf.add(fw_output, bw_output)
		return output
	
	def transformer(self, embed, value):
		with tf.variable_scope("Transformer_Encoder"):
			# Positional Encoding
			embed += positional_encoding(value, num_units=hp.num_units, zero_pad=False, scale=False, scope="post")
			# Dropout
			output = self.multi_head_block(embed)
			return output
	
	def multi_head_block(self, query, causality=False):
		"""
		多头注意力机制
		:param query:
		:param causality:
		:return:
		"""
		for i in range(hp.num_blocks):
			with tf.variable_scope("num_blocks_{}".format(i)):
				# multi head Attention ( self-attention)
				query = multihead_attention(
					queries=query, keys=query, num_units=hp.num_units, num_heads=hp.num_heads,
					dropout_rate=hp.dropout_rate, is_training=self.is_training, causality=causality,
					scope="self_attention")
				# Feed Forward
				query = feedforward(query, num_units=[4 * hp.num_units, hp.num_units])
		return query
	
	def loss_layer(self, inputs):
		"""
		损失函数 L+ = （1-Ew)^2/4  L_ = max(Ex,0)^2
		:return:
		"""
		y = tf.cast(self.y, tf.float32)
		with tf.name_scope("loss_layer"):
			loss_p = y * tf.log(tf.clip_by_value(inputs[:, -1], hp.eps, 1.0))
			loss_m = (1 - y) * tf.log(tf.clip_by_value(inputs[:, 0], hp.eps, 1.0))
			loss = -tf.reduce_sum(loss_p + loss_m)
			return loss
	
	def attention(self, embed, query):
		"""
		注意力机制
		:param embed:
		:param query:
		:return:
		"""
		with tf.name_scope("attention"):
			w = tf.get_variable(name="attention_w", shape=[2 * hp.num_units, hp.attention_size], dtype=tf.float32)
			b = tf.get_variable(name="attention_b", shape=[hp.attention_size], dtype=tf.float32)
			u = tf.get_variable(name="attention_u", shape=[hp.attention_size, 1], dtype=tf.float32)
			value = tf.concat([embed, query], axis=-1)
			value = tf.reshape(value, [-1, 2 * hp.num_units])
			attention = tf.matmul(tf.tanh(tf.matmul(value, w) + b), u)
			attention = tf.reshape(attention, shape=[-1, self.max_len])
			attention = tf.nn.softmax(attention, axis=-1)
			attention = tf.tile(tf.expand_dims(attention, axis=-1), multiples=[1, 1, hp.num_units])
			
			output = tf.reduce_sum(attention * query, axis=1)
			output = layer_normalize(output)
			return output
	
	def predict(self):
		correct_predictions = tf.equal(self.pre_y, self.y)
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
		return accuracy
	
	def optimize(self):
		"""
		优化器
		:return:
		"""
		optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
		train_op = optimizer.minimize(self.loss, global_step=self.global_step)
		return train_op
