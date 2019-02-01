#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-29 下午3:07
# @Author  : 林利芳
# @File    : rnn_siamese.py
import tensorflow as tf
from config.hyperparams import CnnParams as hp
from model.module.modules import embedding, positional_encoding, multihead_attention, feedforward, layer_normalize


class CnnSiameseNetwork(object):
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
			
			key, value = self.siamese()
			self.distance, self.pre_y = self.similar(key, value)
			self.accuracy = self.predict()
			self.loss = self.loss_layer()
			self.train_op = self.optimize()
	
	def siamese(self):
		"""
		孪生网络 transformer + rnn
		:return:
		"""
		x = tf.concat([self.left_x, self.right_x], axis=0)
		seq_lens = tf.concat([self.left_seq_lens, self.right_seq_lens], axis=0)
		# layers embedding multi_head_attention rnn
		embed = embedding(x, vocab_size=self.vocab_size, num_units=self.embedding_size, scale=True, scope="embed")
		
		# output = self.transformer(embed, x)
		inputs = tf.expand_dims(embed, -1)
		output = self.cnn_layer(inputs, 1)
		output = tf.expand_dims(output, -1)
		output = self.cnn_layer(output, 2)
		output = self.attention(embed, output)
		key, value = tf.split(output, 2, axis=0)
		return key, value
	
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
	
	def cnn_layer(self, inputs, layer=1):
		"""
		卷积层 卷积核2,3,4,5 激活层relu 池化层 size=2
		:param inputs: batch T * T
		:param layer: batch T * T
		:return:
		"""
		outputs = []
		d_dim, channel = inputs.get_shape().as_list()[-2:]
		for ii, width in enumerate(hp.kernel):
			with tf.variable_scope("cnn_{}_{}_layer".format(layer, ii + 1)):
				weight = tf.Variable(tf.truncated_normal([width, d_dim, channel, hp.channel], stddev=0.1, name='w'))
				bias = tf.get_variable('bias', [hp.channel], initializer=tf.constant_initializer(0.0))
				output = tf.nn.conv2d(inputs, weight, strides=[1, 1, d_dim, 1], padding='SAME')  # batch T T channel
				output = tf.nn.relu(tf.nn.bias_add(output, bias, data_format="NHWC"))
				
				output = tf.reshape(output, shape=[-1, self.max_len, hp.channel])
				outputs.append(output)
		outputs = tf.concat(outputs, axis=-1)
		return outputs
	
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
	
	def loss_layer(self):
		"""
		损失函数 L+ = （1-Ew)^2/4  L_ = max(Ex,0)^2
		:return:
		"""
		y = tf.cast(self.y, tf.float32)
		with tf.name_scope("output"):
			loss_p = tf.square(1 - self.distance) / 4
			mask = tf.sign(tf.nn.relu(self.distance - hp.margin))
			loss_m = tf.square(mask * self.distance)
			loss = tf.reduce_sum(y * loss_p + (1 - y) * loss_m)
			return loss
	
	def attention(self, embed, query):
		"""
		注意力机制
		:param embed:
		:param query:
		:return:
		"""
		output = tf.reduce_mean(query, axis=1)
		return output
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
	
	@staticmethod
	def similar(key, value):
		"""
		cosine(key,value) = key * value/(|key|*|value|)
		:param key:
		:param value:
		:return:
		"""
		dot_value = tf.reduce_sum(key * value, axis=-1)
		key_sqrt = tf.sqrt(tf.reduce_sum(tf.square(key), axis=-1) + hp.eps)
		value_sqrt = tf.sqrt(tf.reduce_sum(tf.square(value), axis=-1) + hp.eps)
		distance = tf.div(dot_value, key_sqrt * value_sqrt, name="similar")
		pre_y = tf.sign(tf.nn.relu(distance - hp.margin))
		pre_y = tf.cast(pre_y, tf.int32, name='pre')
		return distance, pre_y
	
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
