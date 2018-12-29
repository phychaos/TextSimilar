#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-29 下午3:07
# @Author  : 林利芳
# @File    : siamese_network.py
import tensorflow as tf
from config.hyperparams import HyperParams as hp
from model.module.modules import embedding, positional_encoding, multihead_attention, feedforward, layer_normalize
from model.module.rnn import ForgetLSTMCell, IndRNNCell


class SiameseNetwork(object):
	def __init__(self, vocab_size, num_tags, is_training=True, seg='rnn'):
		self.vocab_size = vocab_size
		self.num_tags = num_tags
		self.is_training = is_training
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.key = tf.placeholder(tf.int32, shape=(None, hp.max_len), name="key")
			self.value = tf.placeholder(tf.int32, shape=(None, hp.max_len), name="value")
			self.y = tf.placeholder(tf.int32, shape=(None,), name="target")
			self.key_seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])
			self.value_seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])
			self.global_step = tf.train.create_global_step()
			
			key, value = self.siamese(seg)
			self.loss = self.loss_layer(key, value)
			self.train_op = self.optimize()
	
	def siamese(self, seg):
		"""
		孪生网络 transformer + rnn
		:param seg:
		:return:
		"""
		key_value = tf.concat(self.key, self.value, axis=0)
		seq_lens = tf.concat(self.key_seq_lens, self.value_seq_lens)
		# layers embedding multi_head_attention rnn
		embed = embedding(key_value, vocab_size=self.vocab_size, num_units=hp.num_units, scale=True, scope="embed")
		
		output = self.transformer(embed, key_value)
		output = self.rnn_layer(output, seq_lens, seg)
		output = self.attention(embed, output)
		key, value = tf.split(0, 2, output)
		return key, value
	
	def rnn_layer(self, embed, seq_lens, seg):
		"""
		创建双向RNN层
		:param embed:
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
		elif seg == 'F-LSTM':
			fw_cell = ForgetLSTMCell(num_units=hp.num_units)
			bw_cell = ForgetLSTMCell(num_units=hp.num_units)
		elif seg == 'IndRNN':
			fw_cell = IndRNNCell(num_units=hp.num_units)
			bw_cell = IndRNNCell(num_units=hp.num_units)
		else:
			fw_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hp.num_units)
			bw_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hp.num_units)
		# 双向rnn
		(fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(
			fw_cell, bw_cell, embed, sequence_length=seq_lens, dtype=tf.float32)
		
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
	
	def loss_layer(self, key, value):
		"""
		损失函数 L+ = （1-Ew)^2/4  L_ = max(Ex,0)^2
		:param key:
		:param value:
		:return:
		"""
		with tf.name_scope("output"):
			distance = self.similar(key, value)
			loss_p = tf.square(1 - distance) / 4
			mask = tf.sign(distance - hp.margin)
			loss_m = tf.square(mask * distance)
			loss = tf.reduce_sum(self.y * loss_p, (1 - self.y) * loss_m)
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
			query = tf.concat(embed, query, axis=-1)
			query = tf.reshape(query, [-1, 2 * hp.num_units])
			attention = tf.matmul(tf.tanh(tf.matmul(query, w) + b), u)
			attention = tf.reshape(attention, shape=[-1, hp.max_len])
			attention = tf.nn.softmax(attention, axis=-1)
			attention = tf.tile(tf.expand_dims(attention, axis=-1), multiples=[1, 1, hp.num_units])
			output = tf.reduce_sum(attention * query, axis=1)
			output = layer_normalize(output)
			return output
	
	@staticmethod
	def similar(key, value):
		"""
		激活函数 v = s/|s| * s^2/(1+s^2) < 1
		:param key:
		:param value:
		:return:
		"""
		dot_value = tf.reduce_sum(key * value, axis=-1)
		key_sqrt = tf.sqrt(tf.reduce_sum(tf.square(key)), axis=-1)
		value_sqrt = tf.sqrt(tf.reduce_sum(tf.square(value)), axis=-1)
		distance = tf.div(dot_value, key_sqrt * value_sqrt)
		
		return distance
	
	def optimize(self):
		"""
		优化器
		:return:
		"""
		optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
		train_op = optimizer.minimize(self.loss, global_step=self.global_step)
		return train_op
