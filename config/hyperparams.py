#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-25 上午10:41
# @Author  : 林利芳
# @File    : hyperparams.py


class HyperParams:
	# training
	batch_size = 32  # alias = N
	lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
	seg = 'GRU'  # seg = [GRU,LSTM,IndRNN,F-LSTM]
	# model
	max_len = 50  # Maximum number of words in a sentence. alias = T.
	# Feel free to increase this if you are ambitious.
	min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
	num_units = 100  # alias = C
	embedding_size = 100
	vocab_size = 10000
	num_blocks = 1  # number of encoder/decoder blocks
	num_epochs = 40
	num_heads = 8
	attention_size = 100
	clip = 5
	dropout_rate = 0.1
	eps = 1e-9
	margin = 0.7
	sinusoid = False  # If True, use sinusoid. If false, positional embedding.


class RNNParams:
	# training
	lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
	seg = 'GRU'  # seg = [GRU,LSTM,IndRNN,F-LSTM]
	# model
	max_len = 50  # Maximum number of words in a sentence. alias = T.
	# Feel free to increase this if you are ambitious.
	min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
	num_units = 100  # alias = C
	embedding_size = 100
	num_epochs = 40
	attention_size = 100
	clip = 5
	dropout_rate = 0.1
	eps = 1e-9
	margin = 0.3
	sinusoid = False  # If True, use sinusoid. If false, positional embedding.


class TransformerParams:
	# training
	lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
	seg = 'GRU'  # seg = [GRU,LSTM,IndRNN,F-LSTM]
	# model
	max_len = 50  # Maximum number of words in a sentence. alias = T.
	# Feel free to increase this if you are ambitious.
	min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
	num_units = 100  # alias = C
	embedding_size = 100
	num_epochs = 40
	num_blocks = 6  # number of encoder/decoder blocks
	num_heads = 8
	attention_size = 100
	clip = 5
	dropout_rate = 0.1
	eps = 1e-9
	margin = 0.3
	sinusoid = False  # If True, use sinusoid. If false, positional embedding.


class MatchPyramidParams:
	# training
	lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
	seg = 'GRU'  # seg = [GRU,LSTM,IndRNN,F-LSTM]
	# model
	max_len = 50  # Maximum number of words in a sentence. alias = T.
	# Feel free to increase this if you are ambitious.
	min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
	num_units = 100  # alias = C
	embedding_size = 100
	num_epochs = 40
	attention_size = 100
	clip = 5
	dropout_rate = 0.1
	eps = 1e-9
	margin = 0.3
	channel = 64  # 通道数
	kernel = [2, 3, 4, 5]  # 核大小
	pool_size = 2  # 池化层大小
	dense_size = 100  # 池化层大小
	sinusoid = False  # If True, use sinusoid. If false, positional embedding.
