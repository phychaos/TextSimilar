#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 19-1-18 下午4:08
# @Author  : 林利芳
# @File    : load_data.py
from config.config import DATA_PKL, VOCAB_PKL
from core.preprocessor import preprocessor
from core.utils import load_data
import numpy as np


def gen_batch_data(l_x, r_x, l_len, r_len, y, batch_size, is_training=True):
	"""
	生成batch数据
	:param l_x:
	:param r_x:
	:param l_len:
	:param r_len:
	:param y:
	:param batch_size:
	:param is_training:训练集，打乱顺序
	:return:
	"""
	if is_training:
		np.random.seed(10)
		shuffle_indices = np.random.permutation(np.arange(len(l_x)))
		l_x = l_x[shuffle_indices]
		r_x = r_x[shuffle_indices]
		l_len = l_len[shuffle_indices]
		r_len = r_len[shuffle_indices]
		y = y[shuffle_indices]
	data_size = len(y)
	num_batch = data_size // batch_size + 1
	
	for ii in range(num_batch):
		start, end = ii * batch_size, (ii + 1) * batch_size
		if end > len(y):
			start, end = data_size - batch_size, data_size
		l_x_batch = l_x[start:end]
		r_x_batch = r_x[start:end]
		l_len_batch = l_len[start:end]
		r_len_batch = r_len[start:end]
		y_batch = y[start:end]
		yield l_x_batch, r_x_batch, l_len_batch, r_len_batch, y_batch


def load_train_data(is_preprocessor=False):
	if is_preprocessor:
		data, vocab = preprocessor()
	else:
		data = load_data(DATA_PKL)
		vocab = load_data(VOCAB_PKL)
	train_l_x, val_l_x, train_l_len, val_l_len, train_r_x, val_r_x, train_r_len, val_r_len, train_y, val_y, max_len = \
		data['train_l_x'], data['val_l_x'], data['train_l_len'], data['val_l_len'], data['train_r_x'], data[
			'val_r_x'], data['train_r_len'], data['val_r_len'], data['train_y'], data['val_y'], data['max_len']
	return train_l_x, val_l_x, train_l_len, val_l_len, train_r_x, val_r_x, train_r_len, val_r_len, train_y, val_y, max_len, vocab


def load_vocab_seq_len():
	data = load_data(DATA_PKL)
	vocab = load_data(VOCAB_PKL)
	max_len = data['max_len']
	return max_len, vocab


def get_feed_dict(model, l_x, r_x, l_len, r_len, y, batch_size, is_training=True):
	"""
	生成feed_dict
	:param model:
	:param l_x:
	:param r_x:
	:param l_len:
	:param r_len:
	:param y:
	:param batch_size:
	:param is_training:
	:return:
	"""
	for l_x_batch, r_x_batch, l_len_batch, r_len_batch, y_batch in gen_batch_data(
			l_x, r_x, l_len, r_len, y, batch_size, is_training=is_training):
		feed_dict = {
			model.left_x: l_x_batch,
			model.right_x: r_x_batch,
			model.y: y_batch,
			model.left_seq_lens: l_len_batch,
			model.right_seq_lens: r_len_batch
		}
		yield feed_dict


def print_info(epoch, step, train_loss, dev_loss, dev_acc):
	loss = round(float(np.mean(train_loss)), 3)
	val_loss = round(float(np.mean(dev_loss)), 3)
	acc = round(float(np.mean(dev_acc)), 4)
	print('**************************************************')
	print("epoch\t{}\tstep\t{}".format(epoch, step))
	print("train_loss\t{}\tdev_loss\t{}\tacc\t{}\n\n".format(loss, val_loss, acc))
