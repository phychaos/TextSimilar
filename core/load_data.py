#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 19-1-18 下午4:08
# @Author  : 林利芳
# @File    : load_data.py
from config.config import DATA_PKL, VOCAB_PKL
from core.preprocessor import preprocessor, pad_sequence, trim
from core.utils import load_data, read_csv
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score


def gen_batch_data(l_x, r_x, l_len, r_len, y, batch_size):
	"""
	生成batch数据
	:param l_x:
	:param r_x:
	:param l_len:
	:param r_len:
	:param y:
	:param batch_size:
	:return:
	"""
	data_size = len(y)
	num_batch = data_size // batch_size + 1
	
	for ii in range(num_batch):
		start, end = ii * batch_size, (ii + 1) * batch_size
		start_batch = 0
		if end > data_size:
			start_batch = end - data_size
			start, end = data_size - batch_size, data_size
		l_x_batch = l_x[start:end]
		r_x_batch = r_x[start:end]
		l_len_batch = l_len[start:end]
		r_len_batch = r_len[start:end]
		y_batch = y[start:end]
		yield l_x_batch, r_x_batch, l_len_batch, r_len_batch, y_batch, start_batch


def load_train_data():
	data = load_data(DATA_PKL)
	train_l_x, val_l_x, train_l_len, val_l_len, train_r_x, val_r_x, train_r_len, val_r_len, train_y, val_y = \
		data['train_l_x'], data['val_l_x'], data['train_l_len'], data['val_l_len'], data['train_r_x'], data[
			'val_r_x'], data['train_r_len'], data['val_r_len'], data['train_y'], data['val_y']
	train_l_x = np.array(train_l_x)
	val_l_x = np.array(val_l_x)
	train_l_len = np.array(train_l_len)
	val_l_len = np.array(val_l_len)
	train_r_x = np.array(train_r_x)
	val_r_x = np.array(val_r_x)
	train_r_len = np.array(train_r_len)
	val_r_len = np.array(val_r_len)
	train_y = np.array(train_y)
	val_y = np.array(val_y)
	return train_l_x, val_l_x, train_l_len, val_l_len, train_r_x, val_r_x, train_r_len, val_r_len, train_y, val_y


def get_feed_dict(model, l_x, r_x, l_len, r_len, y, batch_size):
	"""
	生成feed_dict
	:param model:
	:param l_x:
	:param r_x:
	:param l_len:
	:param r_len:
	:param y:
	:param batch_size:
	:return:
	"""
	for l_x_batch, r_x_batch, l_len_batch, r_len_batch, y_batch, start_batch in gen_batch_data(
			l_x, r_x, l_len, r_len, y, batch_size):
		feed_dict = {
			model.left_x: l_x_batch,
			model.right_x: r_x_batch,
			model.y: y_batch,
			model.left_seq_lens: l_len_batch,
			model.right_seq_lens: r_len_batch
		}
		yield feed_dict, start_batch


def print_info(epoch, step, train_loss, dev_loss, y, pre_y):
	loss = round(float(np.mean(train_loss)), 3)
	val_loss = round(float(np.mean(dev_loss)), 3)
	f1 = round(f1_score(y, pre_y), 4)
	recall = round(recall_score(y, pre_y), 4)
	precision = round(precision_score(y, pre_y), 4)
	print('**************************************************')
	print("epoch\t{}\tstep\t{}\ttrain_loss\t{}\tdev_loss\t{}\t".format(epoch, step, loss, val_loss))
	print("precision\t{}\trecall\t{}\tf1\t{}\n\n".format(precision, recall, f1))


def load_test_data(filename):
	vocab = load_data(VOCAB_PKL)
	max_len = vocab.max_len
	data = read_csv(filename)
	data = [kk[:3] for kk in data]
	idx, left_x, right_x = zip(*data)
	
	left_x = [trim(kk) for kk in left_x]
	right_x = [trim(kk) for kk in right_x]
	
	left_x, left_len = pad_sequence(left_x, vocab, max_len)
	right_x, right_len = pad_sequence(right_x, vocab, max_len)
	
	return idx, left_x, left_len, right_x, right_len, vocab


def save_test_result(filename, idx, predicts):
	import codecs
	with codecs.open(filename, 'w', encoding='utf-8') as fp:
		for _id, pre in zip(idx, predicts):
			fp.writelines('{}\t{}\n'.format(_id, pre))
