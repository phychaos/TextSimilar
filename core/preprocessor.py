#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 19-1-18 下午3:49
# @Author  : 林利芳
# @File    : preprocessor.py
from sklearn.model_selection import train_test_split
import numpy as np
from config.config import DATA_PKL, VOCAB_PKL, ATEC_NLP_DATA, ADD_ATEC_NLP_DATA
from core.utils import save_data, read_csv, load_data
from core.word_embedding import Vocab
import re

PAD = "<PAD>"
UNK = "<UNK>"
PAD2ID = 0
UNK2ID = 0


def trim(text, english=False):
	rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
	if english:
		rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5 ]")
	text = rule.sub('', text)
	sentence = [word for word in text]
	return sentence


def build_vocab(text):
	"""
	构建词库
	:param text: text = [sentence]
	:return:
	"""
	vocab = set()
	for sent in text:
		vocab = vocab | set(sent)
	vocab = {v: k + 2 for k, v in enumerate(list(vocab))}
	vocab[PAD] = PAD2ID
	vocab[UNK] = UNK2ID
	
	v = Vocab()
	v.word2idx = vocab
	return v


def process_label(y):
	result = []
	for label in y:
		try:
			result.append(int(label))
		except:
			result.append(0)
	return result


def preprocessor():
	"""数据预处理"""
	data = read_csv(ATEC_NLP_DATA)
	data.extend(read_csv(ADD_ATEC_NLP_DATA))
	idx, left_x, right_x, y = zip(*data)
	max_len = max([len(x) for x in left_x + right_x])
	y = process_label(y)
	vocab = build_vocab(left_x + right_x)
	
	left_x, left_len = pad_sequence(left_x, vocab, max_len)
	right_x, right_len = pad_sequence(right_x, vocab, max_len)
	
	train_l_x, val_l_x, train_l_len, val_l_len, train_r_x, val_r_x, train_r_len, val_r_len, train_y, val_y = \
		train_test_split(left_x, left_len, right_x, right_len, y, test_size=0.1, random_state=42)
	
	data = {
		"train_l_x": np.array(train_l_x),
		"train_r_x": np.array(train_r_x),
		"train_l_len": np.array(train_l_len),
		"train_r_len": np.array(train_r_len),
		"train_y": np.array(train_y),
		"val_l_x": np.array(val_l_x),
		"val_r_x": np.array(val_r_x),
		"val_l_len": np.array(val_l_len),
		"val_r_len": np.array(val_r_len),
		"val_y": np.array(val_y),
		"max_len": max_len,
	}
	save_data(DATA_PKL, data)
	save_data(VOCAB_PKL, vocab)
	return data, vocab


def pad_sequence(data, vocab, max_len):
	"""
	补全数据
	:param data:
	:param vocab:
	:param max_len:
	:return:
	"""
	seqs_data = []
	seqs_len = []
	for sentence in data:
		sentence = trim(sentence)
		seq_len = len(sentence)
		seqs_len.append(len(sentence))
		sentence = [vocab.word2idx.get(kk, UNK2ID) for kk in sentence] + [PAD2ID] * (max_len - seq_len)
		seqs_data.append(sentence[:max_len])
	return seqs_data, seqs_len



