#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 19-1-18 下午3:49
# @Author  : 林利芳
# @File    : preprocessor.py
import pprint

from sklearn.model_selection import train_test_split
import numpy as np
from config.config import DATA_PKL, VOCAB_PKL, ATEC_NLP_DATA, ADD_ATEC_NLP_DATA, CORPUS_DATA, EXPEND_ATEC_NLP_DATA
from core.utils import save_data, read_csv, load_data
from core.word_embedding import Vocab
import re
import jieba
import collections
from config.synonym import SYNONYM_DICT, SYNONYM_WRONG, PATTERN
import itertools
from config.hyperparams import HyperParams as hp
import sys

try:
	reload(sys)
	sys.setdefaultencoding('utf8')
except:
	pass
jieba.load_userdict(CORPUS_DATA)
PAD = "<PAD>"
UNK = "<UNK>"
PAD2ID = 0
UNK2ID = 0


def extended_corpus(data):
	"""
	扩展语料
	:param data:
	:return:
	"""
	print("同义词替换...\n")
	similar_data = []
	for sub_data in data:
		idx, left_s, right_s, y = sub_data
		idx = idx.replace('\ufeff', '')
		left_s = trim(left_s)
		right_s = trim(right_s)
		data = combine_data(idx, left_s, right_s, y)
		similar_data.extend(data)
	save_expend_data(similar_data, EXPEND_ATEC_NLP_DATA)
	return similar_data


def save_expend_data(data, filename):
	import codecs
	with codecs.open(filename, 'w', encoding='utf-8') as fp:
		for line in data:
			idx, left_x, right_x, y = line
			temp = [idx, ' '.join(left_x), ' '.join(right_x), str(y)]
			fp.writelines('\t'.join(temp) + '\n')


def synonym_replace(sentence):
	"""
	同义词替换
	:param sentence:
	:return:
	"""
	sentences = []
	for word in sentence:
		words = SYNONYM_DICT.get(word, [word])
		sentences.append(words)
	sentences = list(set(itertools.product(*sentences)))
	result = []
	for ii, sub_data in enumerate(sentences):
		sub_data = list(sub_data)
		if sub_data == sentence:
			continue
		result.append(sub_data)
	return result


def combine_data(idx, left_s, right_s, y):
	similar_data = [[idx, left_s, right_s, y]]
	left_sentence = synonym_replace(left_s)
	right_sentence = synonym_replace(right_s)
	left_len, right_len = len(left_sentence), len(right_sentence)
	max_num = max(left_len, right_len)
	if y == '0':
		max_num = 2
	for sub_s in left_sentence[:max_num]:
		temp = [idx, sub_s, right_s, y]
		similar_data.append(temp)
	for sub_s in right_sentence[:max_num]:
		temp = [idx, left_s, sub_s, y]
		similar_data.append(temp)
	return similar_data

	# if y == '1':
	# 	for sub_left_s, sub_right_s in zip(left_sentence[:3], right_sentence[:3]):
	# 		temp = [idx, sub_left_s, sub_right_s, y]
	# 		similar_data.append(temp)
	#
	# if left_len > right_len:
	# 	for sub_left_s, sub_right_s in zip(left_sentence[1:], right_sentence):
	# 		temp = [idx, sub_left_s, sub_right_s, y]
	# 		similar_data.append(temp)
	# elif right_len > left_len:
	# 	for sub_left_s, sub_right_s in zip(left_sentence, right_sentence[1:]):
	# 		temp = [idx, sub_left_s, sub_right_s, y]
	# 		similar_data.append(temp)
	# else:
	# 	data = left_sentence.pop()
	# 	left_sentence.insert(0, data)
	# 	for sub_left_s, sub_right_s in zip(left_sentence, right_sentence):
	# 		temp = [idx, sub_left_s, sub_right_s, y]
	# 		similar_data.append(temp)
	
	return similar_data


def trim(text):
	for rule, region in PATTERN:
		text = rule.sub(region, text)
	sentence = list(jieba.cut(text))
	for ii, word in enumerate(sentence):
		if word in SYNONYM_WRONG:
			word = SYNONYM_WRONG.get(word, word)
			sentence[ii] = word
	return sentence


def build_vocab(text, max_len):
	"""
	构建词库
	:param text: text = [sentence]
	:param max_len: int
	:return:
	"""
	vocab = []
	for sentence in text:
		vocab.extend(sentence)
	count = collections.Counter(vocab).most_common()
	vocab = {v: k + 2 for k, (v, _) in enumerate(count[:hp.vocab_size])}
	vocab[PAD] = PAD2ID
	vocab[UNK] = UNK2ID
	
	v = Vocab()
	v.word2idx = vocab
	v.max_len = max_len
	return v


def process_label(y):
	result = []
	num = 0
	for label in y:
		if label == '1':
			num += 1
		try:
			result.append(int(label))
		except:
			result.append(0)
	print("正样本数\t{}\t负样本数\t{}".format(num, len(y) - num))
	return result


def preprocessor(synonym=False):
	"""数据预处理"""
	if synonym:
		data = read_csv(ATEC_NLP_DATA)
		data.extend(read_csv(ADD_ATEC_NLP_DATA))
		init_num = len(data)
		
		data = extended_corpus(data)
		expand_num = len(data)
		idx, left_x, right_x, y = zip(*data)
		print("初始语料\t{}\t扩展语料\t{}\t新增语料\t{}".format(init_num, expand_num, expand_num - init_num))
	else:
		data = read_csv(EXPEND_ATEC_NLP_DATA)
		idx, left_x, right_x, y = zip(*data)
		
		left_x = [kk.split(' ') for kk in left_x]
		right_x = [kk.split(' ') for kk in right_x]
	y = process_label(y)
	max_len = max(len(kk) for kk in left_x + right_x)
	vocab = build_vocab(left_x + right_x, max_len)
	
	print("最大长度\t{}\t词汇量\t{}".format(max_len, len(vocab.word2idx)))
	
	left_x, left_len = pad_sequence(left_x, vocab, max_len)
	right_x, right_len = pad_sequence(right_x, vocab, max_len)
	
	train_l_x, val_l_x, train_l_len, val_l_len, train_r_x, val_r_x, train_r_len, val_r_len, train_y, val_y = \
		train_test_split(left_x, left_len, right_x, right_len, y, test_size=0.1, random_state=42)
	
	data = {
		"train_l_x": train_l_x,
		"train_r_x": train_r_x,
		"train_l_len": train_l_len,
		"train_r_len": train_r_len,
		"train_y": train_y,
		"val_l_x": val_l_x,
		"val_r_x": val_r_x,
		"val_l_len": val_l_len,
		"val_r_len": val_r_len,
		"val_y": val_y,
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
		seq_len = len(sentence)
		seqs_len.append(len(sentence))
		sentence = [vocab.word2idx.get(kk, UNK2ID) for kk in sentence] + [PAD2ID] * (max_len - seq_len)
		seqs_data.append(sentence[:max_len])
	return seqs_data, seqs_len
