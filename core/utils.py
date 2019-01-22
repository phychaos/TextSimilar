#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Linlifang
# @file: utils.py
# @time: 18-6-27下午6:13
import csv
import pickle


def read_csv(filename, delimiter='\t'):
	"""
	读取csv
	:param filename:
	:param delimiter:
	:return:
	"""
	with open(filename, 'r') as fp:
		data = [[ii for ii in each] for each in csv.reader(fp, delimiter=delimiter)]
	print(data[:10])
	return data


def load_text(filename):
	"""
	加载数据
	:param filename:
	:return:
	"""
	data = []
	with open(filename, 'r') as fp:
		for idx, line in enumerate(fp):
			line = line.strip('\n')
			tokens = line.split()
			data.append(tokens)
	return data


def load_data(filename):
	"""
	加载词汇信息
	:return:
	"""
	with open(filename, 'rb') as fp:
		data = pickle.load(fp)
	return data


def save_data(filename, data):
	with open(filename, 'wb') as fp:
		pickle.dump(data, fp)

