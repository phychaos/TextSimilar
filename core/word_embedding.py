#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 19-1-17 下午5:08
# @Author  : 林利芳
# @File    : word_embedding.py
import numpy as np

from config.config import WORD2VEC_DATA


class Vocab(object):
	def __init__(self):
		self.word2vec = []
		self.word2idx = {'<PAD>': 0, '<UNK>': 1}
	
	def add_word(self, word, vector):
		self.word2idx[word] = len(self.word2idx)
		self.word2vec.append(vector)
	
	def load_word_vectors(self):
		with open(WORD2VEC_DATA, 'r') as f:
			vocab_size, embedding_dim = [int(_) for _ in f.readline().strip().split(' ')]
			self.word2vec = [[0.0] * embedding_dim]
			self.word2vec.append(np.random.uniform(-0.25, 0.25, embedding_dim).round(6).tolist())
			lines = f.readlines()
			for line in lines:
				word, vector = line.strip().split(' ', 1)
				self.add_word(word, [float(_) for _ in vector.split(' ')])
		self.word2vec = np.array(self.word2vec).astype(np.float32)
