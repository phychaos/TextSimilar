#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 19-1-18 下午5:46
# @Author  : 林利芳
# @File    : main.py
import os
import sys

from core.load_data import get_feed_dict, load_test_data, save_test_result
from config.config import checkpoint_dir, TEST_DATA, TEST_RESULT
from model.match_pyramid import MatchPyramidNetwork
from model.rnn_siamese import RnnSiameseNetwork
from config.hyperparams import HyperParams as hp
import tensorflow as tf
import numpy as np


def test(filename=TEST_DATA, outfile=TEST_RESULT, network='rnn'):
	checkpoint_file = checkpoint_dir.format(network)
	idx, left_x, left_len, right_x, right_len, vocab = load_test_data(filename)
	y = np.ones_like(idx)
	vocab_size = len(vocab.word2idx)
	if network == 'rnn':
		model = RnnSiameseNetwork(vocab_size, hp.embedding_size, vocab.max_len, hp.batch_size, False)
	elif network == 'match_pyramid':
		model = MatchPyramidNetwork(vocab_size, hp.embedding_size, vocab.max_len, hp.batch_size, False)
	else:
		return
	sv = tf.train.Supervisor(graph=model.graph, logdir=checkpoint_file, save_model_secs=0)
	with sv.managed_session() as sess:
		predicts = []
		for feed_dict, start_batch in get_feed_dict(model, left_x, right_x, left_len, right_len, y, hp.batch_size):
			pre_y, distince = sess.run([model.pre_y, model.distance], feed_dict=feed_dict)
			predicts.extend(pre_y[start_batch:])
		save_test_result(outfile, idx, predicts)


if __name__ == "__main__":
	test(sys.argv[1], sys.argv[2])
