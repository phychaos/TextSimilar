#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-29 下午3:06
# @Author  : 林利芳
# @File    : run.py
import os
from core.load_data import load_train_data, get_feed_dict, print_info, preprocessor
from config.config import checkpoint_dir, VOCAB_PKL
from core.utils import load_data
from model.rnn_siamese import RnnSiameseNetwork
from model.match_pyramid import MatchPyramidNetwork
from config.hyperparams import HyperParams as hp
import tensorflow as tf


def run(network='rnn'):
	checkpoint_file = checkpoint_dir.format(network)
	if not os.path.exists(checkpoint_file):
		os.mkdir(checkpoint_file)
	train_l_x, val_l_x, train_l_len, val_l_len, train_r_x, val_r_x, train_r_len, val_r_len, train_y, val_y = load_train_data()
	vocab = load_data(VOCAB_PKL)
	vocab_size = len(vocab.word2idx)
	
	batch_size = hp.batch_size
	if network == 'rnn':
		model = RnnSiameseNetwork(vocab_size, hp.embedding_size, vocab.max_len, batch_size, True)
	elif network == 'match_pyramid':
		model = MatchPyramidNetwork(vocab_size, hp.embedding_size, vocab.max_len, batch_size, True)
	else:
		return
	sv = tf.train.Supervisor(graph=model.graph, logdir=checkpoint_file, save_model_secs=150)
	with sv.managed_session() as sess:
		print("start training...\n")
		for epoch in range(1, hp.num_epochs + 1):
			if sv.should_stop():
				break
			train_loss = []
			
			for feed_dict, _ in get_feed_dict(model, train_l_x, train_r_x, train_l_len, train_r_len, train_y,
											  batch_size):
				loss, _, acc, gs = sess.run([model.loss, model.train_op, model.acc, model.global_step],
											feed_dict=feed_dict)
				train_loss.append(loss)
				if gs % 1000 == 0:
					print("epoch\t{}\tstep\t{}\tacc\t{}\tloss\t{}\t".format(epoch, gs, acc, round(loss, 4)))
			dev_loss = []
			predicts = []
			for feed_dict, start in get_feed_dict(model, val_l_x, val_r_x, val_l_len, val_r_len, val_y, batch_size):
				loss, gs, pre_y = sess.run([model.loss, model.global_step, model.pre_y], feed_dict=feed_dict)
				dev_loss.append(loss)
				predicts.extend(pre_y[start:])
			print_info(epoch, gs, train_loss, dev_loss, val_y, predicts)


if __name__ == "__main__":
	# preprocessor(True)
	network = 'match_pyramid'  # network = [rnn match_pyramid cnn]
	run(network)
