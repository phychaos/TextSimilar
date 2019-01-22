#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-29 下午3:06
# @Author  : 林利芳
# @File    : run.py

from core.load_data import load_train_data, get_feed_dict, print_info
from config.config import checkpoint_dir
from model.rnn_siamese import SiameseNetwork
from config.hyperparams import HyperParams as hp
import tensorflow as tf

checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)


def run(is_preprocessor=False):
	train_l_x, val_l_x, train_l_len, val_l_len, train_r_x, val_r_x, train_r_len, val_r_len, train_y, val_y, max_len, vocab = load_train_data(
		is_preprocessor)
	vocab_size = len(vocab.word2idx)
	
	model = SiameseNetwork(vocab_size, hp.embedding_size, max_len, hp.batch_size, is_training=True, seg=hp.seg)
	sv = tf.train.Supervisor(graph=model.graph, logdir=checkpoint_dir, save_model_secs=200)
	with sv.managed_session() as sess:
		print("start training...\n")
		for epoch in range(1, hp.num_epochs + 1):
			if sv.should_stop():
				break
			train_loss = []
			
			for feed_dict, _ in get_feed_dict(model, train_l_x, train_r_x, train_l_len, train_r_len, train_y,
											  hp.batch_size):
				loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)
				train_loss.append(loss)
			dev_loss = []
			predicts = []
			for feed_dict, start_batch in get_feed_dict(model, val_l_x, val_r_x, val_l_len, val_r_len, val_y,
														hp.batch_size, False):
				loss, gs, pre_y = sess.run([model.loss, model.global_step, model.pre_y], feed_dict=feed_dict)
				dev_loss.append(loss)
				predicts.extend(pre_y[start_batch:])
			
			print_info(epoch, gs, train_loss, dev_loss, val_y, predicts)


if __name__ == "__main__":
	run(False)

