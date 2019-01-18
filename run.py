#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-29 下午3:06
# @Author  : 林利芳
# @File    : run.py
import os

from core.load_data import load_train_data, get_feed_dict, print_info, load_vocab_seq_len, load_test_data, \
	save_test_result
from config.config import logdir, checkpoint_dir, model_dir, TEST_DATA, TEST_RESULT
from model.siamese_network import SiameseNetwork
from config.hyperparams import HyperParams as hp
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np

checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)


def run(is_preprocessor=False):
	train_l_x, val_l_x, train_l_len, val_l_len, train_r_x, val_r_x, train_r_len, val_r_len, train_y, val_y, max_len, vocab = load_train_data(
		is_preprocessor)
	vocab_size = len(vocab.word2idx)
	
	model = SiameseNetwork(vocab_size, hp.embedding_size, max_len, hp.batch_size, is_training=True, seg=hp.seg)
	sv = tf.train.Supervisor(graph=model.graph, logdir=checkpoint_dir)
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
			dev_acc = []
			dev_loss = []
			for feed_dict, _ in get_feed_dict(model, val_l_x, val_r_x, val_l_len, val_r_len, val_y, hp.batch_size,
											  False):
				loss, gs, acc = sess.run([model.loss, model.global_step, model.accuracy], feed_dict=feed_dict)
				dev_loss.append(loss)
				dev_acc.append(acc)
			print_info(epoch, gs, train_loss, dev_loss, dev_acc)


def save_model():
	max_len, vocab = load_vocab_seq_len()
	vocab_size = len(vocab.word2idx)
	model = SiameseNetwork(vocab_size, hp.embedding_size, max_len, 1, is_training=False, seg=hp.seg)
	sv = tf.train.Supervisor(graph=model.graph, logdir=checkpoint_dir)
	with sv.managed_session() as sess:
		# saver = tf.train.Saver()
		# saver.restore(sess, checkpoint_file)
		graph_def = model.graph().as_graph_def()
		graph = graph_util.convert_variables_to_constants(sess, graph_def, ['similar', 'pre'])
		with tf.gfile.GFile(os.path.join(model_dir, 'siamese_model.pb'), 'wb') as fp:
			fp.write(graph.SerializeToString())


def test(filename=TEST_DATA, outfile=TEST_RESULT):
	idx, left_x, left_len, right_x, right_len, max_len, vocab = load_test_data(filename)
	y = np.ones_like(idx)
	vocab_size = len(vocab.word2idx)
	model = SiameseNetwork(vocab_size, hp.embedding_size, max_len, hp.batch_size, is_training=False, seg=hp.seg)
	sv = tf.train.Supervisor(graph=model.graph, logdir=checkpoint_dir, save_model_secs=0)
	with sv.managed_session() as sess:
		
		predicts = []
		for feed_dict, start_batch in get_feed_dict(model, left_x, right_x, left_len, right_len, y, hp.batch_size,
													False):
			pre_y, distince = sess.run([model.pre_y, model.distance], feed_dict=feed_dict)
			predicts.extend(pre_y[start_batch:])
		save_test_result(outfile, idx, predicts)


if __name__ == "__main__":
	# run(False)
	test()
# save_model()
