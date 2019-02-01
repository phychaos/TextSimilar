#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-27 上午9:54
# @Author  : 林利芳
# @File    : config.py

import os

PATH = os.getcwd()
ATEC_NLP_DATA = os.path.join(PATH, 'data/atec_nlp_sim_train.csv')
ADD_ATEC_NLP_DATA = os.path.join(PATH, 'data/atec_nlp_sim_train_add.csv')

TEST_DATA = os.path.join(PATH, 'data/atec_nlp_sim_test.csv')
TEST_RESULT = os.path.join(PATH, 'data/test_result.csv')

EXPEND_ATEC_NLP_DATA = os.path.join(PATH, 'data/expend_atec_nlp_{}.csv')

DATA_PKL = os.path.join(PATH, 'data/data.pkl')
VOCAB_PKL = os.path.join(PATH, 'data/vocab.pkl')

CORPUS_DATA = os.path.join(PATH, 'data/corpus.txt')

WORD2VEC_DATA = os.path.join(PATH, 'data/char2vec_300')
logdir = os.path.join(PATH, 'logdir')
checkpoint_dir = "logdir/checkpoints-{}"
model_dir = os.path.join(logdir, "model")

WordChar = "char"
