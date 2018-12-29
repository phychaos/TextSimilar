#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Linlifang
# @file: utils.py
# @time: 18-6-27下午6:13
import csv
from config.config import *
from config.config_path import glove_path, EMBED_VOCAB_PATH
import numpy as np
import pickle


def read_csv(filename, delimiter='\t'):
    """
    读取csv
    :param filename:
    :param delimiter:
    :return:
    """
    with open(filename, 'r') as fp:
        data = [each for each in csv.reader(fp, delimiter=delimiter)]
    return data


def load_data(filename):
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


def load_vocab():
    """
    加载词汇信息
    :return:
    """
    with open(EMBED_VOCAB_PATH, 'rb') as fp:
        embed, vocab_list = pickle.load(fp)
    return vocab_list, embed


def gen_batch_data(data):
    """
    长度补全， '_PAD'
    :param data:
    :return:
    """

    def padding(sent, l):
        return sent + ['_PAD'] * (l - len(sent))

    max_len = max([len(sentence) for sentence in data])
    texts, texts_length = [], []

    for item in data:
        texts.append(padding(item, max_len))
        texts_length.append(len(item))
    batched_data = {'texts': np.array(texts), 'texts_length': np.array(texts_length, dtype=np.int32)}
    return batched_data
