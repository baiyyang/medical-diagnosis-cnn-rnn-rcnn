#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/4/6下午12:21
# * file name   :  data_helper.py


import pickle
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            fields = line.strip().split('\t')
            if len(fields) == 2:
                contents.append(list(fields[0]))
                labels.append(fields[1])
    return contents, labels


def build_vocab_labels(trainfilename, vocab_size=5000):
    """构建词表，并进行存储"""
    data_train, labels = read_file(trainfilename)
    all_data = []
    for content in data_train:
        all_data.extend(content)

    # 构建content词典
    count_pairs = Counter(all_data).most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = list(words)

    vocabulary = {'PAD': 0, 'UNK': 1}
    for word in words:
        vocabulary[word] = len(vocabulary)

    # 将词典写入文件中
    # with open('vocabulary.pkl', 'wb') as fw:
    #     pickle.dump(vocabulary, fw)

    # label词典
    labels = set(labels)
    label2id = dict(zip(labels, range(len(labels))))

    return vocabulary, label2id, labels


# build_vocab('../data/train.txt', 'vocabulary.pkl')
# with open('vocabulary.pkl', 'rb') as fr:
#     vocabulary = pickle.load(fr)
# print(vocabulary)


def process_file(filename, word2id, label2id, max_length=300):
    """"将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word2id[x] if x in word2id.keys() else word2id['UNK'] for x in contents[i]])
        label_id.append(label2id[labels[i]])

    # 使用keras提供的pad_sequences来将文本固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')

    # 将标签转换成one-hot的形式
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(label2id))

    return x_pad, y_pad


# word2id, label2id, labels = build_vocab_labels('../data/train.txt')
# process_file('../data/test.txt', word2id, label2id)


def batch_iter(x, y, batch_size=64, shuffle=True):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    if shuffle:
        indices = np.random.permutation(np.arange(data_len))
        x = x[indices]
        y = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x[start_id: end_id], y[start_id: end_id]

