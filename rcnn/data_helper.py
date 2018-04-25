#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/4/8下午4:56
# * file name   :  data_helper.py


import jieba
import collections
import string
from zhon.hanzi import punctuation
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.contrib import keras as kr


def load_data_and_labels_chinese(train_data_file, test_data_file, sequence_length, vocabulary_size):
    """
    加载中文医疗疾病分类数据集
    :param train_data_file:
    :param test_data_file:
    :param sequence_length:
    :param vocabulary_size:
    :return:
    """
    words = []
    contents = []
    train_datas = []
    test_datas = []
    labels = []
    # 生成训练数据集
    with open(train_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data, label = line.strip().split('\t')
            labels.append(label)
            # 分词
            segments = [seg for seg in jieba.cut(data, cut_all=False)]

            segments_ = [seg.strip() for seg in segments if seg not in punctuation
                         and seg not in string.punctuation]
            contents.append([seg_ for seg_ in segments_ if seg_ != ''])
            words.extend(segments_)
    words = [word for word in words if word != '']
    count = [['UNK', -1], ['PAD', 0]]
    print(len(collections.Counter(words)))
    count.extend(collections.Counter(words).most_common(vocabulary_size - len(count)))
    word2id = {}
    for word, _ in count:
        word2id[word] = len(word2id)
    # id2word = dict(zip(word2id.values(), word2id.keys()))
    # print('dictionary_size:', len(word2id))
    # sentence_max_length = max([len(content) for content in contents])
    # print('sentence_max_length:', sentence_max_length)

    train_datas = []
    for content in contents:
        train_datas.append([word2id[word] if word in word2id.keys() else word2id['UNK'] for word in content])

    # 使用keras提供的pad_sequences来将文本固定长度
    train_datas = kr.preprocessing.sequence.pad_sequences(train_datas,
                                                          sequence_length, padding='post', truncating='post', value=1)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(labels))
    onehot_encoder = OneHotEncoder(sparse=False)
    # shape: [None, num_classes]
    train_labels = onehot_encoder.fit_transform(integer_encoded.reshape(len(integer_encoded), 1))

    # 生成测试数据集
    labels = []
    contents = []
    with open(test_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data, label = line.strip().split('\t')
            labels.append(label)
            # 分词
            segments = [segment for segment in jieba.cut(data, cut_all=False)]
            segments_ = [segment.strip() for segment in segments if segment not in punctuation
                         and segment not in string.punctuation]
            contents.append([seg_ for seg_ in segments_ if seg_ != ''])
    for content in contents:
        test_datas.append([word2id[word] if word in word2id.keys() else word2id['UNK'] for word in content])
    test_datas = kr.preprocessing.sequence.pad_sequences(test_datas,
                                                         sequence_length, padding='post', truncating='post', value=1)

    integer_encoded = label_encoder.fit_transform(np.array(labels))
    onehot_encoder = OneHotEncoder(sparse=False)
    test_labels = onehot_encoder.fit_transform(integer_encoded.reshape(len(integer_encoded), 1))

    return word2id, train_datas, train_labels, test_datas, test_labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index: end_index]


if __name__ == '__main__':
    word2id, train_datas, train_labels, test_datas, test_labels = \
        load_data_and_labels_chinese('../data/train.txt', '../data/test.txt', 300, 10000)
    print(len(train_datas), len(train_labels))
    print(len(test_datas), len(test_labels))
    # print(word2id['UNK'], word2id['PAD'])
    # print(train_datas.shape, train_labels.shape, test_datas.shape, test_labels.shape)
    batchs = batch_iter(list(zip(train_datas, train_labels)), 64, 1)
    for i, batch in enumerate(batchs):
        x_batch, y_batch = zip(*batch)
        print(i, len(x_batch), len(y_batch))



