#!/usr/bin/python
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/3/13下午5:05
# * file name   :  data_helpers.py


import numpy as np
import re
import jieba
import string
from zhon.hanzi import punctuation
import collections
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def clean_str(s):
    """
    Tokenization/string cleaning for all datasets excepts for SSI.
    :param s:
    :return:
    """
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Return split sentences and labels.
    :param positive_data_file:
    :param negative_data_file:
    :return:
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, 'r', encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, 'r', encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_data_and_labels_chinese(train_data_file, test_data_file):
    """
    加载中文医疗疾病分类数据集
    :param train_data_file:
    :param test_data_file:
    :return:
    """
    words = []
    contents = []
    train_datas = []
    test_datas = []
    test_labels = []
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
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(9999))
    word2id = {}
    for word, _ in count:
        word2id[word] = len(word2id)
    # id2word = dict(zip(word2id.values(), word2id.keys()))
    print('dictionary_size:', len(word2id))
    sentence_max_length = max([len(content) for content in contents])
    print('sentence_max_length:', sentence_max_length)

    for content in contents:
        train_data = [word2id[word] if word in word2id.keys() else word2id['UNK'] for word in content]
        train_data.extend([0] * (sentence_max_length - len(train_data)))
        train_datas.append(train_data)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(labels))
    onehot_encoder = OneHotEncoder(sparse=False)
    train_labels = onehot_encoder.fit_transform(integer_encoded.reshape(len(integer_encoded), 1))
    print(train_labels.shape)

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
        test_data = [word2id[word] if word in word2id.keys() else word2id['UNK'] for word in content]
        if sentence_max_length > len(test_data):
            test_data.extend([0] * (sentence_max_length - len(test_data)))
        else:
            test_data = test_data[:sentence_max_length]
        test_datas.append(test_data)

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
    word2id, train_datas, train_labels, test_datas, test_labels = load_data_and_labels_chinese('data/train.txt',
                                                                                               'data/test.txt')
    print(len(train_datas), len(train_labels), len(test_datas), len(test_labels))
    print(train_datas[:5])
    print(train_labels[:5])
    print(test_datas[:5])
    print(test_labels[:5])

