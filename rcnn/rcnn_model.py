#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/4/8下午2:40
# * file name   :  rcnn_model.py


import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


class RCNN(object):
    """构建双向LSTM-CNN text classification网络模型"""
    def __init__(self, embedding_dim, sequence_length, hidden_dim, num_classes, vocabulary_size, dropout_keep_prob,
                 filter_nums, learning_rate, batch_size, epochs):
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes  # 分类的总数
        self.vocabulary_size = vocabulary_size
        self.dropout_keep_prob = dropout_keep_prob
        self.filter_nums = filter_nums
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # placeholders input, output, dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # 词向量映射
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_dim],
                                                       minval=-0.25, maxval=0.25), dtype=tf.float32)
            self.inputs_embeddings = tf.nn.embedding_lookup(embeddings, self.input_x)

        # bi-lstm层
        with tf.name_scope('bi-lstm'):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_fb = LSTMCell(self.hidden_dim)

            # 输出值为(outputs, output_states)的元组，
            # outputs 为包含前向cell输出的tensor和后向cell输出的tensor
            # 假设time_major=false,tensor的shape为[batch_size, max_time, depth]
            # output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_fb,
                inputs=self.inputs_embeddings,
                dtype=tf.float32
            )
            # shape [batch_size, max_time(sequence_length), 2 * depth(hidden_dim)]
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.keep_prob)

        # convolution层
        with tf.name_scope('conv_max_pool'):
            # 使用卷积层时，得需要四个维度[batch, width, height, channel]
            # 目前是[batch_size, sequence_length, 2*hidden_dim, 1]
            output_expanded = tf.expand_dims(output, -1)

            # 卷积核 [height, width, channel, nums]
            # 在这里卷积核的高度为1
            filter_shape = [1, 2 * self.hidden_dim, 1, self.filter_nums]
            conv_w = tf.Variable(tf.random_uniform(shape=filter_shape, minval=-0.25, maxval=0.25),
                                 name='conv_w')
            conv_b = tf.Variable(tf.constant(0.1, shape=[self.filter_nums], name='conv_b'))
            conv_output = tf.nn.conv2d(output_expanded, conv_w, strides=[1, 1, 1, 1], padding='VALID',
                                       name='conv_output')

            # conv ans
            y = tf.nn.relu(tf.nn.bias_add(conv_output, conv_b), name='bias')

            # shape [batch_size, sequence_length, 1, filter_nums] ->
            # [batch_size, sequence_length, filter_nums]
            self.y_pool = tf.reshape(y, shape=[-1, sequence_length, filter_nums])

            # shape [batch_size, filter_nums]
            self.y_pool_flat = tf.reduce_max(self.y_pool, axis=1)

            # Add dropout
            with tf.name_scope('dropout'):
                self.y_dropout = tf.nn.dropout(self.y_pool_flat, self.keep_prob)

            # output scores and predictions
            with tf.name_scope('output'):
                softmax_w = tf.get_variable(name='softmax_w', shape=[self.filter_nums, num_classes],
                                            initializer=tf.contrib.layers.xavier_initializer())
                softmax_b = tf.Variable(tf.constant(0.1, shape=[num_classes], name='softmax_b'))
                self.scores = tf.nn.xw_plus_b(self.y_dropout, softmax_w, softmax_b, name='scores')
                self.predictions = tf.argmax(self.scores, 1, name='predictions')

            # loss
            with tf.name_scope('loss'):
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(loss)

            # accuracy
            with tf.name_scope('accuracy'):
                accuracy = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
                self.accuracy = tf.reduce_mean(tf.cast(accuracy, 'float'), name='accuracy')




