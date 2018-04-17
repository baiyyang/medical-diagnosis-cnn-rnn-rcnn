#!/usr/bin/python
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/3/13下午2:25
# * file name   :  text_cnn.py

import numpy as np
import tensorflow as tf


class MFTextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0):
        """

        :param sequence_length: the length of our sentences. we padded all oer sentences to have the same length.
        :param num_classes: number of classes in the output layer, two in our case(positive and negative)
        :param vocab_size: the size of our vocabulary. This is needed to define the size of our embedding layer.
                            which will have shape [vocabulary_size, embedding_size]
        :param embedding_size: the dimensionality of our embeddings.
        :param filter_sizes: the number of words we want our convolutional filters to cover.
        :param num_filters: the number of filters per filter size
        :param l2_reg_lambda:
        """
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss(optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            # tf.nn.embedding_lookup（tensor, id）其中，tensor就是输入张量，id就是张量对应的索引
            # 返回三维的tensor, [None, sequence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 卷积操作需要的是4维的tensor, [batch, width, height, channel]
            # -1，表示在最后一维，
            # expand_dim 和 reshape都可以改变维度，但是在构建具体的图时，如果没有包含具体的值，使用reshape则会报错
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + mean_features layer for each filter size
        pool_ans = []
        min_val = 1e+9
        for filter_size in filter_sizes:
            with tf.name_scope('conv-meanpool-%s' % filter_size):
                # Convolutional layer
                # 卷积核，[卷积核的高度，卷积核的宽度，图像通的个数，卷积核个数]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                # padding：SAME 表示用0来填充，VALID表示不填充
                # strides：一个长度为4的一维向量，表示在data_format上每一维上移动的步长
                # strides: [batch, height, width, channels],其中batch 和 channels要求为1
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1],
                                    padding='VALID', name='conv')
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                print("filter_size: {}, h shape: {}".format(filter_size, h.shape))

                # Average pooling over the outputs
                pool_flat = tf.reduce_mean(h, axis=-1)
                pool_flat = tf.reduce_mean(pool_flat, axis=-1)
                pool_ans.append(pool_flat)
                print("mean_features_filter_size: {}, h shape: {}".format(filter_size, pool_flat))

                # 获取最小值
                min_val = min(min_val, pool_flat.shape[1])

        # 输出最小值信息
        print("Min val is: {}".format(min_val))

        self.h_pool = []
        for pool_flat in pool_ans:
            self.h_pool.append(pool_flat[:, : min_val])

        self.h_pool_flat = tf.reshape(tf.reduce_mean(self.h_pool, axis=0), shape=[-1, min_val])
        print("h_pool_flat shape: {}".format(self.h_pool_flat.shape))

        # Add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable('W', shape=[min_val, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            # l2正则化
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')




