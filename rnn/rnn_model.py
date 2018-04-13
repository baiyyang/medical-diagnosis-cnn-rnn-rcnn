#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/4/6下午12:26
# * file name   :  rnn_model.py


import tensorflow as tf


class TextRNN(object):
    """RNN 配置参数"""

    # 初始化模型参数
    def __init__(self, embedding_dim, seq_length, num_classes, vocab_size, num_layers, hidden_dim, rnn,
                 dropout_keep_prob, learning_rate, batch_size, num_epochs, print_per_batch, save_per_batch):
        self.embedding_dim = embedding_dim  # 词向量维度
        self.seq_length = seq_length  # 序列长度
        self.num_classes = num_classes  # 类别书
        self.vocab_size = vocab_size  # 词汇表大小
        self.num_layers = num_layers  # 隐藏层层数
        self.hidden_dim = hidden_dim  # 隐藏层神经元
        self.rnn = rnn  # lstm 或者是 gru
        self.dropout_keep_prob = dropout_keep_prob  # dropout保留比例
        self.learning_rate = learning_rate  # 学习率
        self.batch_size = batch_size  # 每批次训练大小
        self.num_epoches = num_epochs  # 总迭代的次数
        self.print_per_batch = print_per_batch  # 没多少词输出一次接轨
        self.save_per_batch = save_per_batch  # 没多少轮存入tensorboard

        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # rnn 模型
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)

        def grn_cell():
            return tf.rnn.GRUCell(self.hidden_dim)

        def dropout():
            if self.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = grn_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            # 随机编码
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('rnn'):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope('score'):
            # 全连接层
            fc = tf.layers.dense(last, self.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(logits=self.logits), 1)

        with tf.name_scope('optimize'):
            # 损失函数，交叉熵
            cross_entroy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entroy)

            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




