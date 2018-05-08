#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/4/7上午10:40
# * file name   :  run_rnn.py


import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from rnn_model import TextRNN
from data_helper import batch_iter, process_file, build_vocab_labels


# Parameters
# ===================================

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string("param_name", "default_val", "description")

# config file path
tf.app.flags.DEFINE_string('train_data_path', '../data/train.txt', 'train data path')
tf.app.flags.DEFINE_string('test_data_path', '../data/test.txt', 'test data path')
tf.app.flags.DEFINE_string('tensorboard_path', 'save/tensorboard', 'tensorboard path')
tf.app.flags.DEFINE_string('save_path', 'save/result', 'save path')

# mode parameters
tf.app.flags.DEFINE_integer('embedding_dim', 64, 'words embedding dim')
tf.app.flags.DEFINE_integer('seq_length', 300, 'every content length')
tf.app.flags.DEFINE_integer('num_classes', 45, 'classification numbers')
tf.app.flags.DEFINE_integer('vocab_size', 5000, 'vocabulary size')
tf.app.flags.DEFINE_integer('num_layers', 2, 'bi-direction lstm')
tf.app.flags.DEFINE_integer('hidden_dim', 128, 'lstm dim size')
tf.app.flags.DEFINE_string('rnn', 'lstm', 'rnn unit lstm/gru')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.8, 'dropout keep prob')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'numbers epochs')
tf.app.flags.DEFINE_integer('print_per_batch', 5, 'print ans every batch')
tf.app.flags.DEFINE_integer('save_per_batch', 10, 'save ans every batch')


def get_time_dif(start_time):
    """获取已经使用的时间(seconds)"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=round(time_dif))


def init():
    """初始化模型"""
    rnn = TextRNN(
        embedding_dim=FLAGS.embedding_dim,
        seq_length=FLAGS.seq_length,
        num_classes=FLAGS.num_classes,
        vocab_size=FLAGS.vocab_size,
        num_layers=FLAGS.num_layers,
        hidden_dim=FLAGS.hidden_dim,
        rnn=FLAGS.rnn,
        dropout_keep_prob=FLAGS.dropout_keep_prob,
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        print_per_batch=FLAGS.print_per_batch,
        save_per_batch=FLAGS.save_per_batch
    )
    return rnn


def evaluate(rnn, sess, x, y):
    """在其他数据集上评估模型的准确率"""
    data_len = len(x)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_iter(x, y):
        batch_len = len(x_batch)
        feed_dict = {
            rnn.input_x: x_batch,
            rnn.input_y: y_batch,
            rnn.keep_prob: 1.0
        }
        loss, acc = sess.run([rnn.loss, rnn.acc], feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


def train(rnn):
    if not os.path.exists(FLAGS.tensorboard_path):
        os.makedirs(FLAGS.tensorboard_path)

    tf.summary.scalar('loss', rnn.loss)
    tf.summary.scalar('accuracy', rnn.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.tensorboard_path)

    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    # 载入训练集
    start_time = time.time()
    word2id, label2id, labels = build_vocab_labels(FLAGS.train_data_path)
    x_train, y_train = process_file(FLAGS.train_data_path, word2id, label2id, rnn.seq_length)
    x_test, y_test = process_file(FLAGS.test_data_path, word2id, label2id, rnn.seq_length)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    total_batch = 0  # 总批次
    best_acc_test = 0.0  # 最佳准确率
    
    # Statistic the number of parameters
    print("the number of parameters is: {}".format(np.sum([np.prod(v.get_shape().as_list())
                                                           for v in tf.trainable_variables()])))
 
    for epoch in range(rnn.num_epoches):
        print('Epoch: {}'.format(epoch + 1))
        batch_train = batch_iter(x_train, y_train, batch_size=rnn.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {
                rnn.input_x: x_batch,
                rnn.input_y: y_batch,
                rnn.keep_prob: rnn.dropout_keep_prob
            }

            if total_batch % rnn.save_per_batch == 0:
                # 每多少次迭代结果写入tensorboard
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s)

            if total_batch % rnn.print_per_batch == 0:
                # 每多少次迭代打印结果
                feed_dict[rnn.keep_prob] = 1.0
                loss_train, acc_train = session.run([rnn.loss, rnn.acc], feed_dict=feed_dict)
                loss_test, acc_test = evaluate(rnn, session, x_test, y_test)

                if acc_test > best_acc_test:
                    # 保存最好结果
                    best_acc_test = acc_test
                    saver.save(sess=session, save_path=FLAGS.save_path)

                time_dif = get_time_dif(start_time)
                # ^, <, > 分别是居中、左对齐、右对齐，后面带宽度，
                # : 号后面带填充的字符，只能是一个字符，不指定则默认是用空格填充。
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, Test Loss: ' \
                      '{3:>6.2}, Test Acc: {4:>7.2%}, Time: {5}'
                print(msg.format(total_batch, loss_train, acc_train, loss_test, acc_test, time_dif))

            session.run(rnn.optim, feed_dict=feed_dict)
            total_batch += 1


def test(rnn):
    start_time = time.time()
    word2id, label2id, labels = build_vocab_labels(FLAGS.train_data_path)
    x_test, y_test = process_file(FLAGS.test_data_path, word2id, label2id, rnn.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, save_path=FLAGS.save_path)

    loss_test, acc_test = evaluate(rnn, session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = rnn.batch_size
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            rnn.input_x: x_test[start_id: end_id],
            rnn.input_y: y_test[start_id: end_id],
            rnn.keep_prob: 1.0
        }
        y_pred_cls[start_id: end_id] = session.run(rnn.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print('Precision, Recall and F1-score...')
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=labels))

    # 混淆矩阵
    print("Confusion Matrix")
    print(metrics.confusion_matrix(y_test_cls, y_pred_cls))

    print('Time usage: {}'.format(get_time_dif(start_time)))


def main(_):
    rnn = init()
    train(rnn)
    test(rnn)


if __name__ == '__main__':
    tf.app.run()

