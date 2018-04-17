#!/usr/bin/python
# -*- coding:utf-8 -*-
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/3/15下午3:41
# * file name   :  medical_diagnosis_train.py


import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
import data_helpers
import pickle
from text_cnn import TextCNN
from text_mf_cnn import MFTextCNN
from text_afc_cnn import AFCTextCNN
from tensorflow.contrib import learn


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_data_file", "./data_10/train.txt",
                       "Data source for the train data.")
tf.flags.DEFINE_string("test_data_file", "./data_10/test.txt",
                       "Data source for the test data.")

# Choose the model
tf.flags.DEFINE_string("model", "AFC-TextCNN", "Choose which the model")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS.__parse
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation

# Load data
print('Loading data....')
print(FLAGS.train_data_file, FLAGS.test_data_file)
# x_test, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
#
# # Build vocabulary
# max_document_length = max([len(x.split(' ')) for x in x_test])
# # 将文本置为统一的长度，不够的话，用0填充
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_test)))
#
# # Randomly shuffle data
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices]
#
# # Split train/test set
# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
# x_train, x_dev = x_shuffled[: dev_sample_index], x_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#
#
# del x, y, x_shuffled, y_shuffled
#
# print('Vocabulary Size: {:d}'.format(len(vocab_processor.vocabulary_)))

vocabulary, train_datas, train_labels, test_datas, test_labels = \
    data_helpers.load_data_and_labels_chinese(FLAGS.train_data_file, FLAGS.test_data_file)
shuffle_indices = np.random.permutation(np.arange(len(train_datas)))
train_datas = np.array(train_datas)
x_train, y_train = train_datas[shuffle_indices], train_labels[shuffle_indices]
x_dev, y_dev = np.array(test_datas), test_labels

print('Train/Dev split: {:d}/{:d}'.format(len(y_train), len(y_dev)))

# Training
with tf.Graph().as_default():
    # 设置允许TensorFlow在首选设备不存在时执行特定操作时回落到设备上。
    # 例如，如果我们的代码在GPU上放置操作，并在没有GPU的机器上运行代码，
    # 则不使用allow_soft_placement会导致错误。
    # 如果设置了log_device_placement，
    # 则TensorFlow会记录它放置哪些设备（CPU或GPU）的操作。 这对调试很有用。
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        if FLAGS.model == "MF-TextCNN":
            cnn = MFTextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocabulary),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )
            print("Use the MF-TextCNN model")
        elif FLAGS.model == "AFC-TextCNN":
            cnn = AFCTextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocabulary),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )
            print("Use the AFC-TextCNN model")
        else:
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocabulary),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )
            print("Use the TextCNN model")

        # define training procedure
        # 在这里，train_op在这里是一个新创建的操作，我们可以运行它来对参数执行梯度更新。
        # train_op的每次执行都是一个训练步骤。
        # TensorFlow会自动计算出哪些变量是“可训练的”并计算出它们的梯度。
        # 通过定义global_step变量并将其传递给优化器，我们允许TensorFlow为我们处理训练步骤的计数。
        # 每次执行train_op时，全局步将自动递增1。
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for model and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
        print('Writing to {}\n'.format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar('loss', cnn.loss)
        acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev Summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints/'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Writer vocabulary
        with open(os.path.join(os.path.curdir, 'vocabulary.pkl'), 'wb') as f:
            pickle.dump(vocabulary, f)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A Single training step
            :param x_batch:
            :param y_batch:
            :return:
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                                                          feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # if step % 100 == 0:
            print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            :param x_batch:
            :param y_batch:
            :param writer:
            :return:
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                                                       feed_dict=feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            # print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return step, loss, accuracy

        # Generate batch
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. for each batch
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print('\nEvaluation:')
                step, loss_total, accuracy_total = 0, 0.0, 0.0
                for dev_batch in data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1):
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    step, loss, accuracy = dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
                    loss_total += loss * FLAGS.batch_size
                    accuracy_total += accuracy * FLAGS.batch_size
                time_str = datetime.datetime.now().isoformat()
                print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss_total / len(x_dev),
                                                                accuracy_total / len(x_dev)))

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_dir, global_step=current_step)
                print('Saved model checkpint to {}\n'.format(path))



