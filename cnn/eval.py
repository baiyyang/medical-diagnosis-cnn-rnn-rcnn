#!/usr/bin/python
# **************************
# * Author      :  baiyyang
# * Email       :  baiyyang@163.com
# * Description :  
# * create time :  2018/3/13下午5:33
# * file name   :  eval.py

import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv


# Parameters
# Data Parameters
tf.flags.DEFINE_string("train_data_file", "../data/train.txt",
                       "Data source for the train data.")
tf.flags.DEFINE_string("test_data_file", "../data/test.txt",
                       "Data source for the test data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./dropout_0.5/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
FLAGS(sys.argv)
print('\nParameters:')
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print('{}={}'.format(attr.upper(), value))

# CHANGE THIS: load data. load your own data here
if FLAGS.train_data_file:
    vocabulary, train_datas, train_labels, test_datas, test_labels\
        = data_helpers.load_data_and_labels_chinese(FLAGS.train_data_file, FLAGS.test_data_file)
    y_test = np.argmax(test_labels, axis=1)
else:
    x_raw = ['a masterpiece four years in the making', 'everything is off.']
    y_test = [1, 0]

x_test = np.array(test_datas)

print('\nEvaluating...\n')

# Evaluation
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

        # Tensors wer want to evaluate
        predictions = graph.get_operation_by_name('output/predictions').outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])


# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print('Total number of test examples: {}'.format(len(y_test)))
    print('Accuracy: {:g}'.format(correct_predictions / float(len(y_test))))


# # Save teh evaluation to a csv
# predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# out_path = os.path.join(FLAGS.checkpoint_dir, '..', 'prediction.csv')
# print('Saving evaluation to {}'.format(out_path))
# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)

