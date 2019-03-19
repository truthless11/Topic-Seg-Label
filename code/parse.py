# -*- coding: utf-8 -*-
"""
@author: Ryuichi Takanobu
@e-mail: gxly15@mails.tsinghua.edu.cn; truthless11@gmail.com
"""
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters.")
tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 12210, "vocabulary size.")
tf.app.flags.DEFINE_integer("labels", 7, "Number of topic labels.")
tf.app.flags.DEFINE_integer("epoch_pre", 2, "Number of epoch on pretrain.")
tf.app.flags.DEFINE_integer("epoch_max", 15, "Maximum of epoch in iterative training.")
tf.app.flags.DEFINE_integer("embed_units", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("hidden_units", 100, "Size of hidden layer.")
tf.app.flags.DEFINE_integer("sample_round", 4, "Sample round in RL.")
tf.app.flags.DEFINE_float("keyword", 3.0, "Coefficient of keyword reward.")
tf.app.flags.DEFINE_float("continuity", 1.0, "Coefficient of continuity reward.")
tf.app.flags.DEFINE_float("learning_rate_srn", 0.0001, "SRN Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_pn", 0.00001, "PN Learning rate.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "Drop out rate.")
tf.app.flags.DEFINE_float("softmax_smooth", 0.5, "Discount rate in softmax.")
tf.app.flags.DEFINE_float("threshold", 0.005, "Threshold to judge the convergence.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory.")
tf.app.flags.DEFINE_string("train_filename", "train", "Filename for training set.")
tf.app.flags.DEFINE_string("valid_filename", "valid", "Filename for validation set.")
tf.app.flags.DEFINE_string("test_filename", "test", "Filename for test set.")
tf.app.flags.DEFINE_string("word_vector_filename", "my_vector", "Filename for word embedding vector.")
tf.app.flags.DEFINE_string("train_dir_srn", "./train_srn", "SRN Training directory.")
tf.app.flags.DEFINE_string("train_dir_pn", "./train_pn", "PN Training directory.")

FLAGS = tf.app.flags.FLAGS

def load_data(path, fname):
    print('Creating %s dataset...' % fname)
    label, text, sentence_len = [], [], []
    with open('%s/%s.csv' % (path, fname)) as f:
        count = 0
        for idx, line in enumerate(f):
            tokens = line.split(',')
            if count == 0:
                count = int(tokens[0])
                sentence_len.append(int(tokens[0]))
            else:
                count -= 1
                label.append(int(tokens[0]))
                text.append([int(x) for x in tokens[1].split('/')])
    keywords = np.loadtxt(path+'/keywords_'+fname, dtype=np.int)
    return label, text, sentence_len, keywords

def build_embed(path, fname):
    print("Loading word vectors...")
    embed = np.zeros([FLAGS.symbols, FLAGS.embed_units], dtype=np.float32)
    with open('%s/%s' % (path, fname)) as f:
        for idx, line in enumerate(f):
            embed[idx] = line.split(' ')
    return embed