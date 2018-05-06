# -*- coding: utf-8 -*-
"""
@author: Ryuichi Takanobu
@e-mail: gxly15@mails.tsinghua.edu.cn; truthless11@gmail.com
"""
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell, DropoutWrapper
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn

class HLSTM(object):
    def __init__(self,
                 num_symbols,
                 num_embed_units,
                 num_units,
                 num_labels,
                 embed,
                 learning_rate=0.001,
                 max_gradient_norm=5.0):
        self.texts = tf.placeholder(tf.int32, [None, None]) # shape: sentence*max_word
        self.text_length = tf.placeholder(tf.int32, [None]) # shape: sentence
        self.labels = tf.placeholder(tf.int32, [None])      # shape: sentence
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        
        # build the embedding table (index to vector)
        self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
        
        self.embed_inputs = tf.nn.embedding_lookup(self.embed, self.texts)   # shape: sentence*max_word*num_embed_units
        fw_cell = DropoutWrapper(BasicLSTMCell(num_units), output_keep_prob=self.keep_prob)
        bw_cell = DropoutWrapper(BasicLSTMCell(num_units), output_keep_prob=self.keep_prob)
        
        middle_outputs, middle_states = bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embed_inputs, self.text_length, dtype=tf.float32, scope="word_rnn")
        middle_outputs = tf.concat(middle_outputs, 2)   # shape: sentence*max_word*(2*num_units)
        
        middle_inputs = tf.expand_dims(tf.reduce_max(middle_outputs, axis=1), 0)    # shape: 1*sentence*(2*num_units)
        top_cell = DropoutWrapper(BasicLSTMCell(num_units), output_keep_prob=self.keep_prob)
        
        outputs, states = dynamic_rnn(top_cell, middle_inputs, dtype=tf.float32, scope="sentence_rnn")
        self.outputs = outputs[0]   # shape: sentence*num_units
        logits = tf.layers.dense(self.outputs, num_labels)
        
        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits), name='loss')
        mean_loss = self.loss / tf.cast(tf.shape(self.labels)[0], dtype=tf.float32)
        self.predict_labels = tf.argmax(logits, 1, 'predict_labels', output_type=tf.int32)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.labels, self.predict_labels), tf.int32), name='accuracy')
        
        self.params = tf.trainable_variables()
            
        # calculate the gradient of parameters
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(mean_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
        
        self.saver = tf.train.Saver(max_to_keep=3, pad_step_number=True)
        
    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
            
    def train_step(self, session, data):
        input_feed = {self.texts: data['texts'],
                self.text_length: data['texts_length'],
                self.labels: data['labels'],
                self.keep_prob: data['keep_prob']}
        output_feed = [self.loss, self.accuracy, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)
    
    def test_step(self, session, data):
        input_feed = {self.texts: data['texts'],
                self.text_length: data['texts_length'],
                self.labels: data['labels'],
                self.keep_prob: 1.0}
        output_feed = [self.loss, self.accuracy]
        return session.run(output_feed, input_feed)
    