# -*- coding: utf-8 -*-
"""
@author: Ryuichi Takanobu
@e-mail: gxly15@mails.tsinghua.edu.cn; truthless11@gmail.com
"""
import numpy as np
import tensorflow as tf

class PolicyGradient(object):
    def __init__(self,
                 num_actions,
                 num_features,
                 sample_round,
                 learning_rate=0.001,
                 softmax_smooth=1.0,
                 max_gradient_norm=5.0):
        self.n_actions = num_actions
        self.n_features = num_features
        self.sample_round = sample_round
        self.beta = softmax_smooth
        
        self.ep_obs, self.ep_as = [[] for _ in range(self.sample_round)], [[] for _ in range(self.sample_round)]
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, 2*self.n_features], name="observations")
            self.tf_idx = tf.placeholder(tf.float32, [None, self.n_actions], name="indices")  # related to previous action
            self.tf_acts = tf.placeholder(tf.int32, [None], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None], name="actions_value")
            self.keep_prob = tf.placeholder(tf.float32)
        
        with tf.name_scope('state'):
            topic_vec = tf.layers.dense(self.tf_idx, self.n_features, name='topic_vec')
            all_obs = tf.concat([self.tf_obs, topic_vec], 1)    # shape=[seq_len, 3*self.n_features]
            all_state = tf.layers.dense(all_obs, self.n_features, activation=tf.nn.elu, name='fc') 
            self.state = tf.layers.dropout(all_state, self.keep_prob, name='dropout')   # shape=[seq_len, n_features]        
        
        with tf.name_scope('policy'):
            all_act = tf.layers.dense(self.state, self.n_actions, name='fc_policy')
            all_act_smoothed = tf.multiply(all_act, self.beta)
            self.all_act_prob = tf.nn.softmax(all_act_smoothed, name='act_prob')    # use softmax to convert to probability
        
        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)  # this is negative log of chosen action
            self.loss = tf.reduce_sum(neg_log_prob * self.tf_vt)
            mean_loss = self.loss / tf.cast(tf.shape(all_act)[0], dtype=tf.float32) # reward guided loss
            pre_loss = tf.reduce_mean(neg_log_prob)
        
        with tf.name_scope('train'):
            self.params = tf.trainable_variables()
            opt = tf.train.AdamOptimizer(self.learning_rate)
            
            gradients = tf.gradients(mean_loss, self.params)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
            
            pre_gradients = tf.gradients(pre_loss, self.params)
            pre_clipped_gradients, self.pre_gradient_norm = tf.clip_by_global_norm(pre_gradients, max_gradient_norm)
            self.pre_update = opt.apply_gradients(zip(pre_clipped_gradients, self.params), global_step=self.global_step)
        
        self.saver = tf.train.Saver(max_to_keep=3, pad_step_number=True)
    
    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
    
    def choose_action(self, session, observation, prev_action, is_train):
        prob_weights = session.run(self.all_act_prob, feed_dict={self.tf_idx: [prev_action], self.keep_prob: 1.0,
                self.tf_obs: observation[np.newaxis, :]})   # shape=[1, n_features]    
        prob_weights = prob_weights.ravel()
        
        # select action w.r.t the actions prob
        action = np.random.choice(len(prob_weights), p=prob_weights) if is_train else np.argmax(prob_weights)
        return action
    
    def store_transition(self, s, a, n_round):
        self.ep_obs[n_round].append(s)
        self.ep_as[n_round].append(a)
    
    def learn(self, session, reward, keep_prob):
        # discount episode rewards
        seq_len = len(self.ep_as[0])
        discounted_ep_rs = np.array([reward] * seq_len).T

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs, axis=0)
        
        loss = .0
        for n_round in range(self.sample_round):
            one_hot = np.zeros([seq_len, self.n_actions], dtype=np.int)
            one_hot[np.arange(1, seq_len), self.ep_as[n_round][:-1]] = 1
            outputs = session.run([self.loss, self.update], feed_dict={
                    self.tf_obs: np.array(self.ep_obs[n_round]),    # shape=[seq_len, 2*n_features]
                    self.tf_idx: one_hot,
                    self.tf_acts: np.array(self.ep_as[n_round]),    # shape=[seq_len]
                    self.tf_vt: discounted_ep_rs[n_round],          # shape=[seq_len]
                    self.keep_prob: keep_prob})
            loss += outputs[0]
        
        self.clean()
        return loss / self.sample_round
        
    def prelearn(self, session, reward, keep_prob):
        seq_len = len(self.ep_as[0])
        one_hot = np.zeros([seq_len, self.n_actions], dtype=np.int)
        one_hot[np.arange(1, seq_len), self.ep_as[0][:-1]] = 1
        outputs = session.run([self.loss, self.update], feed_dict={
                self.tf_obs: np.array(self.ep_obs[0]),
                self.tf_idx: one_hot,
                self.tf_acts: np.array(self.ep_as[0]),
                self.tf_vt: np.array([reward] * seq_len),
                self.keep_prob: keep_prob})
        loss = outputs[0]
        self.clean()
        return loss
    
    def verify(self, session, reward):
        seq_len = len(self.ep_as[0])
        one_hot = np.zeros([seq_len, self.n_actions], dtype=np.int)
        one_hot[np.arange(1, seq_len), self.ep_as[0][:-1]] = 1
        loss = session.run(self.loss, feed_dict={
                self.tf_obs: np.array(self.ep_obs[0]),
                self.tf_idx: one_hot,
                self.tf_acts: np.array(self.ep_as[0]),
                self.tf_vt: np.array([reward] * seq_len),
                self.keep_prob: 1.0})
        self.clean()
        return loss
    
    def clean(self):
        self.ep_obs, self.ep_as = [[] for _ in range(self.sample_round)], [[] for _ in range(self.sample_round)]
        