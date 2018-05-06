# -*- coding: utf-8 -*-
"""
@author: Ryuichi Takanobu
@e-mail: gxly15@mails.tsinghua.edu.cn; truthless11@gmail.com
"""
import numpy as np
import tensorflow as tf
import time
from parser import FLAGS, load_data, build_embed
from srn import HLSTM
from srn_tool import train_srn, evaluate_srn, inference_srn
from pn import PolicyGradient
from pn_tool import pretrain_pn, train_pn, develop_pn, evaluate_pn, inference_pn

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.log_parameters:
        print(FLAGS.__flags)
    label_train, text_train, sentence_len_train, keyword_train = load_data(FLAGS.data_dir, FLAGS.train_filename)
    label_dev, text_dev, sentence_len_dev, keyword_dev = load_data(FLAGS.data_dir, FLAGS.valid_filename)
    label_test, text_test, sentence_len_test, keyword_test = load_data(FLAGS.data_dir, FLAGS.test_filename)
    embed = build_embed(FLAGS.data_dir, FLAGS.word_vector_filename)
    
    SRN_graph = tf.Graph()
    PN_graph = tf.Graph()
    
    with SRN_graph.as_default():
        SRN = HLSTM(FLAGS.symbols,
                      FLAGS.embed_units,
                      FLAGS.hidden_units,
                      FLAGS.labels,
                      embed,
                      FLAGS.learning_rate_srn)
        if FLAGS.log_parameters:
            SRN.print_parameters()
        init_srn = tf.global_variables_initializer()
    
    sess_srn = tf.Session(graph=SRN_graph)
    best_acc = 0.0
    if tf.train.get_checkpoint_state(FLAGS.train_dir_srn):
        print("Reading model parameters from %s" % FLAGS.train_dir_srn)
        SRN.saver.restore(sess_srn, tf.train.latest_checkpoint(FLAGS.train_dir_srn))
        best_acc = np.float(np.loadtxt(FLAGS.train_dir_srn+'/best_acc'))
    else:
        print("Created model with fresh parameters.")
        sess_srn.run(init_srn)
    summary_writer_srn = tf.summary.FileWriter('%s/log' % FLAGS.train_dir_srn, sess_srn.graph)
            
    with PN_graph.as_default():
        PN = PolicyGradient(FLAGS.labels,
                               FLAGS.embed_units,
                               FLAGS.sample_round,
                               FLAGS.learning_rate_pn,
                               FLAGS.softmax_smooth)
        if FLAGS.log_parameters:
            PN.print_parameters()
        init_pn = tf.global_variables_initializer()
    
    sess_pn = tf.Session(graph=PN_graph)
    best_reward = 0.0
    if tf.train.get_checkpoint_state(FLAGS.train_dir_pn):
        print("Reading model parameters from %s" % FLAGS.train_dir_pn)
        PN.saver.restore(sess_pn, tf.train.latest_checkpoint(FLAGS.train_dir_pn))
        best_reward = np.float(np.loadtxt(FLAGS.train_dir_pn+'/best_reward'))
    else:
        print("Created model with fresh parameters.")
        sess_pn.run(init_pn)            
    summary_writer_pn = tf.summary.FileWriter('%s/log' % FLAGS.train_dir_pn, sess_pn.graph)
    
    if not FLAGS.is_train:
        obs_test = inference_srn(SRN, sess_srn, label_test, text_test, sentence_len_test, FLAGS.test_filename)
        inference_pn(PN, sess_pn, obs_test, sentence_len_test, keyword_test, FLAGS.test_filename)
        exit
    
##################################################################
    
    print("Start pre-training ...")
    
    while SRN.epoch.eval(sess_srn) < FLAGS.epoch_pre:
        epoch = SRN.epoch.eval(sess_srn)
        summary = tf.Summary()
        start_time = time.time()
        loss, accuracy = train_srn(SRN, sess_srn, label_train, text_train, sentence_len_train)
        summary.value.add(tag='loss/train', simple_value=loss)
        summary.value.add(tag='accuracy/train', simple_value=accuracy)
        summary_writer_srn.add_summary(summary, epoch)
        print("epoch %d learning rate %.4f epoch-time %.4f loss %.8f accuracy [%.8f]" % (epoch, SRN.learning_rate.eval(sess_srn), time.time()-start_time, loss, accuracy))
        
        loss, accuracy = evaluate_srn(SRN, sess_srn, label_dev, text_dev, sentence_len_dev)
        summary = tf.Summary()
        summary.value.add(tag='loss/dev', simple_value=loss)
        summary.value.add(tag='accuracy/dev', simple_value=accuracy)
        summary_writer_srn.add_summary(summary, epoch)
        print("        dev_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
        
        if accuracy >= best_acc:
            best_acc = accuracy
            SRN.saver.save(sess_srn, '%s/checkpoint' % FLAGS.train_dir_srn, global_step=SRN.global_step)
            np.savetxt(FLAGS.train_dir_srn+'/best_acc', np.array([best_acc]))
        
        loss, accuracy = evaluate_srn(SRN, sess_srn, label_test, text_test, sentence_len_test)
        summary = tf.Summary()
        summary.value.add(tag='loss/test', simple_value=loss)
        summary.value.add(tag='accuracy/test', simple_value=accuracy)
        summary_writer_srn.add_summary(summary, epoch)
        print("        test_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
    
    obs_train = inference_srn(SRN, sess_srn, label_train, text_train, sentence_len_train, FLAGS.train_filename)
    obs_dev = inference_srn(SRN, sess_srn, label_dev, text_dev, sentence_len_dev, FLAGS.valid_filename)
    obs_test = inference_srn(SRN, sess_srn, label_test, text_test, sentence_len_test, FLAGS.test_filename)
    
    while PN.epoch.eval(sess_pn) < FLAGS.epoch_pre:
        epoch = PN.epoch.eval(sess_pn)
        summary = tf.Summary()
        start_time = time.time()
        loss, reward, label_new = pretrain_pn(PN, sess_pn, obs_train, sentence_len_train, keyword_train, label_train)
        change_ratio = 1 - np.sum(np.equal(label_new, label_train), dtype=np.float) / len(keyword_train)
        summary.value.add(tag='loss/train', simple_value=loss)
        summary.value.add(tag='reward/train', simple_value=reward)
        summary.value.add(tag='change_ratio/train', simple_value=change_ratio)
        summary_writer_pn.add_summary(summary, epoch)
        print("epoch %d learning rate %.4f epoch-time %.4f loss %.8f reward [%.8f]" % (epoch, PN.learning_rate.eval(sess_pn), time.time()-start_time, loss, reward))
        label_train = label_new
        
        loss, reward, label_new = develop_pn(PN, sess_pn, obs_dev, sentence_len_dev, keyword_dev)
        change_ratio = 1 - np.sum(np.equal(label_new, label_dev), dtype=np.float) / len(keyword_dev)
        summary = tf.Summary()
        summary.value.add(tag='loss/dev', simple_value=loss)
        summary.value.add(tag='reward/dev', simple_value=reward)
        summary.value.add(tag='change_ratio/dev', simple_value=change_ratio)
        summary_writer_pn.add_summary(summary, epoch)
        print("        dev_set, loss %.8f, reward [%.8f], change_ratio %.4f" % (loss, reward, change_ratio))
        
        if reward >= best_reward:
            best_reward = reward
            PN.saver.save(sess_pn, '%s/checkpoint' % FLAGS.train_dir_pn, global_step=PN.global_step)
            np.savetxt(FLAGS.train_dir_pn+'/best_reward', np.array([best_reward]))
        label_dev = label_new
        
        accuracy = evaluate_pn(PN, sess_pn, obs_test, sentence_len_test, keyword_test, label_test)
        summary = tf.Summary()
        summary.value.add(tag='accuracy/test', simple_value=accuracy)
        summary_writer_pn.add_summary(summary, epoch)
        print("        test_set, accuracy [%.8f]" % (accuracy))
        
    label_train = inference_pn(PN, sess_pn, obs_train, sentence_len_train, keyword_train, FLAGS.train_filename)
    label_dev = inference_pn(PN, sess_pn, obs_dev, sentence_len_dev, keyword_dev, FLAGS.valid_filename)      
    
##################################################################
    
    print("Start training ...")
    
    for _ in range(FLAGS.epoch_max):
        
        epoch = SRN.epoch.eval(sess_srn)
        summary = tf.Summary()
        start_time = time.time()
        loss, accuracy = train_srn(SRN, sess_srn, label_train, text_train, sentence_len_train)
        summary.value.add(tag='loss/train', simple_value=loss)
        summary.value.add(tag='accuracy/train', simple_value=accuracy)
        summary_writer_srn.add_summary(summary, epoch)
        print("epoch %d learning rate %.4f epoch-time %.4f loss %.8f accuracy [%.8f]" % (epoch, SRN.learning_rate.eval(sess_srn), time.time()-start_time, loss, accuracy))
        
        loss, accuracy = evaluate_srn(SRN, sess_srn, label_dev, text_dev, sentence_len_dev)
        summary = tf.Summary()
        summary.value.add(tag='loss/dev', simple_value=loss)
        summary.value.add(tag='accuracy/dev', simple_value=accuracy)
        summary_writer_srn.add_summary(summary, epoch)
        print("        dev_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
        
        if accuracy >= best_acc:
            best_acc = accuracy
            SRN.saver.save(sess_srn, '%s/checkpoint' % FLAGS.train_dir_srn, global_step=SRN.global_step)
            np.savetxt(FLAGS.train_dir_srn+'/best_acc', np.array([best_acc]))
        
        loss, accuracy = evaluate_srn(SRN, sess_srn, label_test, text_test, sentence_len_test)
        summary = tf.Summary()
        summary.value.add(tag='loss/test', simple_value=loss)
        summary.value.add(tag='accuracy/test', simple_value=accuracy)
        summary_writer_srn.add_summary(summary, epoch)
        print("        test_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
            
        obs_train = inference_srn(SRN, sess_srn, label_train, text_train, sentence_len_train, FLAGS.train_filename)
        obs_dev = inference_srn(SRN, sess_srn, label_dev, text_dev, sentence_len_dev, FLAGS.valid_filename)
        obs_test = inference_srn(SRN, sess_srn, label_test, text_test, sentence_len_test, FLAGS.test_filename)
        
        
        epoch = PN.epoch.eval(sess_pn)
        summary = tf.Summary()
        start_time = time.time()
        loss, reward, label_new = train_pn(PN, sess_pn, obs_train, sentence_len_train, keyword_train)
        change_ratio = 1 - np.sum(np.equal(label_new, label_train), dtype=np.float) / len(keyword_train)
        summary.value.add(tag='loss/train', simple_value=loss)
        summary.value.add(tag='reward/train', simple_value=reward)
        summary.value.add(tag='change_ratio/train', simple_value=change_ratio)
        summary_writer_pn.add_summary(summary, epoch)
        print("epoch %d learning rate %.4f epoch-time %.4f loss %.8f reward [%.8f]" % (epoch, PN.learning_rate.eval(sess_pn), time.time()-start_time, loss, reward))
        label_train = label_new
        
        loss, reward, label_new = develop_pn(PN, sess_pn, obs_dev, sentence_len_dev, keyword_dev)
        change_ratio = 1 - np.sum(np.equal(label_new, label_dev), dtype=np.float) / len(keyword_dev)
        summary = tf.Summary()
        summary.value.add(tag='loss/dev', simple_value=loss)
        summary.value.add(tag='reward/dev', simple_value=reward)
        summary.value.add(tag='change_ratio/dev', simple_value=change_ratio)
        summary_writer_pn.add_summary(summary, epoch)
        print("        dev_set, loss %.8f, reward [%.8f], change_ratio %.4f" % (loss, reward, change_ratio))
        
        if reward >= best_reward:
            best_reward = reward
            PN.saver.save(sess_pn, '%s/checkpoint' % FLAGS.train_dir_pn, global_step=PN.global_step)
            np.savetxt(FLAGS.train_dir_pn+'/best_reward', np.array([best_reward]))
        
        accuracy = evaluate_pn(PN, sess_pn, obs_test, sentence_len_test, keyword_test, label_test)
        summary = tf.Summary()
        summary.value.add(tag='accuracy/test', simple_value=accuracy)
        summary_writer_pn.add_summary(summary, epoch)
        print("        test_set, accuracy [%.8f]" % (accuracy))

        if change_ratio < FLAGS.threshold:
            break
        label_dev = label_new
        
        label_train = inference_pn(PN, sess_pn, obs_train, sentence_len_train, keyword_train, FLAGS.train_filename)
        label_dev = inference_pn(PN, sess_pn, obs_dev, sentence_len_dev, keyword_dev, FLAGS.valid_filename)
        inference_pn(PN, sess_pn, obs_test, sentence_len_test, keyword_test, FLAGS.test_filename) 

    inference_pn(PN, sess_pn, obs_test, sentence_len_test, keyword_test, FLAGS.test_filename)  
        