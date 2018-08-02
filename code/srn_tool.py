# -*- coding: utf-8 -*-
"""
@author: Ryuichi Takanobu
@e-mail: gxly15@mails.tsinghua.edu.cn; truthless11@gmail.com
"""
import numpy as np
from parser import FLAGS

def gen_batch_data(labels, text):
    def padding(sent, length):
        return sent + [0] * (length - len(sent))
    
    max_len = np.max([len(item) for item in text])
    texts, texts_length = [], []
        
    for item in text:
        texts.append(padding(item, max_len))
        texts_length.append(len(item))
    
    batched_data = {'texts': np.array(texts), 'texts_length':texts_length, 'labels':labels, 'keep_prob':FLAGS.keep_prob}
    return batched_data

def train_srn(model, sess, label, text, batch_sizes):
    st, ed, cnt, loss, accuracy = 0, 0, 0, .0, .0
    while ed < len(label):
        st, ed, cnt = ed, ed+batch_sizes[cnt], cnt+1
        batch_data = gen_batch_data(label[st:ed], text[st:ed])
        outputs = model.train_step(sess, batch_data)
        loss += outputs[0]
        accuracy += outputs[1]
    sess.run(model.epoch_add_op)
    return loss / len(label), accuracy / len(label)

def evaluate_srn(model, sess, label, text, batch_sizes):
    st, ed, cnt, loss, accuracy = 0, 0, 0, .0, .0
    while ed < len(label):
        st, ed, cnt = ed, ed+batch_sizes[cnt], cnt+1
        batch_data = gen_batch_data(label[st:ed], text[st:ed])
        outputs = model.test_step(sess, batch_data)
        loss += outputs[0]
        accuracy += outputs[1]
    return loss / len(label), accuracy / len(label)

def inference_srn(model, sess, label, text, batch_sizes, part):
    st, ed, cnt = 0, 0, 0
    hidden_states = np.zeros([len(label), FLAGS.hidden_units], dtype=np.float32)
    while ed < len(label):
        st, ed, cnt = ed, ed+batch_sizes[cnt], cnt+1
        batch_data = gen_batch_data(label[st:ed], text[st:ed])
        hidden_states[st:ed] = sess.run(model.outputs, {model.texts:batch_data['texts'], model.text_length:batch_data['texts_length'], model.keep_prob:1.0})
    np.savetxt(FLAGS.train_dir_srn+'/states_'+part, hidden_states)
    return hidden_states