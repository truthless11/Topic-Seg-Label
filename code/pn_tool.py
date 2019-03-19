# -*- coding: utf-8 -*-
"""
@author: Ryuichi Takanobu
@e-mail: gxly15@mails.tsinghua.edu.cn; truthless11@gmail.com
"""
import numpy as np
from parse import FLAGS

def similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def HSMI(batch_obs, action_list):
    val, cnt, precluster, cluster = .0, 0, None, None
    for idx, obs in enumerate(batch_obs):
        if idx > 0:
            if action_list[idx] != action_list[idx-1]:
                cnt += 1
                if precluster is not None:
                    val += similarity(np.mean(cluster, axis=0), np.mean(precluster, axis=0))
                precluster = cluster
                cluster = np.array([obs])
            else:
                cluster = np.vstack([cluster, obs])
        else:
            cluster = np.array([obs])
    if precluster is not None:  # last cluster
        val += similarity(np.mean(cluster, axis=0), np.mean(precluster, axis=0))
    return -val / cnt if cnt > 0 else .0

def determine(model, batch_obs, keywords, labels):
    cluster, reward = np.zeros([1, FLAGS.embed_units]), .0
    for idx, obs in enumerate(batch_obs):
        full_obs = np.hstack([obs, np.mean(cluster, axis=0)])
        action = labels[idx]
        cluster = np.vstack([cluster, obs]) if idx > 0 and action == model.ep_as[0][-1] else np.array([obs])
        reward += FLAGS.continuity * (int(model.ep_as[0][-1] == action) - int(model.ep_as[0][-1] != action)) * similarity(batch_obs[idx-1], obs) if idx > 1 else 0
        reward += FLAGS.keyword * keywords[idx][1] if keywords[idx][0] == action else 0
        if idx == len(keywords) - 1:
            reward /= idx + 1
            full_actions = model.ep_as[0] + [action]
            reward += HSMI(batch_obs, full_actions)
        model.store_transition(full_obs, action, 0)
    return reward

def pretrain_pn(model, sess, observations, batch_sizes, keywords, labels):
    st, ed, cnt, loss, ep_rs_total = 0, 0, 0, .0, .0
    while ed < len(keywords):
        st, ed, cnt = ed, ed+batch_sizes[cnt], cnt+1
        reward = determine(model, observations[st:ed], keywords[st:ed], labels[st:ed])
        loss += model.prelearn(sess, reward, FLAGS.keep_prob)
        ep_rs_total += reward
    sess.run(model.epoch_add_op)
    return loss / len(keywords), ep_rs_total / len(keywords), labels
    
def explore(model, sess, batch_obs, keywords):
    reward = [.0] * FLAGS.sample_round
    for n_round in range(FLAGS.sample_round):
        prev_act, cluster = np.zeros(FLAGS.labels, dtype=np.int), np.zeros([1, FLAGS.embed_units])
        for idx, obs in enumerate(batch_obs):
            full_obs = np.hstack([obs, np.mean(cluster, axis=0)])
            action = model.choose_action(sess, full_obs, prev_act, True) 
            prev_act = np.eye(FLAGS.labels, dtype=np.int)[action]
            cluster = np.vstack([cluster, obs]) if idx > 0 and action == model.ep_as[n_round][-1] else np.array([obs])
            reward[n_round] += FLAGS.continuity * (int(model.ep_as[n_round][-1] == action) - int(model.ep_as[n_round][-1] != action)) * similarity(batch_obs[idx-1], obs) if idx > 1 else 0
            reward[n_round] += FLAGS.keyword * keywords[idx][1] if keywords[idx][0] == action else 0
            if idx == len(keywords) - 1:
                reward[n_round] /= idx + 1
                full_actions = model.ep_as[n_round] + [action]
                reward[n_round] += HSMI(batch_obs, full_actions)
            model.store_transition(full_obs, action, n_round)
    return reward
                
def train_pn(model, sess, observations, batch_sizes, keywords):
    st, ed, cnt, loss, ep_rs_total, labels = 0, 0, 0, .0, .0, np.zeros(len(keywords), dtype=np.int)
    while ed < len(keywords):
        st, ed, cnt = ed, ed+batch_sizes[cnt], cnt+1
        reward = explore(model, sess, observations[st:ed], keywords[st:ed])
        labels[st:ed] = model.ep_as[0]
        ep_rs_total += np.sum(reward) / FLAGS.sample_round
        loss += model.learn(sess, reward, FLAGS.keep_prob)
    sess.run(model.epoch_add_op)
    return loss / len(keywords), ep_rs_total / len(keywords), labels

def exploit(model, sess, batch_obs, keywords):
    prev_act, cluster, reward = np.zeros(FLAGS.labels, dtype=np.int), np.zeros([1, FLAGS.embed_units]), .0
    for idx, obs in enumerate(batch_obs):
        full_obs = np.hstack([obs, np.mean(cluster, axis=0)])
        action = model.choose_action(sess, full_obs, prev_act, False)
        prev_act = np.eye(FLAGS.labels, dtype=np.int)[action]
        cluster = np.vstack([cluster, obs]) if idx > 0 and action == model.ep_as[0][-1] else np.array([obs])
        reward += FLAGS.continuity * (int(model.ep_as[0][-1] == action) - int(model.ep_as[0][-1] != action)) * similarity(batch_obs[idx-1], obs) if idx > 1 else 0
        reward += FLAGS.keyword * keywords[idx][1] if keywords[idx][0] == action else 0
        if idx == len(keywords) - 1:
            reward /= idx + 1
            full_actions = model.ep_as[0] + [action]
            reward += HSMI(batch_obs, full_actions)
        model.store_transition(full_obs, action, 0)
    return reward

def develop_pn(model, sess, observations, batch_sizes, keywords):
    st, ed, cnt, loss, ep_rs_total, labels = 0, 0, 0, .0, .0, np.zeros(len(keywords), dtype=np.int)
    while ed < len(keywords):
        st, ed, cnt = ed, ed+batch_sizes[cnt], cnt+1
        reward = exploit(model, sess, observations[st:ed], keywords[st:ed])
        labels[st:ed] = model.ep_as[0]
        loss += model.verify(sess, reward)
        ep_rs_total += reward
    return loss / len(keywords), ep_rs_total / len(keywords), labels

def test(model, sess, batch_obs):
    prev_act, cluster = np.zeros(FLAGS.labels, dtype=np.int), np.zeros([1, FLAGS.embed_units])
    for idx, obs in enumerate(batch_obs):
        full_obs = np.hstack([obs, np.mean(cluster, axis=0)])
        action = model.choose_action(sess, full_obs, prev_act, False)
        prev_act = np.eye(FLAGS.labels, dtype=np.int)[action]
        cluster = np.vstack([cluster, obs]) if idx > 0 and action == model.ep_as[0][-1] else np.array([obs])
        model.store_transition(full_obs, action, 0)
    
def evaluate_pn(model, sess, observations, batch_sizes, keywords, labels):
    st, ed, cnt, acc = 0, 0, 0, .0
    while ed < len(keywords):
        st, ed, cnt = ed, ed+batch_sizes[cnt], cnt+1
        test(model, sess, observations[st:ed])
        acc += np.sum(np.equal(model.ep_as[0], labels[st:ed]))
        model.clean()
    return acc / len(keywords)

def inference_pn(model, sess, observations, batch_sizes, keywords, part):
    st, ed, cnt = 0, 0, 0
    predicted_labels = np.zeros(len(keywords), np.int)
    while ed < len(keywords):
        st, ed, cnt = ed, ed+batch_sizes[cnt], cnt+1
        exploit(model, sess, observations[st:ed], keywords[st:ed])
        predicted_labels[st:ed] = model.ep_as[0]
        model.clean()
    np.savetxt(FLAGS.train_dir_pn+'/labels_'+part, predicted_labels, fmt='%d')
    return predicted_labels