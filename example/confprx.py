import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pickle
from datetime import datetime
import numpy as np
import random


tfhub_handle_preprocess = '../saved/bert-preprocess'
tfhub_handle_encoder = '../saved/small-bert-uncased'
def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(128, activation='relu', name='dense1')(net)
    net = tf.keras.layers.Dense(32, activation='relu', name='dense2')(net)
    net = tf.keras.layers.Dense(6, activation='softmax', name='classifier')(net)
    return tf.keras.Model(text_input, net)

def get_onehot_label(ori_label):
    ori_l = np.array(ori_label)
    onehot = np.zeros((ori_l.size, ori_l.max()+1 ))
    onehot[np.arange(ori_l.size), ori_l] = 1
    return np.array(onehot)
def get_token():
    text_inp = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocess_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocess_layer(text_inp)
    bert_encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False, name='BERT_encoder')
    outputs = bert_encoder(encoder_inputs)
    net = outputs['pooled_output']
    return tf.keras.Model(text_inp, net)

def get_init_data(inp, label, encode, init_num=50):
    init_list = []
    all_idx = np.array([i for i in range(len(encode))])
    rand_idx = random.sample(range(0, len(encode)), init_num)
    
    train_idx = rand_idx[:]
    train_inp = inp[rand_idx]
    train_label = label[rand_idx]
    train_encode = encode[rand_idx]
    
    pool_idx = np.delete(all_idx, rand_idx, axis = 0)
    pool_inp = np.delete(inp, rand_idx, axis=0)
    pool_label = np.delete(label, rand_idx, axis=0)
    pool_encode = np.delete(encode, rand_idx, axis=0)
    
    train_d = {
        'idx': train_idx,
        'inp': train_inp,
        'lab': train_label,
        'enc': train_encode,
    }
    pool_d = {
        'idx': pool_idx,
        'inp': pool_inp,
        'lab': pool_label,
        'enc': pool_encode,
    }
    return train_d, pool_d

def updata_data(query_idx, train_d, pool_d):
    add_idx = pool_d['idx'][query_idx]
    add_inp = pool_d['inp'][query_idx]
    add_lab = pool_d['lab'][query_idx]
    add_enc = pool_d['enc'][query_idx]
    
    train_idx = np.hstack((train_d['idx'], add_idx))
    train_inp = np.hstack((train_d['inp'], add_inp))
    train_lab = np.vstack((train_d['lab'], add_lab))
    train_enc = np.vstack((train_d['enc'], add_enc))
    
    pool_idx = np.delete(pool_d['idx'], query_idx, axis=0)
    pool_inp = np.delete(pool_d['inp'], query_idx, axis=0)
    pool_lab = np.delete(pool_d['lab'], query_idx, axis=0)
    pool_enc = np.delete(pool_d['enc'], query_idx, axis=0)
    
    new_train_d = {
        'idx': train_idx,
        'inp': train_inp,
        'lab': train_lab,
        'enc': train_enc,
    }
    new_pool_d = {
        'idx': pool_idx,
        'inp': pool_inp,
        'lab': pool_lab,
        'enc': pool_enc,
    }
    return new_train_d, new_pool_d
def get_topk_idx(list_,k):
    lst = list_[:]
    index_k = []
    for i in range( k ):
        index_i = lst.index(min(lst))
        index_k.append(index_i)
        lst[index_i] = 99999999
    return index_k

def top_k_recall(y_true, y_pred, k=100):
    true_k_idx = get_topk_idx( y_true, k)
    pred_k_idx = get_topk_idx( y_pred, k)
    same_num = len(set(true_k_idx) & set(pred_k_idx ))
    print(same_num)
    return 1.0*same_num/k

def get_informativeness(model, data):
    pred_soft = model.predict(data)
    info = tf.reduce_max(pred_soft,axis=1).numpy()
    return info
