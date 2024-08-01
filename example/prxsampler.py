import copy, time
import numpy as np 
import tensorflow as tf
from module.confprx import get_informativeness

def sample_pooling(i,N0,w,W,delete_num, select_proxy, pool_data, sample_pool_num, begin, end, select_time, train_idx, train_inp, train_lab, train_enc, train_idx0, train_inp0, train_lab0, train_enc0,):
    tp1 = time.time()
#------ pipeline -------
    w_pool_idx, w_pool_inp, w_pool_lab, w_pool_enc  = copy.deepcopy(pool_data['idx'][begin:end]), copy.deepcopy(pool_data['inp'][begin:end]),copy.deepcopy(pool_data['lab'][begin:end]),copy.deepcopy(pool_data['enc'][begin:end])
#------ data selection  -------
    pool_info_prd = get_informativeness(select_proxy, w_pool_enc)
    sorted_id = sorted(range(len(pool_info_prd)), key=lambda k: pool_info_prd[k], reverse=False) 
    tp2 = time.time()
    timeps = tp2 - tp1
    select_time = select_time + timeps
    query_id = sorted_id[:sample_pool_num]
#-----  pooling：
    if i <= N0:
        del_pooling_id,del_query_id = copy.deepcopy(sorted_id[-delete_num:]),copy.deepcopy(query_id)
        delete_id = del_pooling_id + del_query_id
    else:
        del_query_id = copy.deepcopy(query_id)
        delete_id = del_query_id            
    add_idx,add_inp,add_lab,add_enc = w_pool_idx[query_id],w_pool_inp[query_id],w_pool_lab[query_id],w_pool_enc[query_id]
    train_num = len(train_idx)
    train_num_step = train_num//W
    if w != W-1:
        train_num_begin, train_num_end = w*train_num_step, (w+1)*train_num_step
    else:
        train_num_begin,train_num_end = w*train_num_step, train_num
    pipe_train_idx, pipe_train_inp, pipe_train_lab, pipe_train_enc =  np.hstack((train_idx0[train_num_begin:train_num_end], add_idx)),np.hstack((train_inp0[train_num_begin:train_num_end], add_inp)), np.vstack((train_lab0[train_num_begin:train_num_end], add_lab)), np.vstack((train_enc0[train_num_begin:train_num_end], add_enc))  
    train_idx, train_inp, train_lab, train_enc = np.hstack((train_idx, add_idx)), np.hstack((train_inp, add_inp)), np.vstack((train_lab, add_lab)), np.vstack((train_enc, add_enc))
    w_pool_idx, w_pool_inp, w_pool_lab, w_pool_enc = np.delete(w_pool_idx, delete_id, axis=0), np.delete(w_pool_inp, delete_id, axis=0), np.delete(w_pool_lab, delete_id, axis=0), np.delete(w_pool_enc, delete_id, axis=0)  
    return w_pool_idx, w_pool_inp, w_pool_lab, w_pool_enc, train_idx, train_inp, train_lab, train_enc, pipe_train_idx, pipe_train_inp, pipe_train_lab, pipe_train_enc,select_time  


def proxy():
    inp = tf.keras.layers.Input(shape=(512,))
    x = tf.keras.layers.Dense(128, activation='relu')(inp)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inp, x)