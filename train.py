import tensorflow as tf  
from tqdm import tqdm
from construct_graph import generateAdjs
import scipy
from utils import *
import models
from models import  *
from sklearn import metrics
import os
import time
import numpy as np
from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(20)
import argparse
################
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--path', default='./save/',type=str,help='Path to the bed file')
parser.add_argument('--dataset', default='GSE11420x_encode',type=str,help='The prefix name of the dataset')
parser.add_argument('--k', default=5,type=int,help='The length of K-mer')
parser.add_argument('--hash', default='fbd7c1da229c4007bf2d7b8a1ba1cf03',type=str,help='The hash of the task')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
configs = tf.ConfigProto()
# config.gpu_options.allow_growth = True
configs.gpu_options.per_process_gpu_memory_fraction = 0.2

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#######################################
##########Load the three adjacy matrices i.e similarity, coexsiting and inclusive matrices################
def loadAdjs(tfids,Kmers):	
    sizes = [Kmers]
    ext = ".npz"
    strsize =""
    for size in sizes:
        strsize+=str(size)
    path = './save/'+tfids+'/adjs'
    if not os.path.exists(path):
        os.mkdir(path)
    filename = path+"/"+tfids + "_encode_" + strsize + "_" + "adj_inclu" + ext
    if os.path.exists(filename):
        labelname = path+"/"+tfids + "_encode_" + strsize + "_" + "labels.txt.npy"
        labels = np.load(labelname)
        adj_inclu = scipy.sparse.load_npz(filename)
        filename = path+"/"+tfids + "_encode_" + strsize + "_" + "adj_sim" + ext
        adj_sim = scipy.sparse.load_npz(filename)
        filename = path+"/"+tfids + "_encode_" + strsize + "_" + "adj_coo" + ext
        adj_coo = scipy.sparse.load_npz(filename)
    else:
        adj_inclu, adj_sim, adj_coo, labels = generateAdjs(tfids,Kmers)
        filename = path+"/"+tfids + "_encode_" + strsize + "_" + "adj_inclu" + ext
        scipy.sparse.save_npz(filename, adj_inclu)
        filename = path+"/"+tfids + "_encode_" + strsize + "_" + "adj_sim" + ext
        scipy.sparse.save_npz(filename, adj_sim)
        filename = path+"/"+tfids + "_encode_" + strsize + "_" + "adj_coo" + ext
        scipy.sparse.save_npz(filename, adj_coo)
        labelname = path+"/"+tfids + "_encode_" + strsize + "_" + "labels.txt"
        np.save(labelname,labels)
    return adj_inclu, adj_sim, adj_coo, labels
###################training the MMGraph tools########################################	
def one_task(name,Kmers,adj_coo, adj_sim, adj_inclu, idx_train, idx_val, idx_test, idx_all, labels, options, save=False):
    n = len(labels)
    test_mask = np.zeros(n,dtype=np.int)
    val_mask = np.zeros(n,dtype=np.int)
    train_mask = np.zeros(n,dtype=np.int)
    train_mask[idx_train] = 1
    val_mask[idx_val] = 1
    test_mask[idx_test] = 1
    Wcoo = preprocess_adj(adj_coo) 
    Wsim = preprocess_adj(adj_sim) 
    Ect = scipy.sparse.identity(adj_coo.shape[0])
    Ect = preprocess_features(Ect)
    Est = scipy.sparse.identity(adj_sim.shape[0])
    Est = preprocess_features(Est)
    Winclu = adj_inclu ##inclusive array

    placeholders = {
        'Wcoo':tf.compat.v1.placeholder(tf.float32),
        'Wsim':tf.compat.v1.placeholder(tf.float32),
        'Ect':tf.compat.v1.sparse_placeholder(tf.float32, shape=None),
        'Est':tf.compat.v1.sparse_placeholder(tf.float32, shape=None),
        'Winclu':tf.compat.v1.placeholder(tf.float32, shape=None), #inclusive
        'labels':tf.compat.v1.placeholder(tf.float32, shape=None),
        'labels_mask': tf.compat.v1.placeholder(tf.bool),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'training': tf.compat.v1.placeholder_with_default(0., shape=())
    }
    
    # #build the model
    model = dGAT("gat", placeholders, Ect[2][1], options)
    # #Initializing session
    sess = tf.Session(config=configs)
    
    # # define model evaluation function
    def evaluate(Ect, Est, Winclu, Wcoo, Wsim, label, mask, placeholders):
        feed_dict_val = construct_feed_dict(
            Ect, Est,Winclu, Wcoo, Wsim, label, mask, placeholders)
        loss,preds,labels, seq_hidden, kmer_hidden_sim,kmer_hidden_co,attention_sim,attention_co,kmer_attention_sim,kmer_attention_co = sess.run([model.loss, model.preds, model.labels,model.seq_hidden, model.kmer_hidden_sim,model.kmer_hidden_co,model.attention_sim,model.attention_co,model.kmer_attention_sim,model.kmer_attention_co], feed_dict=feed_dict_val)
        return loss,preds,labels,seq_hidden, kmer_hidden_sim,kmer_hidden_co,attention_sim,attention_co,kmer_attention_sim,kmer_attention_co
    sess.run(tf.global_variables_initializer())
    # train model
    feed_dict = construct_feed_dict(Ect, Est, Winclu, Wcoo, Wsim, labels, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']:options['dropout']})
    for epoch in tqdm(range(options['epochs']+1)):
        sess.run([model.opt_op, model.loss, model.preds,model.seq_hidden, model.kmer_hidden_sim,model.kmer_hidden_co,model.attention_sim,model.attention_co], feed_dict=feed_dict)
        if epoch % 2 == 0:
            val_loss, preds, labels,_,_,_,_,_,_,_= evaluate(Ect, Est,Winclu, Wcoo, Wsim, labels, val_mask, placeholders)
            # print(preds)
            val_auc,fpr,tpr,thresholds = com_auc(labels,preds,idx_val)
            print("epoch %d: valid loss = %f, val_auc = %f" %(epoch, val_loss, val_auc))
            
        test_loss, preds, labels,seq_hidden, kmer_hidden_sim,kmer_hidden_co,attention_sim,attention_co,kmer_attention_sim,kmer_attention_co= evaluate(Ect, Est,Winclu, Wcoo, Wsim, labels, test_mask, placeholders)
        test_auc,fpr,tpr,thresholds = com_auc(labels, preds,idx_test)
        print("epoch %d: test_auc = %f" %(epoch, test_auc))
        path = './download/'+name 
        if not os.path.exists(path):
            os.mkdir(path)
        path = './save/'+name+'/TFBS'
        if not os.path.exists(path):
            os.mkdir(path)
        seq_path = path+'/'+name+'_encode'+str(Kmers)+'_seq'
        np.save(seq_path, seq_hidden[-1])
        kmer_path = path+'/'+name+'_encode'+str(Kmers)+'_kmer_sim'
        np.save(kmer_path, kmer_hidden_sim[-1])
        kmer_path = path+'/'+name+'_encode'+str(Kmers)+'_kmer_co'
        np.save(kmer_path, kmer_hidden_co[-1])
        kmer_path = path+'/'+name+'_encode'+str(Kmers)+'_attention_sim'
        np.save(kmer_path, kmer_attention_sim[-1])
        kmer_path = path+'/'+name+'_encode'+str(Kmers)+'_kmer_attention_co'
        np.save(kmer_path, kmer_attention_co[-1])


        path = './save/'+name+'/output'
        if not os.path.exists(path):
            os.mkdir(path)
        
        out_test=path+'/'+name+'_encode_test.txt'
        np.savetxt(out_test,[np.squeeze(labels[idx_test,:]), np.squeeze(preds[idx_test,:])])
        out_val=path+'/'+name+'_encode_val.txt'
        np.savetxt(out_val,[np.squeeze(labels[idx_val,:]), np.squeeze(preds[idx_val,:])])
###################################################################################
def motif_task(args):
    dataset = args.dataset
    Kmers = int(args.k)
    # Settings
    options = {}
    options['model'] = 'gat'
    options['epochs'] = 300
    options['dropout'] = 0.3
    options['weight_decay'] = 0.001
    options['hidden1'] = 100
    options['learning_rate'] = 0.02 
    ########################
    # tfid = args.dataset
    tfid = args.hash
    adj_inclu, adj_sim, adj_coo, labels = loadAdjs(tfid,Kmers)
    n = len(labels)
    #########################################
    adj_inclu = adj_inclu.toarray()
    idx_all = np.array([i for i in range(n)],dtype='int')
    idx_train = idx_all[:int(0.8*n)]
    idx_val = idx_all[int(0.8*n):int(0.9*n)]
    idx_test = idx_all[int(0.9*n):]
    options['hidden_shape'] = adj_coo.toarray().shape
    one_task(args.hash,Kmers,adj_coo, adj_sim, adj_inclu, idx_train, idx_val, idx_test, idx_all, labels, options, True)
##########################main##############
if __name__=='__main__':
    start_time = time.time()
    motif_task(args)
    end_time=time.time()
    total_time=np.array(end_time-start_time)
    print('total_time:',total_time)

