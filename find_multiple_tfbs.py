# -*- coding: utf-8 -*-
import math
import scipy
from utils import *
from load_data import load_encode_test, load_encode_train,load_motif_seq
import numpy as np
import time
import Levenshtein
import os
from scipy.spatial.distance import cosine
from com_mi import calc_MI
import sys
import argparse
from numpy.random import seed
seed(123)
import tensorflow as tf  
tf.compat.v1.set_random_seed(123)
from tensorflow import set_random_seed
################
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--hash', default='fbd7c1da229c4007bf2d7b8a1ba1cf03',type=str,help='The The hash of the task')
parser.add_argument('--path', default='./save/',type=str,help='Path to the bed file')
parser.add_argument('--dataset', default='GSE11420x_encode',type=str,help='The prefix name of the dataset')
args = parser.parse_args()
# ##############################
def construct_vocab(seqs, size, thresh = 3):#
    word_vocab = {}
    word_freq = {}
    n_sqs = len(seqs)
    for i in range(n_sqs):
        subseqs=[]
        seq=seqs[i]
        subseqs = [seq[t:t+size] for t in range(0, len(seq)-size+1)]
        word_vocab[seq]=subseqs
        for subseq in subseqs:
            freq_keys=list(word_freq.keys())
            if subseq not in freq_keys:
                word_freq.setdefault(subseq,1)
            else:
                word_freq[subseq]+=1
        subseqs=[]
    return word_vocab,word_freq
def kmer_seq(word_vocab,word_freq,seqs):
    keys=list(word_freq.keys())#KMERS
    keys_seqs={}
    for kmer in keys:
        kseqs = [seq for seq  in seqs if kmer in word_vocab[seq]]
        keys_seqs[kmer]=kseqs
    return keys_seqs
###################################
def tf_ks(seq,size=5):
    sword_freq = {}
    for i in range(0,len(seq)-size+1,1):
        subseq = seq[i:i+size]
        if subseq not in sword_freq:
            sword_freq[subseq] = 1
        else:
            sword_freq[subseq]+=1
    return sword_freq
#####################################
def standardization(data):
	data = np.array(data)
	mu = np.mean(data, axis=0)
	sigma = np.std(data, axis=0)
	nor_data = (data-mu)/sigma 
	nor_data = list(nor_data)
	return nor_data
###################co edge###########
def Acoo_value(nums,K12,K1,K2):    
    if K12>0:
        pk1k2 = K12/nums
        pk1 = (K1+K12)/nums
        pk2 = (K2+K12)/nums
        Acoo_k1k2 = np.log(pk1k2/(pk1*pk2+1e-3))
    else:
        Acoo_k1k2 = 0
    weight = Acoo_k1k2
    return weight
########################################   
def Acoo(seqs,word_freq,word_vocab,threshholds=0):
    keys=list(word_freq.keys())#KMERS
    rows_cols = [[i,j] for i in range(len(keys)) for j in range(i, len(keys))]
    kmers = [[keys[i],keys[j]] for i in range(len(keys)) for j in range(i, len(keys))]
    rows=np.random.rand(len(kmers))
    cols=np.random.rand(len(kmers))
    weights=np.random.rand(len(kmers))
    kmer_seqs = kmer_seq(word_vocab,word_freq,seqs)
    for t in range(len(kmers)):
        K12 = len(set(kmer_seqs[kmers[t][0]]) & set(kmer_seqs[kmers[t][1]]))
        if K12>0:
            K1 = len(set(kmer_seqs[kmers[t][0]]))
            K2 = len(set(kmer_seqs[kmers[t][1]]))
            weight = Acoo_value(len(seqs),K12,K1,K2)
        else:
            weight = 0
        weights[t] = weight
        rows[t]= rows_cols[t][0]
        cols[t] = rows_cols[t][1]
    node_size = len(keys)
    weights = standardization(weights) #
    adj = scipy.sparse.csr_matrix((weights, (list(rows), list(cols))),shape=(node_size, node_size))
    Acoo = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return Acoo
#############################################
def generateAdjs(tfids,Kmers=5):
    threshholds = Kmers
    pos_test_seqs, neg_test_seqs, test_seqs = load_encode_test(tfid)
    pos_train_seqs, neg_train_seqs,train_seqs = load_encode_train(tfid)
    seqs = train_seqs + test_seqs
    pos_seqs = pos_train_seqs + pos_test_seqs
    neg_seqs = neg_train_seqs + neg_test_seqs
    pos_seqs = [seq[0] for seq in pos_seqs]
    neg_seqs = [seq[0] for seq in neg_seqs]
    seqs = [seq[0] for seq in seqs]
    word_vocab, word_freq = construct_vocab(seqs, size=Kmers)
    neg_word_vocab, neg_word_freq = construct_vocab(neg_seqs, size=Kmers)
    strsize=str(Kmers)
    adj_coo = Acoo(pos_seqs, word_freq, word_vocab, threshholds=0)
    return  adj_coo.toarray(), word_vocab, word_freq, neg_word_freq, pos_seqs, neg_seqs,seqs


def load_embed(name,Kmers):
    seq_path = './save/'+name+'/TFBS/'+name+'_encode'+str(Kmers)+'_seq.npy'
    kmer_sim_path = './save/'+name+'/TFBS/'+name+'_encode'+str(Kmers)+'_kmer_sim.npy'
    kmer_co_path = './save/'+name+'/TFBS/'+name+'_encode'+str(Kmers)+'_kmer_co.npy'
    att_sim_path = './save/'+name+'/TFBS/'+name+'_encode'+str(Kmers)+'_attention_sim.npy'
    att_co_path = './save/'+name+'/TFBS/'+name+'_encode'+str(Kmers)+'_attention_co.npy'
    seq_embed = np.load(seq_path)
    kmer_embed_sim = np.load(kmer_sim_path)
    kmer_embed_co = np.load(kmer_co_path)
    att_sim=np.load(att_sim_path)
    att_co=np.load(att_co_path)
    return seq_embed,kmer_embed_sim,kmer_embed_co,att_sim,att_co
def construct_data(data,values):
    datas=dict()
    for i in range(len(data)):
        datas[data[i]]=values[i,:]
    return datas
##### merge candidate TFBSs###############
def Merges(tfid):
    inputname='./save/'+tfid+'/motifs/'+tfid+'_encode.fa'
    FileR = open(inputname,"r")
    out_path = './download/'+tfid+'/TFBSs'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_name = './download/'+tfid+'/TFBSs/'+tfid+'_encode_merge.fa'
    FileW = open(out_name,"w+")
    done = 0
    x = FileR.readline()
    y = FileR.readline()
    i = int(x.split("_")[1])
    start = int(x.split("_")[2])
    end = int(x.split("_")[3])
    temp = y
    while(not done):
        x = FileR.readline()
        y = FileR.readline()
        if(x != ''):
            if(int(x.split("_")[1]) == i):
                if(int(end - int(x.split("_")[2])) >= 0):
                    cha = end - int(x.split("_")[2])
                    temp = temp.strip() + y[cha:]
                    end = int(x.split("_")[3])
                else:
                    FileW.write(">seq_"+ str(i) +"_" + str(start) + "_" + str(end) + "\n")
                    FileW.write(temp.strip() + "\n")
                    start = int(x.split("_")[2])
                    end = int(x.split("_")[3])
                    temp = y
            else:
                FileW.write(">seq_"+ str(i) +"_" + str(start) + "_" + str(end) + "\n")
                FileW.write(temp.strip() + "\n")
                i = int(x.split("_")[1])
                start = int(x.split("_")[2])
                end = int(x.split("_")[3])
                temp = y
        else:
            done = 1
    FileR.close()
    FileW.close()
#######
def construct_att_data(seq,kmer,values):
    print(len(seq),len(kmer),values.shape)
    datas = {s: {k: values[i][j] for j, k in enumerate(kmer)} for i, s in enumerate(seq)}
    return datas
def TFBSs(tfid, Kmers):
    seq_embed,kmer_embed_sim,kmer_embed_co,att_sim,att_co = load_embed(tfid, Kmers)
    adj_coo, word_vocab, word_freq,neg_word_freq, pos_seqs,neg_seqs,seqs = generateAdjs(tfid,Kmers)
    all_kmers = list(word_freq.keys())
    neg_all_kmers = list(neg_word_freq.keys())
    kmer_s=list(word_freq.keys())
    kmers_sim_data=construct_att_data(seqs,kmer_s,att_sim)
    kmers_co_data=construct_att_data(seqs,kmer_s,att_co)
    neg_sim_att=[]
    neg_co_att=[]
    #1  calcualte MI
    for j in range(len(neg_seqs)):
        neg_seq_mers = word_vocab[neg_seqs[j]]
        for k in range(len(neg_seq_mers)):
            if neg_seq_mers[k] in kmer_s:
                neg_sim = kmers_sim_data[neg_seqs[j]][neg_seq_mers[k]]
                neg_sim_att.append(neg_sim)
            if neg_seq_mers[k] in kmer_s:
                neg_co = kmers_co_data[neg_seqs[j]][neg_seq_mers[k]]
                neg_co_att.append(neg_co)
    neg_average_sim_att = np.mean(np.array(neg_sim_att))
    neg_average_co_att = np.mean(np.array(neg_co_att))
####################################
    ranges=[neg_average_sim_att,neg_average_co_att]
    
    path = './save/'+tfid+'/motifs'
    if not os.path.exists(path):
        os.mkdir(path)
    
    seqname = path+'/'+tfid+'_encode.txt' # store TFBS in txt file
    faname = path+'/'+tfid+'_encode.fa'# store TFBS in fasta file
    file1=open(seqname,'w+')
    file2=open(faname,'w+')
    count =0
    for i in range(len(pos_seqs[:])):
        seq_mers = word_vocab[pos_seqs[i]]
        for j in range(len(seq_mers)):
            att_sim =kmers_sim_data[pos_seqs[i]][seq_mers[j]] ##att_sim(p,i)
            att_co = kmers_co_data[pos_seqs[i]][seq_mers[j]] ##att_co(p,i)
            #2 get denoised dnMI
            if att_sim - ranges[0] >0 or att_co-ranges[1]>0: 
                cp_index = int(j+math.ceil((Kmers-1)/2)) #startkp
                #3 get the Kseed
                right_frag = int(cp_index+Kmers)
                left_frag = int(cp_index-Kmers)                 
                if right_frag < 101 and left_frag>0:
                    #4 find multiple TFBSs with different lengths
                    kr_index = all_kmers.index(pos_seqs[i][cp_index:right_frag]) #kr(p,i)
                    kl_index = all_kmers.index(pos_seqs[i][left_frag:cp_index]) #kl(p,i)
                    coexisting_proba = adj_coo[kl_index, kr_index] ##coexisting probability of kl(p,i) and kr(p,i)
                    if coexisting_proba > 0.5:
                        seq = pos_seqs[i][left_frag:right_frag]
                        file1.writelines(seq+'\n')
                        strs='>'+'seq'+'_'+str(i)+'_'+str(left_frag)+'_'+str(right_frag)+'\n'
                        file2.writelines(strs)
                        file2.writelines(seq+'\n')
                        count += 1
    file1.close()
    file2.close()
    Merges(tfid) ##### merge overlaped TFBSs to a longer TFBS.
if __name__=='__main__':
    tfid = args.hash
    print('Find multiple motifs.....')
    TFBSs(tfid, Kmers=5)
    print('Done!')
