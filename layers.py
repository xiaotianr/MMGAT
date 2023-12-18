import tensorflow as tf
from inits import *

def dot(x, y, sparse=True):
	if sparse:
		res = tf.sparse_tensor_dense_matmul(x, y)
	else:
		res = tf.matmul(x, y)
	return res
def sparse_dropout(x, keep_prob):
	keep_tensor = keep_prob + tf.random_uniform(tf.shape(x))
	drop_mask = tf.cast(tf.floor(keep_tensor), dtype=tf.bool)
	out = tf.sparse_retain(x, drop_mask)
	return out * (1.0/keep_prob)
#############the first lyaer of GNN#############

class GATLayer1(tf.keras.layers.Layer):
    def __init__(self,in_features, out_features, dropout, alpha, concat=True,stdv=0.2):
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = normal([in_features, out_features],stdv=stdv) 
        self.a = normal([2*out_features,1],stdv=stdv) 
        self.leakyrelu = tf.keras.layers.LeakyReLU(alpha)

    def __call__(self, inputs,adj):
        h = inputs
        Wh = dot(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * tf.ones_like(e)
        attention = tf.where(adj > 0, e, zero_vec)
        attention = tf.nn.softmax(attention, axis=1)
        attention_d = tf.nn.dropout(attention, rate=self.dropout)
        h_prime = tf.matmul(attention_d, Wh)
        return tf.keras.activations.elu(h_prime),attention
        

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = tf.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = tf.matmul(Wh, self.a[self.out_features:, :])
        e = tf.add(Wh1, tf.transpose(Wh2))
        return self.leakyrelu(e)
class GATLayer2(tf.keras.layers.Layer):
    def __init__(self,in_features, out_features, dropout, alpha, concat=True,stdv=0.01):
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = normal([in_features, out_features],stdv=stdv) 
        self.leakyrelu = tf.keras.layers.LeakyReLU(alpha)

    def __call__(self, h_s,h_k1,h_k2):
        # h_s = tf.sparse.to_dense(h_s)
        Wh = tf.matmul(h_s, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e1 = self.leakyrelu(tf.matmul(Wh, tf.transpose(h_k1)))
        e2 = self.leakyrelu(tf.matmul(Wh, tf.transpose(h_k2)))
        zero_vec = -9e15 * tf.ones_like(e1)
        attention1 = tf.where(h_s > 0, e1, zero_vec)
        attention1 = tf.nn.softmax(attention1, axis=-1)
        attention1_d = tf.nn.dropout(attention1, rate=self.dropout)
        h_prime1 = tf.matmul(attention1_d, h_k1)
        attention2 = tf.where(h_s > 0, e2, zero_vec)
        attention2 = tf.nn.softmax(attention2, axis=-1)
        attention2_d = tf.nn.dropout(attention2, rate=self.dropout)
        h_prime2 = tf.matmul(attention2_d, h_k2)
        h_prime=tf.add(h_prime1,h_prime2)
        return tf.keras.activations.elu(h_prime),attention1,attention2
        
class GATLayer3(tf.keras.layers.Layer):
    def __init__(self,in_features, out_features,stdv=0.2):
        self.in_features = in_features
        self.out_features = out_features

        self.W = normal([in_features, out_features],stdv=stdv) 
        self.b = zeros([out_features]) 

    def __call__(self, inputs):
        outputs=[]
        output=tf.matmul(inputs,self.W)
        outputs.append(output)
        outputs+=self.b
        return outputs




