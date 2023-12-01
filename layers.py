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
class GraphConvolution():
	def __init__(self, name, input_dim, output_dim, placeholders, use_dropout=True,
		support_id = None, sparse_inputs=False, act = tf.nn.relu, bias=True, concat = True):
		self.act = act
		self.sparse_inputs = sparse_inputs
		self.concat = concat

		self.debug = None
		self.placeholders = placeholders
		self.use_dropout = use_dropout
		if self.use_dropout:
			self.dropout = 0.00001
		else:
			self.dropout = tf.constant(0.0)
		self.name = name
		self.bias = bias
		if support_id is None:
			self.support = placeholders['support0']
		else:
			self.support = placeholders['support' + str(support_id)]
		self.vars={}
		with tf.variable_scope(self.name +'_vars'):
			for i in range(len(self.support)):
				if concat:
					tmp = int(output_dim/(1.0 * len(self.support)))
				else:
					tmp = output_dim
				self.vars['weights_'+str(i)] = normal([input_dim, tmp], name='weights_' + str(i))
			if self.bias:
				self.vars['bias'] = zeros([output_dim], name='bias')

	def __call__(self, inputs):
		x = inputs
		if self.use_dropout:			
			x = sparse_dropout(x, 1-self.dropout)

		outputs=[]
		for i in range(len(self.support)):
			pre_sup = dot(x, self.vars['weights_'+str(i)],sparse=True)
			output = dot(self.support[i], pre_sup,sparse=True)
			outputs.append(output)
		if self.concat:
			outputs=tf.concat(outputs,axis=-1)
		else:
			outputs=tf.add_n(outputs)/(1.0 * len(self.support))
		
		if self.bias:
			outputs+=self.vars['bias']
		outputs = tf.layers.batch_normalization(outputs,axis = 1)
		return self.act(outputs)
##############the second layer of GNN#######################
class GraphConvolution2():
	def __init__(self, placeholders,hidden_shape=[1024, 1024],hidden_dim=800,output_dim=1,name='GAT', use_dropout=False,
		sparse_inputs=False, act = tf.nn.relu, bias=True):
		self.act = act
		self.sparse_inputs = sparse_inputs

		self.placeholders = placeholders
		self.dropout = placeholders['dropout']
		self.use_dropout = use_dropout
		if self.use_dropout:
			self.dropout = placeholders['dropout']
		else:
			self.dropout = tf.constant(0.0)
		self.name = name
		self.bias = bias

		self.vars={}
		
		self.vars['weights_'+str(20)] = normal(hidden_shape, name='weights_' + str(20))
		self.vars['weights_'+str(21)] = normal([hidden_dim, output_dim], name='weights_' + str(21))
		if self.bias:
				self.vars['bias'] = zeros([output_dim], name='bias')

	def __call__(self, input1,input2):
		x1 = input1
		x2 = input2#1024*800
		if self.use_dropout:
			x = tf.nn.dropout(x1, 1 - self.dropout)
		outputs=[]
		pre_sup = tf.matmul(x1, self.vars['weights_'+str(20)], b_is_sparse=False)#1996*1024
		
		pre_sup = dot(pre_sup, x2,sparse=False)#1996*800
		pre_sup=self.act(pre_sup)
		output = dot(pre_sup,self.vars['weights_'+str(21)],sparse=False)
		# output = tf.layers.batch_normalization(output,axis = 1)
		outputs.append(output)
		
		if self.bias:
			outputs+=self.vars['bias']
		# return self.act(outputs),pre_sup # OUTPU AND embedding
		return outputs,pre_sup

class GATLayer2(tf.keras.layers.Layer):
    def __init__(self,in_features, out_features, dropout, alpha, concat=True):
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = normal([in_features, out_features]) 
        self.a = normal([2*out_features,1]) 
        self.leakyrelu = tf.keras.layers.LeakyReLU(alpha)

    def __call__(self, inputs,adj):
        h = inputs
        Wh = dot(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * tf.ones_like(e)
        attention = tf.where(adj > 0, e, zero_vec)
        attention = tf.nn.softmax(attention, axis=1)
        attention = tf.nn.dropout(attention, rate=self.dropout)
        h_prime = tf.matmul(attention, Wh)

        if self.concat:
            return tf.keras.activations.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = tf.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = tf.matmul(Wh, self.a[self.out_features:, :])
        e = tf.add(Wh1, tf.transpose(Wh2))
        return self.leakyrelu(e)


# #############the first lyaer of GNN#############
# class GraphConvolution():
# 	def __init__(self, name, input_dim, output_dim, placeholders, use_dropout=True,
# 		support_id = None, sparse_inputs=False, act = tf.nn.relu, bias=True, concat = True):
# 		self.act = act
# 		self.sparse_inputs = sparse_inputs
# 		self.concat = concat

# 		self.debug = None
# 		self.placeholders = placeholders
# 		self.use_dropout = use_dropout
# 		if self.use_dropout:
# 			self.dropout = 0.00001
# 		else:
# 			self.dropout = tf.constant(0.0)
# 		self.name = name
# 		self.bias = bias
# 		if support_id is None:
# 			self.support = placeholders['support0']
# 		else:
# 			self.support = placeholders['support' + str(support_id)]
# 		self.vars={}
# 		with tf.variable_scope(self.name +'_vars'):
# 			for i in range(len(self.support)):
# 				if concat:
# 					tmp = int(output_dim/(1.0 * len(self.support)))
# 				else:
# 					tmp = output_dim
# 				self.vars['weights_'+str(i)] = normal([input_dim, tmp], name='weights_' + str(i))
# 			if self.bias:
# 				self.vars['bias'] = zeros([output_dim], name='bias')

# 	def __call__(self, inputs):
# 		x = inputs
# 		if self.use_dropout:			
# 			x = sparse_dropout(x, 1-self.dropout)

# 		outputs=[]
# 		for i in range(len(self.support)):
# 			pre_sup = dot(x, self.vars['weights_'+str(i)],sparse=True)
# 			output = dot(self.support[i], pre_sup,sparse=True)
# 			outputs.append(output)
# 		if self.concat:
# 			outputs=tf.concat(outputs,axis=-1)
# 		else:
# 			outputs=tf.add_n(outputs)/(1.0 * len(self.support))
		
# 		if self.bias:
# 			outputs+=self.vars['bias']
# 		outputs = tf.layers.batch_normalization(outputs,axis = 1)
# 		return self.act(outputs)
# ##############the second layer of GNN#######################
# class GraphConvolution2():
# 	def __init__(self, placeholders,hidden_shape=[1024, 1024],hidden_dim=800,output_dim=1,name='GCN', use_dropout=False,
# 		support_id = None, sparse_inputs=False, act = tf.nn.relu, bias=True, concat = True):
# 		self.act = act
# 		self.sparse_inputs = sparse_inputs
# 		self.concat = concat

# 		self.debug = None
# 		self.placeholders = placeholders
# 		self.dropout = placeholders['dropout']
# 		self.use_dropout = use_dropout
# 		if self.use_dropout:
# 			self.dropout = placeholders['dropout']
# 		else:
# 			self.dropout = tf.constant(0.0)
# 		self.name = name
# 		self.bias = bias

# 		self.vars={}
		
# 		with tf.variable_scope(self.name +'_vars'):
# 			self.vars['weights_'+str(20)] = normal(hidden_shape, name='weights_' + str(20))
# 			self.vars['weights_'+str(21)] = normal([hidden_dim, output_dim], name='weights_' + str(21))
# 			self.vars['weights_'+str(22)] = normal([hidden_dim, hidden_dim], name='weights_' + str(22))
# 			if self.bias:
# 				self.vars['bias'] = zeros([output_dim], name='bias')

# 	def __call__(self, input1,input2):
# 		x1 = input1
# 		x2 = input2#1024*800
# 		if self.use_dropout:
# 			x = tf.nn.dropout(x1, 1 - self.dropout)
# 		outputs=[]
# 		pre_sup = tf.matmul(x1, self.vars['weights_'+str(20)], b_is_sparse=False)#1996*1024
		
# 		pre_sup = dot(pre_sup, x2,sparse=False)#1996*800
# 		output = dot(pre_sup,self.vars['weights_'+str(21)],sparse=False)
# 		# output = tf.layers.batch_normalization(output,axis = 1)
# 		outputs.append(output)
		
# 		if self.bias:
# 			outputs+=self.vars['bias']
# 		return self.act(outputs),pre_sup # OUTPU AND embedding


