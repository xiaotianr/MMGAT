import numpy as np   
import tensorflow as tf   
from layers import *
### Sigmoid function
def masked_sigmoid_cross_entropy(preds, y, mask):
	mask = tf.cast(mask, dtype=tf.float32)
	mask = tf.expand_dims(mask,-1)
	y = tf.multiply(y, mask)
	preds = tf.multiply(preds, mask)
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=preds)
	return tf.reduce_mean(loss)

def masked_dice_cross(preds, y, mask):
	mask = tf.cast(mask, dtype=tf.float32)
	mask = tf.expand_dims(mask,-1)
	y = tf.multiply(y, mask)
	preds = tf.multiply(preds, mask)
	smooth = 1.e-5
	smooth_tf = tf.constant(smooth, tf.float32)
	pred_flat = tf.cast(preds, tf.float32)
	true_flat = tf.cast(y, tf.float32)
	intersection = tf.reduce_sum(pred_flat * true_flat, axis=-1) + smooth_tf
	loss = intersection / tf.multiply(tf.norm(pred_flat, ord=2), tf.norm(true_flat, ord=2))
	# denominator = tf.reduce_sum(pred_flat, axis=-1) + tf.reduce_sum(true_flat, axis=-1) + smooth_tf
	# loss = 1 - tf.reduce_mean(intersection / denominator)
	return loss

class dGAT():
	def __init__(self, name, placeholders, input_dim, options):
		self.input_dim = input_dim
		# self.output_dim = placeholders['labels'].get_shape().as_list()[1]
		self.placeholders = placeholders
		lr = options['learning_rate']
		self.options = options
		self.name = name
		self.vars={}

		self.Ect = placeholders['Ect']
		self.Est = placeholders['Est']
		self.Winclu = placeholders['Winclu']
		self.outputs = None
		self.seq_hidden = []
		self.kmer_hidden = []

		self.loss = 0

		lr = tf.compat.v1.train.natural_exp_decay(options['learning_rate'], options['epochs'], 5, 0.001, staircase = False)
		self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = lr,beta1=0.5,beta2=0.5,epsilon=1e-03)
		
		self.opt_op = None
		self.build()

	def _build(self):
		#  The first layer to learn the k-mer embedding from coexisting graph
		self.gnnlayers_coo=GATLayer2(
			in_features = self.input_dim,
			out_features = self.options['hidden1'],
			dropout =self.placeholders['dropout'],
			alpha=0.2,
		)
		#  The first layer to learn the k-mer embedding from similarity graph
		self.gnnlayers_sim=GATLayer2(
			in_features = self.input_dim,
			out_features = self.options['hidden1'],
			dropout = self.placeholders['dropout'],
			alpha=0.2,
		)
		#  The second layer to learn the sequence embedding of inclusive graph		
		self.gnnlayers_inclu=GraphConvolution2(
			output_dim = 1,
			hidden_shape = self.options['hidden_shape'],
			hidden_dim = self.options['hidden1']*2,
			placeholders = self.placeholders,
			act = tf.nn.relu,
			use_dropout=True,
			sparse_inputs = True,
			bias = True,
			)

	def build(self):
		self._build()

		Ec = self.gnnlayers_coo(self.Ect,self.placeholders['Wcoo'])
		Es = self.gnnlayers_sim(self.Est,self.placeholders['Wsim'])
		# 扁平化 Ec 和 Es

		kmer_hidden = tf.concat([Ec,Es], axis=-1)
		hidden,seq_hidden = self.gnnlayers_inclu(self.Winclu,kmer_hidden)
		self.seq_hidden.append(seq_hidden)
		self.kmer_hidden.append(kmer_hidden)

		self.outputs = tf.squeeze(input=hidden,axis=[0])
		self._loss()
		self.opt_op = self.optimizer.minimize(self.loss)
	def _loss(self):
		self.loss = masked_sigmoid_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
		self.preds = self.outputs
		self.labels = self.placeholders['labels']
