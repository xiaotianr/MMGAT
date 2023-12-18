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
		self.kmer_hidden_sim = []
		self.kmer_hidden_co = []
		self.attention_sim = []
		self.attention_co = []
		self.kmer_attention_sim = []
		self.kmer_attention_co = []

		self.loss = 0

		lr = tf.compat.v1.train.natural_exp_decay(options['learning_rate'], options['epochs'], 5, 0.001, staircase = False)
		self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = lr,beta1=0.5,beta2=0.5,epsilon=1e-03)
		
		self.opt_op = None
		self.build()

	def _build(self):
		#  The first layer to learn the k-mer embedding from coexisting graph
		self.gatlayers_coo=GATLayer1(
			in_features = self.input_dim,
			out_features = self.options['hidden1'],
			dropout =self.placeholders['dropout'],
			alpha=0.2,
			stdv=self.options['stdv']
		)
		#  The first layer to learn the k-mer embedding from similarity graph
		self.gatlayers_sim=GATLayer1(
			in_features = self.input_dim,
			out_features = self.options['hidden1'],
			dropout = self.placeholders['dropout'],
			alpha=0.2,
			stdv=self.options['stdv']
		)
		#  The second layer to learn the sequence embedding of inclusive graph		
		self.gatlayers_inclu=GATLayer2(
			in_features = self.input_dim,
			out_features = self.options['hidden1'],
			dropout = self.placeholders['dropout'],
			alpha=0.2,
			)
		self.gatlayers3=GATLayer3(
			in_features = self.options['hidden1'],
			out_features = 1,
			stdv=self.options['stdv']
			)

	def build(self):
		self._build()

		Ec,co_att = self.gatlayers_coo(self.Ect,self.placeholders['Wcoo'])
		Es,sim_att = self.gatlayers_sim(self.Est,self.placeholders['Wsim'])
		seq_hidden,attention_co,attention_sim = self.gatlayers_inclu(self.Winclu,Ec,Es)
		output=self.gatlayers3(seq_hidden)
		self.seq_hidden.append(seq_hidden)
		self.kmer_hidden_sim.append(Es)
		self.kmer_hidden_co.append(Ec)
		self.attention_sim.append(attention_sim)
		self.attention_co.append(attention_co)
		self.kmer_attention_sim.append(sim_att)
		self.kmer_attention_co.append(co_att)

		self.outputs = tf.squeeze(input=output,axis=[0])
		self._loss()
		self.opt_op = self.optimizer.minimize(self.loss)
	def _loss(self):
		self.loss = masked_sigmoid_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
		self.preds = self.outputs
		self.labels = self.placeholders['labels']
