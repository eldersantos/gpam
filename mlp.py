"""

by jeferson de souza 16/08/2014

"""
"""

Slices Samples

a[start:end] # items start through end-1
a[start:]    # items start through the rest of the array
a[:end]      # items from the beginning through end-1
a[:]         # a copy of the whole array

a[-1]    # last item in the array
a[-2:]   # last two items in the array
a[:-2]   # everything except the last two items
a[:,1] #get the second column

slice(start, stop, increment)

"""
import numpy as np
import random
import libxml
import math
import matplotlib.pyplot as plt
import pickle

class Neurons:

	def __init__(self,conections):
		self.currentBias = random.random()
		self.weight = np.random.random((conections))
		self.previousBias = 0.0
		self.inputs = 0.0

	def __getitem__(self, Neuron):
		if isinstance(self.weight,slice):
			return self.__class__(self[x]
					     for x in xrange(*self.weight.indices.len(self)))	

class Layers(Neurons):

	def __init__(self,number_neurons,conections):

		self.neuron = np.empty((number_neurons),dtype = object)
		for i in range(0, int(number_neurons)):		
			self.neuron[i] = Neurons(conections)
		self.inputs = np.zeros((number_neurons))
		self.gradient = np.zeros((number_neurons))
		self.delta = np.zeros((number_neurons))
		self.n_neurons = number_neurons

	def __getitem__(self, Layer):
		if isinstance(self.neuron,slice):
			return self.__class__(self[x]
					     for x in xrange(*self.neuron.indices.len(self)))	
class MLP(Layers):

	def __init__(self, *args):
		
		if(len(args)==3):
			
			inputs = args[0] #numero de padroes de entrada
			outputs = args[1] # numero de neuronios na camada de saida
			hidden = args[2] # vetor que com dimensao do numero de camadas escondidas e os neuronios em cada uma
		
			self.layer = np.empty((hidden.size+2), dtype = object)
		
			self.layer[0] = Layers(inputs,0)
			self.layer[1] = Layers(hidden[0],inputs)
		 
			if hidden.size > 1:
				for i in range(2,self.layer.size-1):		
					self.layer[i] = Layers(hidden[i-1],hidden[i-2])

			self.layer[self.layer.size-1] = Layers(outputs,hidden[hidden.size-1])

			self.erro = 0.0
			self.quadratic_erro = 0.0
			self.plot_graph = 0
			self.learningRate = 0.9 # random.random()
			self.momentum = random.random()
			self.learningDescent = 1 # random.random()
			self.epochs = 100
			self.plotar = 0
			self.max_normalization = 0
		else:
			print ('Invalid Arguments ')

	def __getitem__(self, MLP):
		if isinstance(self.layer,slice):
			return self.__class__(self[z]
					     for x in xrange(*self.layer.indices.len(self))
					     	   for y in xrange(*self.layer.neuron.indices.len(self))
							 for z in xrange(*self.layer.neuron.weight.indices.len(self)))	

	def set_learningRate(self,value):
		self.learningRate = value

	def set_momentum(self,value):
		self.momentum = value

	def set_epochs(self,value):
		self.epochs = value

	def set_learningDescent(self,value):
		self.learningDescent = value

	def graph_on(self, value):
		self.plot_graph = value

	def sigmoidal(self, vj):
		return 1/(1+np.exp(-vj))

	def devSigmoidal(self,y):
		return y*(1.0-y)

	def training_set(self,inputs,outputs):	
			
		self.samples = inputs
		self.out = outputs
		self.a,self.b = self.out.shape
		self.erro = np.zeros((self.a,self.b))
		
		self.max_normalization = np.zeros((inputs.shape[1]))
		for i in xrange(0, self.max_normalization.shape[0]):
			self.max_normalization[i] = np.max((inputs[:,i]),axis = 0)
		
	def teste(self):

		self.layer[1].neuron[0].weight[0] = 0.2
		self.layer[1].neuron[0].weight[1] = 0.4
		self.layer[1].neuron[0].weight[2] = -0.5
		self.layer[1].neuron[0].currentBias = -0.4
		
		self.layer[1].neuron[1].weight[0] = -0.3
		self.layer[1].neuron[1].weight[1] = 0.1
		self.layer[1].neuron[1].weight[2] = 0.2	
		self.layer[1].neuron[1].currentBias = 0.2
		
		self.layer[2].neuron[0].weight[0] = -0.3
		self.layer[2].neuron[0].weight[1] = -0.2
		self.layer[2].neuron[0].currentBias = 0.1
	 
	def forward(self, set_predict):

		predict  = np.zeros((int(self.samples.shape[0]),self.layer[self.layer.size-1].n_neurons))	

		for ii in range(0,self.samples.shape[0]):
			
			self.layer[0].inputs[:] = self.samples[ii:ii+1]
	
			for i in range(0,self.layer[1].neuron.size):

				self.wx = self.layer[1].neuron[i].weight[:] * self.samples[ii:ii+1]	
				
				self.layer[1].inputs[i] = self.sigmoidal(np.sum(self.wx[:]) + self.layer[1].neuron[i].currentBias)

			for i in range(2,self.layer.size):

				for j in range(0,self.layer[i].neuron.size):

					self.wx = self.layer[i].neuron[j].weight[:] * self.layer[i-1].inputs[:] 
							
					self.layer[i].inputs[j] = self.sigmoidal(np.sum(self.wx[:]) + self.layer[i].neuron[j].currentBias)
					
			for i in range(0, predict.shape[1]):
 		
				predict[ii,i] = self.layer[self.layer.size-1].inputs[i]

				self.erro[ii,i] = self.out[ii,i] - self.layer[self.layer.size-1].inputs[i]

			if(set_predict == 0):
				self.backward(self.erro[ii])

		if(set_predict == 1):
			
			return predict
					
	def backward(self,erro):

			self.layer[self.layer.size-1].gradient[:] = erro[:]*self.devSigmoidal(self.layer[self.layer.size-1].inputs[:])	
						
			for i in range(self.layer.size-2,0,-1):
				
				self.sum = np.zeros((self.layer[i].neuron.size))
						
				for j in range(0, self.layer[i].neuron.size):
					
					for k in range(0, self.layer[i+1].neuron.size):

						self.sum[j] += self.layer[i+1].gradient[k]*self.layer[i+1].neuron[k].weight[j]
						
					self.layer[i].gradient[j] = self.devSigmoidal(self.layer[i].inputs[j])*self.sum[j]
								
			
			for i in range(self.layer.size-2,-1,-1):
			
				for k in range(0, self.layer[i].neuron.size):
											
					for j in range(0, self.layer[i+1].neuron.size):
					
						self.layer[i+1].neuron[j].weight[k] += (self.momentum*self.layer[i+1].delta[j]) + self.learningRate * self.layer[i+1].gradient[j] * self.layer[i].inputs[k]
						
						
			for i in range(self.layer.size-1,0,-1):
			
				for j in range(0, self.layer[i].neuron.size):
					
					self.layer[i].neuron[j].currentBias += self.learningRate * self.layer[i].gradient[j]
					self.layer[i].delta[j] = self.learningRate * self.layer[i].gradient[j]
					
	def square_error(self,index):

		self.a,self.b = self.erro.shape
		for i in range(0, self.a):
			for j in range(0, self.b):
				self.quadratic_erro[index] += math.pow(self.erro[i,j],2)
		self.quadratic_erro[index] /= (self.a*self.b)
		
	def normalize(self, inputs):
		
		for i  in range(0, inputs.shape[1]):
			inputs[:,i] /= self.max_normalization[i]

		return inputs

	def denormalization(self, inputs):
		
		for i  in range(0, inputs.shape[1]):
			inputs[:,i] *= self.max_normalization[i]

		return inputs

	def shuffle(self):

		self.a1,self.b1 = self.out.shape
		self.a2,self.b2 = self.samples.shape			

		self.dados = np.concatenate((self.samples,self.out),axis = 1)
		self.dados = np.random.permutation(self.dados)
					
		self.samples = self.dados[:,0:self.b2]
		
		self.out = self.dados[:,self.b2:self.b2+self.b1]
		
	def train_mlp(self, inputs, outputs):

		self.training_set(inputs, outputs)
	
		self.train()

	def keep_train_mlp(self,inputs,outputs):

		self.samples = self.denormalization(self.samples)

		self.samples = np.concatenate(self.samples,inputs)
	
		self.out = self.denormalization(self.out)

		self.out = np.concatenate(self.out,outputs)
		
		self.training_set(inputs, outputs)

		self.train()

	def train(self):

		self.quadratic_erro = np.zeros((self.epochs), float)

		self.shuffle()

		self.samples = self.normalize(self.samples)
	
		self.out = self.normalize(self.out)
		
		for ii in range(0, self.epochs):

			self.forward(0)

			self.learningRate *= self.learningDescent

			self.square_error(ii)
			
	def predict(self, inputs):

		self.samples = inputs
		return self.denormalization(self.forward(1))

	def plot_learning_curve(self):
		
		plt.xlabel('Epochs')
		plt.ylabel('Quadratic Error')
		plt.title('Quadratic Error Curve')
		plt.plot(self.quadratic_erro, 'r-')	
		plt.show()

	def save_mlp(self, mlp, path_and_namefile):
		
		path_and_namefile += ".pk1"

		with open(path_and_namefile, 'wb') as save:
			pickle.dump(mlp,save,pickle.HIGHEST_PROTOCOL)
		
	@staticmethod
	def load_mlp( path_and_namefile):

		#path_and_namefile += ".pk1"
		try:
			with open(path_and_namefile, 'rb') as input:
				mlp = pickle.load(input)

			return mlp
		except:
			print ("Open file error, try again")

		

#******************************************************************************************************

