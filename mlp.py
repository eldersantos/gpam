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
#import libxml
import math
import matplotlib.pyplot as plt
import pickle
from cross_validation import SCV

'''


'''
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

'''


'''

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
	'''
	MLP is the main class of the this library, responsible to train the network,
	get the error and update the weigths.
	'''	

	def __init__(self, *args):
		'''
		:param args: array of integer values
			args[0] = number of inputs
			args[1] = number of output neurons
			args[2] = vector with the number of hidden neurons
		'''
		
		if(len(args) == 3):
			
			inputs = args[0]
			outputs = args[1]
			hidden = args[2]
		
			self.layer = np.empty((hidden.size+2), dtype = object)
			
			self.layer[0] = Layers(inputs,0)
			self.layer[1] = Layers(hidden[0],inputs)
		 
			if hidden.size > 1:
				for i in range(2,self.layer.size-1):		
					self.layer[i] = Layers(hidden[i-1],hidden[i-2])

			self.layer[self.layer.size-1] = Layers(outputs,hidden[hidden.size-1])

			self.erro = 0.01
			self.quad_erro_train = 0.0
			self.quad_erro_validation = 0.0
			self.on_validation = False
			self.learningRate =  random.random()
			self.momentum = random.random()
			self.learningDescent =  0.2
			self.epochs = 200
			self.plotar = 0
			self.max_normalization = 0
		else:
			print ('Invalid Arguments ')
			return 0

	def __getitem__(self, MLP):
		if isinstance(self.layer,slice):
			return self.__class__(self[z]
				for x in xrange(*self.layer.indices.len(self))
					for y in xrange(*self.layer.neuron.indices.len(self))
					for z in xrange(*self.layer.neuron.weight.indices.len(self)))	



	def set_learningRate(self, value):
		self.learningRate = value

	def set_momentum(self, value):
		self.momentum = value

	def set_epochs(self, value):
		self.epochs = value

	def set_learningDescent(self, value):
		self.learningDescent = value

	def set_erro(self, value):
		self.erro = value

	def sigmoidal(self, vj):
		return 1 / (1 + np.exp(-vj))

	def devSigmoidal(self, y):
		return y * (1.0 - y)

	def get_validationError (self):
		return self.quad_erro_validation.min()

	def get_trainingError (self):
		return self.quad_erro_train.min()

	
	def training_set(self, inputs, outputs):
		'''
			configure the samples and output set to train the MLP
		'''	
		self.samples = inputs

		try: 
			a,b = outputs.shape
			self.out = outputs
		except ValueError:
			self.out= np.zeros((outputs.shape[0], 1))
			self.out[:,0] = outputs[:]
	
		self.max_normalization = np.zeros((inputs.shape[1]))

		for i in xrange(0, self.max_normalization.shape[0]):
			self.max_normalization[i] = np.max((inputs[:,i]), axis = 0)
	
	
	def validation_set(self, inputs ,outputs):
		'''
		configure the validation and output set to view mlp generalization capacity
		'''
		try: 
			a,b = outputs.shape
			self.validation_out = outputs
		except ValueError:
			self.validation_out = np.zeros((outputs.shape[0], 1))
			self.validation_out[:,0] = outputs[:]

		self.validation = inputs
		self.on_validation = True
	
		
	def teste(self):
		'''
		this function is only test to verify if the mlp still works fine after update actions	
		this test is available in the book "Data Mining Concepts and Techniques" pages 405, 406.
		'''	
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

	 
	def forward(self, set_model, _samples, _output = None):
		'''
		variable: set_model [ 0 == training; 1 == validation_set (without backward); 2 == prediction mode] 
		forward function receive the set_model, samples and corresponding output, and carry out the first step of mlp
		'''
		predict  = np.zeros((int(_samples.shape[0]),self.layer[self.layer.size-1].n_neurons))	

		try:
			_erro = np.zeros((_output.shape[0],_output.shape[1]))

		except ValueError:
			_erro = np.zeros((_output.size,1))

		for ii in range(0,_samples.shape[0]):
			self.layer[0].inputs[:] = _samples[ii:ii+1]
	
			for i in range(0,self.layer[1].neuron.size):
				self.wx = self.layer[1].neuron[i].weight[:] * _samples[ii:ii+1]	
				self.layer[1].inputs[i] = self.sigmoidal(np.sum(self.wx[:]) + self.layer[1].neuron[i].currentBias)

			for i in range(2,self.layer.size):
				for j in range(0,self.layer[i].neuron.size):
					self.wx = self.layer[i].neuron[j].weight[:] * self.layer[i-1].inputs[:] 
					self.layer[i].inputs[j] = self.sigmoidal(np.sum(self.wx[:]) + self.layer[i].neuron[j].currentBias)
					
			for i in range(0, predict.shape[1]):
				predict[ii,i] = self.layer[self.layer.size-1].inputs[i]
				_erro[ii,i] = _output[ii,i] - self.layer[self.layer.size-1].inputs[i]

			if(set_model == 0):
				self.backward(_erro[ii])

		if(set_model == 0 or set_model == 1):
			return _erro

		if(set_model == 2):
			return predict
		
		
	def backward(self, erro):
		'''
		backward function is call inside forward step the according set_model.
		obs: if is situation training the weights are update 
		'''	
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
			

	def square_error(self, erro):
		'''
		Calculate the square mean error (MSE)
		'''		
		quadratic = 0
		for i in xrange(0, erro.shape[0]):
			for j in xrange(0, erro.shape[1]):
				quadratic += math.pow(erro[i,j],2)
		return quadratic / erro.size
	

	def normalize(self, inputs):
		'''
		Normalize the data set
		'''	
		for i  in range(0, inputs.shape[1]):
			inputs[:,i] /= self.max_normalization[i]

		return inputs


	def denormalization(self, inputs):		
		'''
			Denormalize the data set
		'''	
		for i  in range(0, inputs.shape[1]):
			inputs[:,i] *= self.max_normalization[i]

		return inputs

	'''
		Change the order of samples in data set
	'''
	def shuffle(self, _samples, _output):
		try:
			a1,b1 = _output.shape
		except ValueError:
			asd = _output
			_output = np.zeros((asd.shape[0],1))
			_output[:,0] = asd[:]
			a1,b1 = _output.shape
	
		a2,b2 = _samples.shape			

		dados = np.concatenate((_samples,_output), axis = 1)
		dados = np.random.permutation(dados)
					
		_samples = dados[:,0:b2]
		_output = dados[:,b2:b2 + b1]

		return self.normalize(_samples), self.normalize(_output)
	
	
		
	def train_mlp(self, inputs, outputs):
		'''
		Getting started mlp train process 
		'''
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
		self.quad_erro_train = np.zeros((self.epochs), float)
		self.quad_erro_validation = np.zeros((self.epochs), float)
		t_in, t_out = self.shuffle(self.samples, self.out)
				
		if(self.on_validation):
			v_in, v_out = self.shuffle(self.validation, self.validation_out)

		for i in xrange(0, self.epochs):
			self.quad_erro_train[i] = self.square_error(self.forward(0, t_in, t_out))
			if(self.on_validation):
				self.quad_erro_validation[i] = self.square_error(self.forward(1, v_in, v_out))
				if (self.quad_erro_validation[i] <= self.erro):
					return
		
					
	def predict(self, inputs):
		return self.denormalization(self.forward(inputs, 2))

	def plot_learning_curve(self):
		
		plt.xlabel('Epochs')
		plt.ylabel('Quadratic Error')
		plt.title('Quadratic Error Curve')

		y = np.arange(0,self.epochs)		
		
		p2 = plt.plot(y,self.quad_erro_train)
			
		if(self.on_validation):
			p1 = plt.plot(y,self.quad_erro_validation)
			plt.legend([p1[0], p2[0]], ['Validation','Training'])	
		else:
			plt.legend([p2[0]],['Training'])

			plt.plot(y,self.quad_erro_train)	
		plt.show()

	def save_mlp(self, mlp, path_and_namefile):
		
		path_and_namefile += ".pk1"

		with open(path_and_namefile, 'wb') as save:
			pickle.dump(mlp, save, pickle.HIGHEST_PROTOCOL)
		
	@staticmethod
	def load_mlp( path_and_namefile):
		try:
			with open(path_and_namefile, 'rb') as input:
				mlp = pickle.load(input)

			return mlp
		except:
			print ("Open file error, try again")


