"""
Jeferson de Souza 03/09/2014

"""
import numpy as np
import random
import cv2

class neuron:
	
	def __init__(self, conections):
		
		self.weights  = np.random.random((conections))	
		self.value = 0
		print("init", self.weights)

class som (neuron):

	def __init__(self, attributes, nNeurons):
		
		self.neurons = np.empty((nNeurons), dtype = object)			
		self.learningRate = random.random()
		self.sigma =0.0000001
		self.stopCriteria = 0.00000001

		for i in range(0, nNeurons):

			self.neurons[i] = neuron(attributes)

	def set_learningRate(value):
		self.learningRate = value

	def normalize(self, inputs):

		inputs = inputs.astype('float')
		a,b = inputs.shape
		for i in range(0, b):
			inputs[:,i] /= np.max(inputs[:,i]) 

		return inputs

	def train_som(self,inputs, epochs):
		
		self.initLearning = self.learningRate
		self.initSigma = self.sigma

		inputs = self.normalize(inputs)
		
		for i in range(0, epochs):
			
			print(i)
			self.run(inputs)

			self.learningRate = self.initLearning * np.exp(-1*(i/epochs))
			self.sigma = self.initSigma * np.exp(-1*(i/epochs))

		for j in range(0, self.neurons.size):
					
			print(j,self.neurons[j].weights[:])			 	
				
	def run(self, inputs):

		a,b = inputs.shape
		aa = 0
		bb = 0
		for ii in range(0, a):
			
			index = self.winner(inputs[ii:ii+1])	
			
			for i in range(0, self.neurons.size):	

				if(np.sqrt(pow(self.neurons[i].value - self.neurons[index].value,2))<= self.sigma):
					self.update(i,inputs[ii:ii+1])
								 	
		
	def winner(self, inputs):

		win = 0
		value = 1000
		result = np.zeros((self.neurons[0].weights.size))

		for i in range(0, self.neurons.size):	

				result[:] = self.neurons[i].weights[:] - inputs[:]

				self.neurons[i].value = np.sqrt(np.sum(pow(result[:],2)))
				
				if(self.neurons[i].value < value):
					value = self.neurons[i].value
					win = i
		return win
			
	def update(self, index, inputs):
			self.neurons[index].weights[:] += self.gauss(index,inputs)*self.learningRate*np.sum(inputs[:] - self.neurons[index].weights[:])

	def gauss(self, index, inputs):
			
			return np.exp(-1*(np.sum(pow(inputs[:] - self.neurons[index].weights[:],2)) / (2*pow(self.sigma,2)))) 


	def predict(self, inputs):

		a, b = inputs.shape

		inputs = self.normalize(inputs)

		truco = np.zeros(a)

		for i in range(0, a):
			
			truco[i] = self.winner(inputs[i:i+1])
		
		return 	truco

	def paint_img(self, img,result):

		colors = np.zeros((self.neurons.size, 3))
		
		for i in range(0,self.neurons.size):
			for j in range(0,3):
				colors[i,j] = random.randint(0,255)
		
		a,b,c = img.shape
		k = 0
		for i in range(0, a):
		
			for j in range(0, b):
				
				img[i,j] = colors[result[k],:]
				k+=1
						
					
#***********************************************************************************************************
im1 = cv2.imread("/home/jeferson/Desktop/GramaAsfalto2.png")


im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)

im2 = cv2.imread("/home/jeferson/Desktop/VideoEditado_01.png")

vtr = np.copy(im2)

im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

im1 = im1.reshape(im1.size,1)

im2 = im2.reshape(im2.size,1)

a,b,c = vtr.shape

#im1 = im1.reshape(a,b)
#im1 = np.random.random((50,1))
#im2 = np.random.random((50,1))

a  = som(1,3)

a.train_som(im1,2)

result = a.predict(im2)

print("Histogram",np.histogram(result))

a.paint_img(vtr,result)

cv2.namedWindow("img")
cv2.imshow("img",vtr)
cv2.waitKey(0)

