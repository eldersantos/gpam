import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

class PCA:
	
	@staticmethod
	def calc(inputs, variance = 0.95):
		
		asd = ((inputs - np.mean(inputs.T,axis = 1))/np.std(inputs.T, axis = 1)).T
		
		eigen_values, eigen_vectors = np.linalg.eig(np.cov(asd))

		scores = np.dot(eigen_vectors.T,asd)
		
		percentual = eigen_values[:] / np.sum(eigen_values)
		
		soma = 0

		for i in xrange(0,eigen_values.shape[0]):
			if(soma < variance):
				soma += 1
			else:
				break
				
		return eigen_values, eigen_vectors , scores[:,:soma]

class Neuron:

	def __init__(self, connections, side_connections):
		
		self.weight = np.random.random((connections))
		self.side_weight = np.random.random((side_connections))		
		self.output = 0

class Ann_PCA (Neuron):

	def __init__(self, in_inputs, out_neuron):

		self.max_normalization = 0
		self.min_normalization = 0
		self.learning_rate = random.random()
		self.mu = random.random()
		self.epochs = 300
		self.stop_criteria = 1
		self.erro = []
		
		
		self.layer = np.empty((out_neuron), dtype = object)

		for i in xrange(0, out_neuron):
			self.layer[i] = Neuron(in_inputs,i)

	def set_learning_rate(self, value):
		self.learning_rate = value

	def get_pesos(self):
		return self.pesos

	def set_mu_rate(self,value):
		self.mu  = value
		
	def set_epochs(self, value):
		self.epochs = value
		
	def training_set(self,inputs):

		self.max_normalization = np.zeros((inputs.shape[1]))
		self.min_normalization = np.zeros((inputs.shape[1]))

		for i in xrange(0, self.max_normalization.shape[0]):

			self.max_normalization[i] = np.max((inputs[:,i]),axis = 0)
			self.min_normalization[i] = np.min((inputs[:,i]),axis = 0)

			if(self.min_normalization[i] > 0):
				self.min_normalization[i] = 0
		
		self.samples = ((inputs - np.mean(inputs.T,axis = 1))/np.std(inputs.T, axis = 1))	

	def get_eigenvector(self,n):
		return self.layer[n].weight

	def normalize(self, inputs):
		
		for i  in range(0, inputs.shape[1]):
			inputs[:,i] -= self.min_normalization[i]
			inputs[:,i]  /= (self.max_normalization[i]-self.min_normalization[i])
		return inputs

	def padrao(self):

		self.layer[1].side_weight[0] = 0.6;
		self.layer[2].side_weight[0] = 0.5;
		self.layer[2].side_weight[1] = 0.4;
		self.layer[3].side_weight[0] = 0.3;
		self.layer[3].side_weight[1] = 0.2;
		self.layer[3].side_weight[2] = 0.1;

	def forward(self, _samples, prediction):

		output = np.zeros((_samples.shape[0] , self.layer.shape[0]))
		x_linha = np.zeros((_samples.shape[1]))

		for ii in xrange(0, _samples.shape[0]):

			for i in xrange(0, self.layer.shape[0]):

				self.layer[i].output = np.sum(self.layer[i].weight[:] * _samples[ii,:]) 
				output[ii,i] = self.layer[i].output 

			if(prediction == 0):	
				for i in xrange(0, x_linha.shape[0]):
					qwe = 0					
					for j in xrange(0, self.layer.shape[0]):
						qwe += self.layer[j].weight[i] * self.layer[j].output 
					x_linha[i] += _samples[ii,i] - qwe
				#qwe = 0
				for i in xrange(0, self.layer.shape[0]):
					#a = np.copy(self.layer[i].weight[:])	
					for j in xrange(0,self.layer[i].weight.shape[0]):
						self.layer[i].weight[j] += self.learning_rate * ((self.layer[i].output * x_linha[j]) - (pow(self.layer[i].output,2) * self.layer[i].weight[j]))
					#qwe += np.sqrt(np.sum(np.power(a[:] - self.layer[i].weight[:],2)))					
			
			   		
		'''
		for ii in xrange(0, _samples.shape[0]):

			if((self.stop_criteria <= 0) and (prediction == 0)): 
				break
		
			for i in xrange(0, self.layer.shape[0]):

				soma = 0
				for j in xrange(0, i):
					soma += self.layer[j].output * self.layer[i].side_weight[j]
					print "soma",i, j, self.layer[j].output, self.layer[i].side_weight[j]
				
				self.layer[i].output = (np.sum(self.layer[i].weight[:] * _samples[ii,:]) + soma)

				print "saida", i,self.layer[i].output

				output[ii,i] = self.layer[i].output

			if(prediction == 0):
				
				for i in xrange(0, self.layer.shape[0]):

					for j in xrange(0, self.layer[i].weight.shape[0]):

						self.layer[i].weight[j] += self.learning_rate * _samples[ii,j] * self.layer[i].output

				self.normalize_weigths()
				zxc = 0
				for i in xrange(0, self.layer.shape[0]):
				
					for j in xrange(0, self.layer[i].side_weight.shape[0]):
						
						#print "atualizacao  ",i,j,self.layer[i].side_weight[j], self.layer[j].output
						
						self.layer[i].side_weight[j] += self.mu *(-1) * self.layer[i].output * self.layer[j].output
						zxc += abs(self.layer[i].side_weight[j])

				self.stop_criteria = zxc

				self.pesos.append(zxc)
		'''

			
		if(prediction == 1):
			return output

	def plot_convergence_curve(self):

		plt.xlabel('Epoch')
		plt.ylabel('Squared Error')
		plt.title('Squared Error Curve')
		p2 = plt.plot(np.asarray(self.erro))
			
		#plt.legend([p2[0]],['weights'])

		plt.show()

	def normalize_weigths(self):

		for i in xrange(0, self.layer.shape[0]):

			modulo = np.linalg.norm(self.layer[i].weight)
			
			for j in xrange(0, self.layer[i].weight.shape[0]):
				self.layer[i].weight[j] /= modulo
						
	def training_PCA(self):

		self.normalize_weigths()
		for i in xrange(0, self.epochs):
			self.forward(self.samples, 0)
			W = np.zeros((self.layer.shape[0]*self.layer[0].weight.shape[0]))
			W.resize(self.layer.shape[0],self.layer[0].weight.shape[0])
			for m in xrange(self.layer.shape[0]):
				eigv = self.get_eigenvector(m)
				for n in xrange(eigv.shape[0]):
					W[m][n] = eigv[n]
			W = W.T
			A = self.samples.T
			self.erro.append(np.mean(np.sum(pow(A-np.dot(W,np.dot(W.T,A)),2)))/self.samples.shape[0])
			if(self.erro[-1]<=self.stop_criteria):
				break

		self.normalize_weigths()
		print "ann eigen vector"
		for i in xrange(0, self.layer.shape[0]):
			print self.get_eigenvector(i)
				
			
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

	def predict(self, inputs):
		
		inputs = ((inputs - np.mean(inputs.T,axis = 1))/np.std(inputs.T, axis = 1))
		
		return self.forward(inputs, 1)
		

#A = np.random.random((3,200))
#A = np.array([ [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9], [2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1] ])
 
#carregar arquivo
A = np.loadtxt("./datasets/wine.data")

#instanciar classe
ann = Ann_PCA(A.shape[1],A.shape[1])

#passando pra rede o conjunto de treinamento
ann.training_set(A)

ann.set_learning_rate(0.0001)

#ann.set_mu_rate(0.001)
ann.stop_criteria = ann.learning_rate

ann.set_epochs(100)
#mando treinar rede
ann.training_PCA()

print "Orthogonal?: ", np.dot(ann.get_eigenvector(0),ann.get_eigenvector(1))

#retorno dos dados transformados
truco = ann.predict(A)

media_ann =  np.mean(truco,axis = 0)

#print "media ",media_ann

media_ann = np.abs(media_ann)

#print "ann \n", (media_ann[:] / np.sum((media_ann))), np.sum((media_ann[:] / np.sum(media_ann))) 

print "ann \n", media_ann[:]

a_value, a_vet, score =  PCA.calc(A)

#print "pca eigen vector", a_vet

#print "pca \n", a_value[:] / np.sum(a_value) 
print "pca \n", a_value[:]

print '%', a_value[0] / 5
print '%', a_value[1] / 5
print '%', a_value[2] / 5
print '%', a_value[3] / 5
print '%', a_value[4] / 5

#print "diferenca ", (a_value[:] / np.sum(a_value)) - abs(media_ann[:] / np.sum(media_ann)) 

ann.plot_convergence_curve()

'''

print np.linalg.norm(ann.get_eigenvector(0))

rotated_space = A*ann.get_eigenvector(0).T

attr = [0,1]
plt.xlabel('Attr '+str(attr[0]))
plt.ylabel('Attr '+str(attr[1]))
plt.title('Rotated Space')
p2 = plt.plot(rotated_space[:,attr[0]],'*r')

#plt.legend([p2[0]],['weights'])

plt.show()

'''

'''
coeff, score, eigen = PCA.calc(A.T)

figure()
subplot(121)
# every eigenvector describe the direction
# of a principal component.
m = mean(A,axis=1)
plot([0, -coeff[0,0]*2]+m[0], [0, -coeff[0,1]*2]+m[1],'--k')
plot([0, coeff[1,0]*2]+m[0], [0, coeff[1,1]*2]+m[1],'--k')
plot(A[0,:],A[1,:],'ob') # the data
axis('equal')
subplot(122)
# new data
plot(score[1,:],'*g')
axis('equal')

print coeff.shape #auto vetores
print coeff[:]
print eigen.shape #auto valores
print eigen[:], np.sum(eigen)
print score.shape # nova representacao dos dados
print score[:]
show()

'''
