"""
	Produced by Paulo Burke 06/11

	OBS: O loko fez em um poco mais de 2hs ahauhauahuauha
	OBS2: Mais 2h de debug ASUHsahuasuhuaHS
"""

import numpy as np

class SCV:

	def __init__(self, parameters, _kfold = 10):
		self.kfold = _kfold

		#Assuming the classes are the last column on the dataset(parameters)
		classes = []
		for i in parameters:
			classes.append(i[-1]);
			np.loadtxt(classes)

		self.nameClasses = np.unique(classes)
		self.nameClasses = self.nameClasses.tolist()
		self.dictClasses = {}
		for i in range(len(self.nameClasses)):
			self.dictClasses[self.nameClasses[i]] = i
		self.invertDictClasses = dict((v,k) for k,v in self.dictClasses.iteritems())
		self.separatedParameters = [[] for _ in range(len(self.nameClasses))]
		for i in range(parameters.shape[0]):
			self.separatedParameters[self.dictClasses[classes[i]]].append(parameters[i].tolist())
		self.foldsParameters = [[] for _ in range(self.kfold)]
		self.foldsClasses = [[] for _ in range(self.kfold)]
		for i in range(self.kfold):
			for j in range(len(self.separatedParameters)):
				indice = int(len(self.separatedParameters[j]) / self.kfold)
				self.foldsParameters[i] += self.separatedParameters[j][indice * i:indice * (i + 1)]
				self.foldsClasses[i] += [self.invertDictClasses[j]] * indice


	def select_fold_combination(self, k = 0):
		trainingSet = []
		trainingOutput = []
		validationSet = self.foldsParameters[k]
		validationOutput = self.foldsClasses[k]
		train = out = 0
		for i in range(len(self.foldsParameters)):
			if i != k:
				#trainingSet.append(self.foldsParameters[i])
				#trainingOutput.append(self.foldsClasses[i])
				try:
					train = np.concatenate((train,np.array(self.foldsParameters[i])), axis = 0)
					out = np.concatenate((out,np.array(self.foldsClasses[i])), axis = 0)
				except ValueError:
					train = np.array(self.foldsParameters[i])
					out = np.array(self.foldsClasses[i])

		return train, out, np.array(validationSet), np.array(validationOutput)
		
