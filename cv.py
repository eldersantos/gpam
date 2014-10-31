"""
Anonymous // a long time ago

"""

class CrossValidationParameters:

	def __init__(self, numFolds, learnRate, momentum, maxEpochs, minMse, learnDecrease):
		self.numFolds = numFolds
		self.learnRate = learnRate
		self.momentum = momentum
		self.maxEpochs = maxEpochs
		self.minMse = minMse
		self.learnDecrease = learnDecrease



class CrossValidation:

	def __init__(self, crossValidationParameters):
		self.crossValidationParameters = crossValidationParameters



