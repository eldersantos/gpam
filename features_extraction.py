'''

	By jeferson de Souza 05/09

'''
import numpy as np
import math

class Basic:

	def __init__(self, size_neightborhood):
		
		self.size = size_neightborhood*2 +1
		self.neightbor = size_neightborhood
		self.mx = self.mask_x()
		self.my = self.mask_y()
	
	def mask_x(self):

			m = np.zeros((self.size,self.size))

			k  = (int(self.size)/2)

			for i in xrange(0, self.size):
		
				for j in xrange(0, self.size):

					if(i == 0):
						m[i,j] = 1
						if(j == k):
							m[i,j] = 2

					elif(i == (self.size-1)):	
						m[i,j] = 1
						if(j == k):
							m[i,j] = 2

			return m
			
	def mask_y(self):

			m = np.zeros((self.size,self.size))

			k  = (int(self.size)/2)

			for i in xrange(0, self.size):
		
				for j in xrange(0, self.size):

					if(j == 0):
						if(i == k):
							m[i,j] = 2
						else:	
							m[i,j] = 1

					elif(j == self.size-1):
						if(i == k):
							m[i,j] = 2
						else:	
							m[i,j] = 1
			return m


	def media (self, mat):
   		
     		return (np.sum(mat[:])-mat[self.size/2,self.size/2])/(pow(self.size,2)-1)

	def gradient(self, mat):
		
		gy = mat*self.my
		gx = mat*self.mx
		
		return math.sqrt(pow(np.sum(gy[0:,0]) - np.sum(gy[0:,self.size-1]),2) + pow(np.sum(gx[0:,0]) - np.sum(gx[0:,self.size-1]),2))
			
		
	def extract(self, matrix):

		
		a, b = matrix.shape

		result = np.zeros(((a*b)-(a+b), 2))
		
		k = 0
		size = (a*b)-(a+b)
		for i in xrange(self.neightbor, a - self.neightbor ):
	
			for j in xrange(self.neightbor, b - self.neightbor):

				mat = matrix[i-self.neightbor: i + self.size/2 +1, j -self.neightbor:j+self.size/2 +1]		
			
				if(k < size):
					result[k,0] = self.media(mat)
					result[k,1] = self.gradient(mat)
					k+=1
				else:
					return result

		return result

#**********************************************************************************

				
