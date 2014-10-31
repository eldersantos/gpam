"""
kMeans by Jeferson de Souza 07/08/2014
"""

import cv2 as opencv
import numpy as np
import math as math
import matplotlib.pyplot as plt
from random import randint

#*********************************************************************************
class Atributos:

	def media (self, img, x, y,neighbor):

		k = 0
		width,height = img.shape
  		
     		truco = (((x+neighbor+1) - (x-neighbor)) * ((y+neighbor+1) - (y-neighbor)))	
   
     		self.a = np.zeros(truco, dtype = int)	

     		for i in range(x-neighbor, x+neighbor+1):
			
       			for j in range(y-neighbor, y+neighbor+1):
         
				if i >= width or j >= height:
			        	self.a[k] = 0           
     	 			else:
	   				self.a[k] = img[i,j]  	   
         			k+=1

     		if np.sum(self.a)-img[x,y] < 0:
	        	return 0		
     		return (np.sum(self.a)-img[x,y])/(truco-1) 

 	
		
#*********************************************************************************
class Centroide:

	def __init__(self):
		self.media = 0
		self.gradientx = 0
		self.gradientY = 0
		self.cluster = [];
        	self.position = [];
		
#*********************************************************************************
class kMeans(Centroide):
	
	def __init__(self, nImg = 1):
		self.imgs = nImg
		self.atributos =0
		self.centroide =0
                self.atr = Atributos()
				      		      
	def createCentroide(self, n,neighbor, img):
	        
		width,height = img.shape
		self.centroide = []
				
		for i in range(0,n):
			self.centroide.append(Centroide())	
			self.centroide[i].media = self.atr.media(img,randint(neighbor, width),randint(neighbor, height),neighbor)
		   	print "Centroide", self.centroide[i].media

	def extractAtributes(self, img, neighbor,index): 
		
		width, height = img.shape	
		self.atributos = np.zeros((self.imgs,width,height), dtype= float)
	        
		for i in range(neighbor,width-neighbor):
			for j in range(neighbor, height-neighbor):
		        	self.atributos[index,i, j] = self.atr.media(img,i,j,neighbor)

	def clustering(self):
		
		minimo = np.array(([-1,-1]),dtype = int)
		img, width, height = self.atributos.shape
		
		for z in range(0, img):	
			for i in range(0,width):
				for j in range(0, height):
		        		for k in range(0,len(self.centroide)):
			    			if minimo[0] == -1:
			       				minimo = [math.sqrt(math.pow(self.centroide[k].media - self.atributos[z,i,j],2)),k]	
			    
			    			else:
			       				a = math.sqrt(math.pow(self.centroide[k].media - self.atributos[z,i,j],2))	
			        if(a < minimo[0]):
			       		minimo = [a,k]
				  	 
		        self.centroide[minimo[1]].cluster.append(self.atributos[z,i,j])	
		        self.centroide[minimo[1]].position.append(np.array([i,j]))
			 
		      	minimo = [-1,-1] 
		        
  	def updateCentroide(self):

		soma = 0
		
		for i in range(0, len(self.centroide)):
			for j in range(0, len(self.centroide[i].cluster)):
				soma += self.centroide[i].cluster[j]
			
		   	if len(self.centroide[i].cluster) > 0:		
		        	self.centroide[i].media = soma/len(self.centroide[i].cluster)
		   	else:
		      		self.centroide[i].media = 0;
                 
		   	soma = 0

		   	print "Centroide Novo",i,self.centroide[i].media	
			
		del self.centroide[i].cluster[0:len(self.centroide[i].cluster)]	
		del self.centroide[i].position[0:len(self.centroide[i].position)]

	def predict(self, img, imgResult, neighbor):
		
		cor = []
		minimo = np.array(([-1,-1]),dtype = int)
		a = 0
		
		for k in range(0, len(self.centroide)):
			cor.append(np.array(([randint(0,255),randint(0,255),randint(0,255)]),dtype =int))		
		  	
                width, height = img.shape	

		for i in range(0, width):
			for j in range(0,height):
				for k in range(0, len(self.centroide)):
			   		if minimo[0] == -1:
		              			minimo = [math.sqrt(math.pow(self.centroide[k].media - self.atr.media(img,i,j,neighbor),2)),k]
			   		else:
			      			a = math.sqrt(math.pow(self.centroide[k].media - self.atr.media(img,i,j,neighbor),2)) 		       
			      		if(a < minimo[0]):
				 		minimo = [a,k]
			
				imgResult[i,j] = cor[minimo[1]]
				minimo = [-1,-1]

	def ploting(self):

		mt = np.array((self.centroide[0].cluster), dtype = int)

		plt.plot(mt,mt,mt,'ro')
	
		plt.axis([0,255,0,255])
		plt.show()		
    	   	 			
#*********************************************************************************
img = opencv.imread('/home/jeferson/Desktop/GramaAsfalto2.png')
result1 = opencv.imread('/home/jeferson/Desktop/VideoEditado_.png')

imgGray = opencv.cvtColor(img,opencv.COLOR_BGR2GRAY)
result2 = opencv.cvtColor(result1,opencv.COLOR_BGR2GRAY)

k = kMeans(1)
k.extractAtributes(imgGray, 2,0)
k.createCentroide(2,2,imgGray)

for i in range(0,3):
   k.clustering()
   k.updateCentroide()

k.predict(imgGray,img,2)

opencv.namedWindow("Treino")
opencv.imshow("Treino",imgGray)

opencv.namedWindow("Resultado")
opencv.imshow("Resultado",img)

opencv.waitKey(0)
