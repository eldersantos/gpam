import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.lda import LDA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

dataset = np.loadtxt("./datasets/wine.data")
dataset_output = dataset[:,-1]
dataset_input = dataset[:,0:12]
names = np.array(['class1', 'class2', 'class3'])

b = {'data' : dataset_input, 'target' : dataset_output, 'target_names' : names}

X = b['data']
y = b['target']
target_names = b['target_names']

#X = np.array([[-1, -1, 0], [-2, -1, -1], [-3, -2, -2], [1, 1, 1], [2, 1, 0], [3, 2, 1]])

pca = PCA(n_components=10)
X1 = -scale(pca.fit_transform(X))
#print(pca.explained_variance_ratio_) 
print 'variance2 '
print pca.get_covariance()

#print X
print 'X1'
print X1

#print X1[0:4,0]
#print X1[0:4,1]

Y = pca.inverse_transform(X1)

plt.scatter(X[:, 0], X[:, 1]) 
plt.show()

#plt.plot(X, y, 'o')
#plt.plot(x2, y + 0.5, 'o')
#plt.ylim([-0.5, 1])
#plt.show()
'''

plt.plot(X1[0:59,0],X1[0:59,1],'o', markersize=7, color='blue', alpha=0.5, label=target_names[0])


plt.plot(X1[60:130,0], X1[60:130,1],'^', markersize=7, color='red', alpha=0.5, label=target_names[1])

plt.plot(X1[131:,0], X1[131:,1],'x', markersize=7, color='red', alpha=0.5, label=target_names[2])

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.legend()
plt.title('PCA Transformed samples with class labels')

plt.show()
'''
