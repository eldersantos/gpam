import pylab as pl
from itertools import cycle
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

class pca_reduction:
  def __init__(self):

    np.set_printoptions(threshold=np.nan)
    #self.pca = PCA(n_components=2, whiten=True)
    #if (dataset == 1):
    #self.load_dataset_breast()

    #if (dataset == 2):
    #self.load_dataset_glass()

    #if (dataset == 3):
    #self.load_dataset_iris()
    '''
    np.set_printoptions(threshold=np.nan)
    dataset = np.loadtxt("./datasets/wine.data")
    self.X = dataset[:,0:12]
    self.y = dataset[:,-1]
    self.names = np.array(['class1', 'class2', 'class3'])
    res = self.pca_1(self.X, self.y, self.names)
    self.plot_wine(res[-1])
    '''

    dataset = np.loadtxt("./datasets/breast-cancer.data")
    dataset = dataset[dataset[:,-1].argsort()]
    print dataset
    self.X = dataset[:,0:9]
    self.y = dataset[:,-1]
    self.names = np.array(['class1', 'class2', 'class3'])
    res = self.pca_1(self.X, self.y, self.names)
    self.plot_breast(res[-1])

    #self.plot()

  def get_cov(self):
    return self.pca.get_covariance()

  def get_score(self):
    return self.pca.score_samples(self.X)

  def plot(self):
    self.pca = self.pca.fit(self.X)
    X_pca = self.pca.transform(self.X)
    plot_2D(X_pca, self.y, self.names)
    #plot_scores(self.get_score())

  def plot_iris(self, score):
    pl.figure()
    pl.scatter(score[0,0:50], score[1,0:50], marker='o', label='setosa', color = 'blue')
    pl.scatter(score[0,50:100], score[1,50:100], marker='o', label='versicolor', color = 'red')
    pl.scatter(score[0,100:], score[1,100:], marker='o', label = 'virginica', color = 'black')
    pl.legend()
    pl.show()

  def plot_wine(self, score):
    pl.figure()
    pl.scatter(score[0,0:59], score[1,0:59], marker='o', label='wine 1', color = 'blue')
    pl.scatter(score[0,59:130], score[1,59:130], marker='o', label='wine 2', color = 'red') 
    pl.scatter(score[0,130:], score[1,130:], marker='o', label = 'wine 3', color = 'black')
    pl.legend()
    pl.show()

  def plot_breast(self, score):
    pl.figure()
    pl.scatter(score[0,0:458], score[1,0:458], marker='o', label='benign', color = 'blue')
    pl.scatter(score[0,458:], score[1,458:], marker='o', label='malignant', color = 'red') 
    pl.legend()
    pl.show()

  def calc(inputs, variance = 0.95):
    asd = ((inputs - np.mean(inputs.T,axis = 1)) / np.std(inputs.T, axis = 1)).T
    eigen_values, eigen_vectors = np.linalg.eig(np.cov(asd))
    scores = np.dot(eigen_vectors.T, asd)
    percentual = eigen_values[:] / np.sum(eigen_values)  
    soma = 0

    for i in xrange(0,eigen_values.shape[0]):
      if(soma < variance):
        soma += 1
      else:
        break
        
    return eigen_values, eigen_vectors , scores[:,:soma]

  def pca_1(self, A, target, names):
    samples = ((A - np.mean(A.T, axis = 1)) / np.std(A.T, axis = 1)).T
    auto_value, auto_vet = np.linalg.eig(np.cov(samples))
    score = np.dot(auto_vet.T, samples)
    percentual = auto_value[:] / np.sum(auto_value)
    print percentual
    
    return auto_vet, auto_value, score

  def load_dataset_wine(self):
    dataset = np.loadtxt("./datasets/wine.data")
    self.X = dataset[:,0:12]
    self.y = dataset[:,-1]
    self.names = np.array(['class1', 'class2', 'class3'])

  def load_dataset_iris(self):
    dataset = np.loadtxt("./datasets/iris.data")
    self.X = dataset[:,0:4]
    self.y = dataset[:,-1]
    self.names = np.array(['class1', 'class2', 'class3'])

  def load_dataset_breast(self):
    dataset = np.loadtxt("./datasets/breast-cancer.data")
    self.X = dataset[:,0:8]
    self.y = dataset[:,-1]
    for i in xrange(0, len(self.y)):
      self.y[i] = self.y[i] / 2

    self.names = np.array(['benign', 'malignant'])

  def load_dataset_glass(self):
    dataset = np.loadtxt("./datasets/glass.data")
    self.X = dataset[:,1:10]
    self.y = dataset[:,-1]
    self.names = np.array(['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7'])

def plot_scores(data):
  pl.figure()
  pl.scatter(data)
  pl.legend()
  pl.show()

def plot_2D(data, target, target_names):
  colors = cycle('rgbcmykw')
  target_ids = range(len(target_names))
  pl.figure()
  for i, c, label in zip(target_ids, colors, target_names):
    pl.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)
  pl.legend()
  pl.show()

if __name__ == '__main__':
  argv = sys.argv
  pr = pca_reduction()
  #print 'X = %s' %pr.X
  #print 'y = %s' %pr.y
  #print 'names = %s' %pr.names