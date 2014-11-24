import pylab as pl
from itertools import cycle
from sklearn.decomposition import PCA
import numpy as np
import sys

class pca_reduction:
  def __init__(self, dataset):

    #if (dataset == 1):
    self.load_dataset_breast()

    #if (dataset == 2):
    #load_dataset_glass(self)

    #if (dataset == 3):
    #self.load_dataset_wine(self)

    self.plot()

  def plot(self):
    pca = PCA(n_components=2, whiten=True).fit(self.X)
    X_pca = pca.transform(self.X)
    plot_2D(X_pca, self.y, self.names)

  def load_dataset_wine(self):
    dataset = np.loadtxt("./datasets/wine.data")
    self.X = dataset[:,0:12]
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

def plot_2D(data, target, target_names):
  colors = cycle('rgbcmykw')
  target_ids = range(len(target_names))
  pl.figure()
  for i, c, label in zip(target_ids, colors, target_names):
    pl.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label, marker='D')
  pl.legend()
  pl.show()

if __name__ == '__main__':
  argv = sys.argv
  pr = pca_reduction(argv[1])
  print 'X = %s' %pr.X
  print 'y = %s' %pr.y
  print 'names = %s' %pr.names