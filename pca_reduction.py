from sklearn.datasets import load_iris
import pylab as pl
from itertools import cycle
from sklearn.decomposition import PCA

class pca_reduction:
  def __init__(self):
    iris = load_iris()
    self.X = iris.data
    self.y = iris.target
    self.names = iris.target_names
    self.plot()

  def plot(self):
    pca = PCA(n_components=2, whiten=True).fit(self.X)
    X_pca = pca.transform(self.X)
    plot_2D(X_pca, self.y, self.names)

def plot_2D(data, target, target_names):
  colors = cycle('rgbcmykw')
  target_ids = range(len(target_names))
  pl.figure()
  for i, c, label in zip(target_ids, colors, target_names):
    pl.scatter(data[target == i, 0], data[target == i, 1],
          c=c, label=label)
  pl.legend()
  pl.show()

if __name__ == '__main__':
  pr = pca_reduction()
  print 'X = %s' %pr.X
  print 'y = %s' %pr.y
  print 'names = %s' %pr.names