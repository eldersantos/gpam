import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import numpy as np

np.set_printoptions(threshold=np.nan)

dataset = np.loadtxt("./datasets/breast-cancer.data")
dataset_output = dataset[:,-1]
dataset_input = dataset[:,0:9]
names = np.array(['benign', 'malignant', 'none'])

b = {'data' : dataset_input, 'target' : dataset_output, 'target_names' : names}

X = b['data']
y = b['target']
target_names = b['target_names']

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show

'''
plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of IRIS dataset')
'''


