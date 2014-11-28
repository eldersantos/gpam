from net_pca import Ann_PCA, PCA, Neuron
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#ann = Ann_PCA.load_pca("pca_wine.pk1")

database = np.loadtxt("./datasets/wine.data")
print database

'''
ann = Ann_PCA(database.shape[1],database.shape[1])

ann.training_set(database)

ann.set_learning_rate(0.0001)

ann.set_mu_rate(0.0001)

ann.training_pca()

output = ann.predict(database)

ann.plot_convergence_curve()

ann.save_pca(ann, "pca_iris")

'''

#output = ann.predict(database)

#ann.plot_convergence_curve()

a_value, a_vector, score = PCA.calc(database)

'''
print "ANN "
for i in xrange(0, ann.layer.shape[0]):
	print ann.layer[i].weight
'''

print "PCA ", a_vector.T

print "Score", score


print "AutoValores ", a_value
somaVect = np.sum(a_value)

print somaVect

'''
pl.figure()
pl.scatter(score[0:50,0], score[0:50,1], marker='o', label='setosa', color = 'blue')
pl.scatter(score[50:100,0], score[50:100,1], marker='o', label='versicolor', color = 'red')
pl.scatter(score[100:,0], score[100:,1], marker='o', label = 'virginica', color = 'black')
pl.legend()
pl.show()
'''

'''
pl.figure()
pl.scatter(score[0:59,0], score[0:59,1], marker='o', label='wine 1', color = 'blue')
pl.scatter(score[59:130,0], score[59:130,1], marker='o', label='wine 2', color = 'red') 
pl.scatter(score[130:,0], score[130:,1], marker='o', label = 'wine 3', color = 'black')
pl.legend()
pl.show()
'''

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(score[:,0], score[:,1], score[:,2], marker='o')

plt.show()



print (a_value[0] / somaVect) + (a_value[1] / somaVect) 

'''
print a_value[1] / a_value.shape[0]
print a_value[2] / a_value.shape[0]
print a_value[3] / a_value.shape[0]
print a_value[4] / a_value.shape[0]




#print "ANN ", np.mean(output[:])
'''