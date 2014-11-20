import numpy as np
import matplotlib.pyplot as plt

a = np.array([1,2,3], dtype=float)
b = np.array([10,11,12])
d = np.array(['teste', 2, 3.5])
c = a + b.astype('float')

print c.shape
print c.size
print c.ndim

x = np.arange(10.0,100.0, 5)

z = np.arange(100).reshape(10,10)
np.set_printoptions(threshold='nan')
print z

N = 20
y = np.zeros(N)
x1 = np.linspace(0, 100, N)
x2 = np.linspace(0, 100, N, endpoint=False)
x3 = np.logspace(0.0, 2.0)
print x1
print x2
print x3
#plt.plot(x1, y, 'o')
#plt.plot(x2, y + 0.5, 'o')
#plt.ylim([-0.5, 1])
#plt.show()