import numpy as np
from mlp import MLP
from cross_validation import SCV

in_geral = np.loadtxt("./datasets/breast-cancer.data")

#scv = SCV(in_geral, 2)
#t,tt,v, vv = scv.select_fold_combination()

#print(t.shape, tt.shape, v.shape, vv.shape)
hide = np.array([5])
ann = MLP.load_mlp("./trained_mlp.pk1")
ret = ann.predict(in_geral)

print("Training Error: ", ann.get_validationError())
print("Validation Error: ", ann.get_trainingError())

print(ret)

ann.plot_learning_curve()