import numpy as np
from mlp import MLP
from cross_validation import SCV

in_geral = np.loadtxt("./datasets/breast-cancer.data")

scv = SCV(in_geral, 2)
t,tt,v, vv = scv.select_fold_combination()

print(t.shape, tt.shape, v.shape, vv.shape)
hide = np.array([5])
ann = MLP(t.shape[1], t.shape[1], hide)
ann.validation_set(v,v)
ann.train_mlp(t, t)

print("Training Error: ", ann.get_validationError())
print("Validation Error: ", ann.get_trainingError())

if (ann.get_validationError() < 0.02):
	ann.save_mlp(ann, "./trained_mlp")

ann.plot_learning_curve()