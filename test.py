import numpy as np
from mlp import MLP
from cross_validation import SCV

in_geral = np.loadtxt("./datasets/iris.data")

scv = SCV(in_geral, 5)
t,tt,v, vv = scv.select_fold_combination()

print(t.shape, tt.shape, v.shape, vv.shape)
hide = np.array([2])
print(hide[0])
ann = MLP(t.shape[1], t.shape[1], hide)
ann.set_learningRate(0.95)
ann.set_learningDescent(0.5)
ann.set_momentum(0.02)
ann.set_erro(0.005)
ann.set_epochs(100)
ann.validation_set(v,v)
ann.train_mlp(t, t)

print("Training Error: ", ann.get_validationError())
print("Validation Error: ", ann.get_trainingError())

#ann.save_mlp(ann, "./trained_mlp")

ann.plot_neurons("Neurons")

ann.plot_learning_curve("Neurons")


ann = MLP.load_mlp("./trained_mlp.pk1")