
'''
Autoencoder
'''

import sys
import getopt
import numpy as np
from mlp import MLP
from cross_validation import SCV

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def run_mlp(dataset_path, neurons):
	dataset_input = np.loadtxt(dataset_path)
	dataset_output = dataset_input[:, - 1]
	input = dataset_input[:,:-1]
	print(input)
	print(dataset_output)
	scv = SCV(dataset_input, 2)
	training, training_out, validation, validation_out = scv.select_fold_combination()
	print("training ", training)
	print("training out ",training_out)
	print("validation ",validation)
	print("validation_out ", validation_out)
	hide = np.array([int(neurons)])
	print(training.shape[1])
	print(hide[0])
	ann = MLP(training.shape[1],training.shape[1], hide)
	ann.set_learningRate(0.95)
	ann.set_learningDescent(0.5)
	ann.set_momentum(0.02)
	ann.set_erro(0.005)
	ann.validation_set(validation, validation)
	ann.train_mlp(training, training)
	print("Validation Error: ", ann.get_validationError())
	print("Training Error: ", ann.get_trainingError())
	title = str(neurons) + " Neurons"
	ann.save_json_mlp(ann, "./jsonmlp")
	#ann.plot_learning_curve(title)
	#ann.plot_neurons(title)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
            raise Usage(msg)

        run_mlp(argv[1], argv[2])

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())



