# import pydot
# print pydot.find_graphviz()

from keras.models import model_from_json
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.optimizers import SGD

import numpy as np
import scipy.misc


def lambda_handler(event, context):
    return 'Lambda done: ' + mode
