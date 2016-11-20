# import pydot
# print pydot.find_graphviz()

from keras.models import model_from_json
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.optimizers import SGD
import os

import numpy as np
import scipy.misc

class Dataset:
    CIFAR10  = 'CIFAR10'
    CIFAR100 = 'CIFAR100'

path = ''

def load_and_scale_imgs(img_names):
    imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)), (2, 0, 1)).astype('float32') for img_name in img_names]
    return np.array(imgs) / 255

def classify(img_names, imgs):
    mode = Dataset.CIFAR10

    # read model from json file into memory
    if mode == Dataset.CIFAR10:
        origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        dataset_dir = cifar10.get_file("cifar-10-batches-py", origin=origin, untar=True)
        class_names = np.load(os.path.join(dataset_dir, "batches.meta"))
        model_json = 'cifar10.json'
        model_h5 = 'cifar10.h5'
        label_name_key = 'label_names'
    if mode == Dataset.CIFAR100:
        origin = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        dataset_dir = cifar100.get_file("cifar-100-python", origin=origin, untar=True)
        class_names = np.load(os.path.join(dataset_dir, "meta"))
        model_json = 'cifar100.json'
        model_h5 = 'cifar100.h5'
        label_name_key = 'fine_label_names'

    model = model_from_json(open(path + model_json).read())
    model.load_weights(path + model_h5)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    #
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png')

    predictions = model.predict_classes(imgs)

    result = ''
    for i in xrange(len(img_names)):
        result += ('<br>{0} = {1}\n'.format(img_names[i], class_names[label_name_key][predictions[i]]))

    return result

if __name__ == '__main__':
    img_names = ['standing-cat.jpg', 'dog-face.jpg', 'bird.jpeg', 'car.jpeg', 'truck.jpeg', 'ape.jpg', 'duck.jpg', 'mustbebird.jpeg',
                 'cnBird.jpeg', 'cnBird2.jpeg', '3birds.jpeg'
                 ]
    for i in xrange(len(img_names)):
        img_names[i] = path + '../cifar_images/' + img_names[i]
    imgs = load_and_scale_imgs(img_names)
    print (classify(img_names, imgs))

