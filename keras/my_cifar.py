# import pydot
# print pydot.find_graphviz()

from keras.models import model_from_json
from keras.datasets import cifar10
from keras.optimizers import SGD

import numpy as np
import scipy.misc

def load_and_scale_imgs(img_names):
    for i in xrange(len(img_names)):
        img_names[i] = '../cifar_images/' + img_names[i]

    imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)), (2, 0, 1)).astype('float32') for img_name in img_names]
    return np.array(imgs) / 255

def load_mapping():
    mapping = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    return mapping

if __name__ == '__main__':
    img_names = ['standing-cat.jpg', 'dog-face.jpg', 'bird.jpeg', 'car.jpeg', 'truck.jpeg', 'ape.jpg', 'duck.jpg', 'mustbebird.jpeg',
                 'cnBird.jpeg', 'cnBird2.jpeg', '3birds.jpeg'
                 ]
    imgs = load_and_scale_imgs(img_names)

    model = model_from_json(open('cifar10.json').read())
    model.load_weights('cifar10.h5')

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png')



    predictions = model.predict_classes(imgs)

    mapping = load_mapping()
    for i in xrange(len(img_names)):
        print('{0} = {1}'.format(img_names[i], mapping[predictions[i]]))

