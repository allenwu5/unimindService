import os
import struct
from array import array

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot

"""
Source: https://github.com/sorki/python-mnist/
"""

class MnistInstance(object):
    def __init__(self, image, label):
        self.iImage = image
        self.iLabel = label


class Mnist(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.iTrainInstances = []
        self.iTestInstances = []

    def load_testing(self, count):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        assert(ims.shape[0] == len(labels))
        for i in xrange(count):
            self.iTestInstances.append(MnistInstance((ims[i]/128.0).reshape(784).tolist(), labels[i]))


    def load_training(self, count):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        assert(ims.shape[0] == len(labels))
        for i in xrange(count):
            self.iTrainInstances.append(MnistInstance((ims[i]/128.0).reshape(784).tolist(), labels[i]))

    @classmethod
    def load(cls, path_img, path_lbl):
        count = 0
        with open(path_lbl, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", f.read())
            count = len(labels)

        with open(path_img, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            # image_data = array("B", file.read())
            img = np.fromfile(f, dtype=np.uint8).reshape(count, rows, cols)

        # images = []
        # for i in range(size):
        #     images.append([0] * rows * cols)
        #
        # for i in range(size):
        #     images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return img, labels

    @classmethod
    def show(cls, image):
        """
        Render a given numpy.uint8 2D array of pixel data.
        """
        imgplot = pyplot.imshow(image, cmap=mpl.cm.Greys)
        imgplot.set_interpolation('nearest')
        pyplot.show()

