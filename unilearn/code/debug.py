import numpy as np
import matplotlib.pyplot as plt

class Way(object):
    n0Size = 29
    n1Size = 12
    n2Size = 4

    cmap_theme = 'summer'
    interpolation = None

    w3 = None

    def getW3(self):
        if w3 is None:
            self.w3 = np.loadtxt(self.path + 'Layer03_weights.json')
        return self.w3

    def __init__(self, path = ""):
        assert(len(path) > 1)
        self.path = path

        self.n0 = np.loadtxt(path + 'Layer00_neurons.json')
        self.n1 = np.loadtxt(path + 'Layer01_neurons.json')
        self.n2 = np.loadtxt(path + 'Layer02_neurons.json')
        self.n3 = np.loadtxt(path + 'Layer03_neurons.json')
        self.n4 = np.loadtxt(path + 'Layer04_neurons.json')

        self.w1 = np.loadtxt(path + 'Layer01_weights.json')
        self.w2 = np.loadtxt(path + 'Layer02_weights.json')
        # self.w3 = np.loadtxt(path + 'Layer03_weights.json')
        self.w4 = np.loadtxt(path + 'Layer04_weights.json')

    def showN0(self):
        plt.imshow(self.n0.reshape(self.n0Size, self.n0Size), cmap=self.cmap_theme, interpolation=self.interpolation)
        plt.show()

    def showN1(self, index = 0):
        plt.imshow(self.n1.reshape(20 * self.n1Size, self.n1Size), cmap=self.cmap_theme, interpolation=self.interpolation)
        plt.show()

    def showN2(self, index = 0):
        plt.imshow(self.n2.reshape(5, 10 * self.n2Size, self.n2Size)[index], cmap=self.cmap_theme, interpolation=self.interpolation)
        plt.show()

    def showN3(self, index = 0):
        plt.plot(self.n3)
        plt.ylabel('n3')
        plt.draw()

    def showN4(self, index = 0):
        plt.plot(self.n4)
        plt.ylabel('n4')
        plt.draw()

    def showW1(self, index = 0):
        # w1 = 20 x 1 x ( 1 + 5 x 5)
        s = self.w1.reshape(20, 1, 26)[index][0]
        s = s[1:].reshape(5, 5)
        plt.imshow(s, cmap=self.cmap_theme, interpolation=self.interpolation)
        plt.show()

    def showW2(self, index = 0):
        # w2 = 50 x 20 x ( 1 + 5 x 5)
        s = self.w2.reshape(50, 20, 26)[index][0]
        s = s[1:].reshape(5, 5)
        plt.imshow(s, cmap=self.cmap_theme, interpolation=self.interpolation)
        plt.show()

    def showW1Plot(self):
        s = self.w1
        plt.plot(s)
        plt.ylabel('w1')
        plt.draw()

    def showW2Plot(self, index = 0):
        s = self.w2#.reshape(13, 2000)[0]
        plt.plot(s)
        plt.ylabel('w2')
        plt.draw()

    def showW3Plot(self, index = 0):
        s = self.getW3()
        plt.plot(s)
        plt.ylabel('w3')
        plt.draw()

    def showW4Plot(self, index = 0):
        s = self.w4
        plt.plot(s)
        plt.ylabel('w4')
        plt.draw()



m = Way('/Users/len/Documents/work/unimindService/unilearn/debug/mike/')
t = Way('/Users/len/Documents/work/unimindService/unilearn/debug/theano/')

index = 19

# first blue, second green

# t.showW2Plot()
t.showW2()
plt.show()


# m.showW2Plot()
m.showW2()

plt.show()

# 29^2


# w1 = 20 x 1 x ( 1 + 5 x 5)
#
# plt.plot(w1)
# plt.ylabel('w1')
# plt.show()


#
# s = w1.reshape(20, 1, 26)[0][0][1:]
# plt.imshow(s.reshape(5, 5), cmap='summer')
# plt.show()

