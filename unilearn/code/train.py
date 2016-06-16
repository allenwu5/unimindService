import common as cm
import numpy as np
import matplotlib.pyplot as plt

class Train(object):

    epochs = 5 # 10 ~ 20

    def __init__(self):
        self.eta = 0.01

    def run(self, mnist, rhwd):
        for l in xrange(self.epochs):
            penalty = 0
            for ins in mnist.iTrainInstances:
                nnInput = ins.iImage


                # plt.imshow(np.array(nnInput).reshape(28, 28))
                # plt.show()

                # //print("NN Iteration for label: \(ins.iLabel)==============================")


                # //print(">", terminator:" ")
                output = rhwd.NN.forward(nnInput)
                outputLabel = cm.getOutputLabel(output)
                print('Train {} > {}'.format(ins.iLabel, outputLabel))

                if ins.iLabel != outputLabel:
                    penalty += 1
                    # // 10 outputs for 0 ~ 9
                    desiredOutput = [-1 for i in xrange(10)]
                    desiredOutput[ins.iLabel] = 1
                    # //print("<", terminator:" ")
                    rhwd.NN.backPropagate(output, desiredOutput, self.eta)

            print('Epoch {} done'.format(l + 1))

            if penalty == 0:
                break



