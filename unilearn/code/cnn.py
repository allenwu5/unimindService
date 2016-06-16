import numpy as np

class NeuralNetwork(object):

    iArchive = 'Cache/neuralNetworkArchive'

    def __init__(self):
        self.layers = []

    # // Think with known weights
    def forward(self, input):
        firstLayer = self.layers[0]

        assert(len(firstLayer.neurons) == len(input))
        # // feed input to first layer
        for i in xrange(len(input)):
            firstLayer.neurons[i].value = input[i]

        # firstLayer.debugPrint()

        # // forward layer by layer
        for i in xrange(1, len(self.layers)):
            self.layers[i].forward()
            # layers[i].debugPrint()

        lastLayer = self.layers[-1]
        # // get outputs via last layer

        output = []
        for i in xrange(len(lastLayer.neurons)):
            output.append(lastLayer.neurons[i].value)

        return output

    # // Training weights
    def backPropagate(self, actualOutput, desiredOutput, eta):
        # // Xnm1 means Xn-1
        #
        # // Backpropagates through the neural net
        # // Proceed from the last layer to the first, iteratively
        # // We calculate the last layer separately, and first,
        # // since it provides the needed derviative
        # // (i.e., dErr_wrt_dXnm1) for the previous layers
        #
        # // nomenclature:
        # //
        # // Err is output error of the entire neural net
        # // Xn is the output vector on the n-th layer
        # // Xnm1 is the output vector of the previous layer
        # // Wn is the vector of weights of the n-th layer
        # // Yn is the activation value of the n-th layer,
        # // i.e., the weighted sum of inputs BEFORE
        # //    the squashing function is applied
        # // F is the squashing function: Xn = F(Yn)
        # // F' is the derivative of the squashing function
        # //   Conveniently, for F = tanh,
        # //   then F'(Yn) = 1 - Xn^2, i.e., the derivative can be
        # //   calculated from the output, without knowledge of the input

# w, h = 8, 5.
# Matrix = [[0 for x in range(w)] for y in range(h)]

        differentials = []
        for ii in xrange(len(self.layers)):
            differentials.append([0.0 for x in xrange(len(self.layers[ii].neurons))])


# //        int iSize = m_Layers.size();
#
# //        differentials.resize( iSize );


        # // start the process by calculating dErr_wrt_dXn for the last layer.
        # // for the standard MSE Err function
        # // (i.e., 0.5*sumof( (actual-target)^2 ), this differential is simply
        # // the difference between the target and the actual

        lastLayer = self.layers[-1]
        for ii in xrange(len(lastLayer.neurons)):
            differentials[-1][ ii ] = actualOutput[ ii ] - desiredOutput[ ii ]
            # //print(differentials[differentials.count-1][ ii ])


#         // store Xlast and reserve memory for
#         // the remaining vectors stored in differentials
# //
# //        differentials[ differentials.count-1 ] = dErr_wrt_dXlast  // last one


        # // now iterate through all layers including
        # // the last but excluding the first, and ask each of
        # // them to backpropagate error and adjust
        # // their weights, and to return the differential
        # // dErr_wrt_dXnm1 for use as the input value
        # // of dErr_wrt_dXn for the next iterated layer

# //        let eta:Double = 0.0005;
        for bIt in xrange(len(differentials)- 1, -1, -1):
            self.layers[bIt].backPropagate(differentials[bIt], differentials[bIt - 1], eta)

class Neuron(object):

    def __init__(self):
        self.value = 0.0
        self.connections = []

    def AddConnection(self, neuronIndex, weightIndex):
        self.connections.append(Connection(neuronIndex, weightIndex))

class Connection(object):

    def __init__(self, neuronIndex, weightIndex):
        self.neuronIndex = neuronIndex
        self.weightIndex = weightIndex

class Layer(object):
    ULONG_MAX = 65535
    dSigmoidFactor = 0.66666667 / 1.7159

    def sigmoid(self, f):
        return 1.7159 * np.tanh(0.66666667 * f)
    def dSigmoid(self, f):
        return self.dSigmoidFactor * (1.7159 + f ) * (1.7159 - f)

    def __init__(self, label, prev = None):
        self.label = label
        self.prev = prev
        self.neurons = []
        self.weights = []

    def forward(self):
        assert(len(self.prev.neurons) > 0)
        # must be different layer (i.e. different label)
        assert(self.prev.label != self.label)

        for n in self.neurons:
            firstConn = n.connections[0]

            assert(firstConn.weightIndex < len(self.weights))

            # // weight of the first connection is the bias;
            # // its neuron-index is ignored

            bias = self.weights[firstConn.weightIndex]

            sum = 0.0

            for i in xrange(1, len(n.connections)):
                conn = n.connections[i]
                assert(conn.weightIndex < len(self.weights))
                assert(conn.neuronIndex < len(self.prev.neurons))

                sum += self.weights[conn.weightIndex] * self.prev.neurons[conn.neuronIndex].value

            # // activation function
            n.value = self.sigmoid(sum + bias)

    def backPropagate(self, dErr_wrt_dXn, dErr_wrt_dXnm1, eta):
        dErr_wrt_dYn = []
        # // calculate equation (3): dErr_wrt_dYn = F'(Yn) * dErr_wrt_Xn

        for ii in xrange(len(self.neurons)):
            output = self.neurons[ ii ].value
            dErr_wrt_dYn.append(self.dSigmoid( output ) * dErr_wrt_dXn[ ii ])

        # // calculate equation (4): dErr_wrt_Wn = Xnm1 * dErr_wrt_Yn
        # // For each neuron in this layer, go through
        # // the list of connections from the prior layer, and
        # // update the differential for the corresponding weight

        dErr_wrt_dWn = [0.0 for i in xrange(len(self.weights))]

        ii = 0
        for n in self.neurons:
            # // for simplifying the terminology

            for c in n.connections:
                x = 0.0
                nIdx = c.neuronIndex

                if nIdx == self.ULONG_MAX:
                    x = 1.0  #// this is the bias weight
                else :
                    x = self.prev.neurons[nIdx].value

                dErr_wrt_dWn[ c.weightIndex ] += dErr_wrt_dYn[ ii ] * x
            ii+=1

        # // calculate equation (5): dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn,
        # // which is needed as the input value of
        # // dErr_wrt_Xn for backpropagation of the next (i.e., previous) layer
        # // For each neuron in this layer

        ii = 0
        for n in self.neurons:
            # // for simplifying the terminology

            for c in n.connections:
                nIdx = c.neuronIndex
                if nIdx != self.ULONG_MAX:
                    # // we exclude ULONG_MAX, which signifies
                    # // the phantom bias neuron with
                    # // constant output of "1",
                    # // since we cannot train the bias neuron

                    # // nIndex = kk

                    dErr_wrt_dXnm1[ nIdx ] += dErr_wrt_dYn[ ii ] * self.weights[ c.weightIndex ]

            ii+=1  #// ii tracks the neuron iterator

        # // calculate equation (6): update the weights
        # // in this layer using dErr_wrt_dW (from
        # // equation (4)    and the learning rate eta
        #
        # // turing bias too here

        for jj in xrange(len(self.weights)):
            oldValue = self.weights[ jj ]
            d = dErr_wrt_dWn[ jj ]
            diff = eta * d

            newValue = oldValue - diff

            self.weights[jj] = newValue