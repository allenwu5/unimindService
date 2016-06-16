import cnn
import random

class RecognizeDigits(object):
    ULONG_MAX = 65535
    iInputLen = 28

    def __init__(self):
        self.NN = cnn.NeuralNetwork()
        self.iInputArea = self.iInputLen * self.iInputLen

    def initNN(self):
        # // initialize and build the neural net


        # // NN.initialize()

        # // layer zero, the input layer.
        # // Create neurons: exactly the same number of neurons as the input
        # // vector of 29x29=841 pixels, and no weights/connections


        pLayer0 = cnn.Layer('Layer00')
        self.NN.layers.append(pLayer0)

        for ii in xrange(self.iInputArea):
            pLayer0.neurons.append(cnn.Neuron())

        # // layer one:
        # // This layer is a convolutional layer that
        # // has 6 feature maps.  Each feature
        # // map is 13x13, and each unit in the
        # // feature maps is a 5x5 convolutional kernel
        # // of the input layer.
        # // So, there are 13x13x6 = 1014 neurons, (5x5+1)x6 = 156 weights

        featMapCount = 20
        featMapSize = 12
        featMapArea = featMapSize * featMapSize
        kernelSize = 5
        kernelArea = kernelSize * kernelSize
        kernelWeightCount = 1 + kernelArea

        neuronCount = featMapCount * featMapArea #// feat count times feat map area
        weightCount = featMapCount * kernelWeightCount #// feat count times kernel weight


        pLayer1 = cnn.Layer('Layer01',pLayer0)

        self.NN.layers.append(pLayer1)

        for ii in xrange(neuronCount):
            pLayer1.neurons.append(cnn.Neuron())

        for ii in xrange(weightCount):
            # // uniform random distribution
            pLayer1.weights.append(0.05 * self.UNIFORM_PLUS_MINUS_ONE())

        # // interconnections with previous layer: this is difficult
        # // The previous layer is a top-down bitmap
        # // image that has been padded to size 29x29
        # // Each neuron in this layer is connected
        # // to a 5x5 kernel in its feature map, which
        # // is also a top-down bitmap of size 13x13.
        # // We move the kernel by TWO pixels, i.e., we
        # // skip every other pixel in the input image


        kernelTemplate = []
        for i in xrange(kernelSize):
            for j in xrange(kernelSize):
                kernelTemplate.append(i * self.iInputLen + j)

# //        int fm;  // "fm" stands for "feature map"

        # // 29^2 becomes 20 x 12^2 neurons and 20 x (1 + 5^2) weights

        move = 2
        for fm in xrange(featMapCount):
            for ii in xrange(featMapSize):
                for jj in xrange(featMapSize):
                    # // 26 is the number of weights per feature map
                    numWeight = fm * kernelWeightCount
                    n = pLayer1.neurons[ jj + ii*featMapSize + fm*featMapArea ]

                    n.AddConnection( self.ULONG_MAX, numWeight)  #// bias weight
                    numWeight+=1

                    for kk in xrange(kernelArea):
                        # // note: max val of index == 840,
                        # // corresponding to 841 neurons in prev layer
                        #
                        # // convolutino here...
                        n.AddConnection( move * (jj + self.iInputLen * ii) + kernelTemplate[kk], numWeight)
                        numWeight+=1

        # // layer two:
        # // This layer is a convolutional layer
        # // that has 50 feature maps.  Each feature
        # // map is 5x5, and each unit in the feature
        # // maps is a 5x5 convolutional kernel
        # // of corresponding areas of all 6 of the
        # // previous layers, each of which is a 13x13 feature map
        # // So, there are 5x5x50 = 1250 neurons, (5x5+1)x6x50 = 7800 weights

        l2FeatMapCount = 50
        l2FeatMapSize = 4
        l2FeatMapArea = l2FeatMapSize * l2FeatMapSize
        l2NeuronCount = l2FeatMapCount * l2FeatMapArea
        l2WeightCount = kernelWeightCount * featMapCount * l2FeatMapCount


        pLayer2 = cnn.Layer('Layer02', pLayer1 )
        self.NN.layers.append( pLayer2 )

        for ii in xrange(l2NeuronCount):
            pLayer2.neurons.append(cnn.Neuron())

        for ii in xrange(l2WeightCount):
            pLayer2.weights.append(0.05 * self.UNIFORM_PLUS_MINUS_ONE())

        # // Interconnections with previous layer: this is difficult
        # // Each feature map in the previous layer
        # // is a top-down bitmap image whose size
        # // is 13x13, and there are 6 such feature maps.
        # // Each neuron in one 5x5 feature map of this
        # // layer is connected to a 5x5 kernel
        # // positioned correspondingly in all 6 parent
        # // feature maps, and there are individual
        # // weights for the six different 5x5 kernels.  As
        # // before, we move the kernel by TWO pixels, i.e., we
        # // skip every other pixel in the input image.
        # // The result is 50 different 5x5 top-down bitmap
        # // feature maps


        kernelTemplate2 = []
        for i in xrange(kernelSize):
            for j in xrange(kernelSize):
                kernelTemplate2.append(i * featMapSize + j)

        maxNeuronIndex = 0
        for fm in xrange(l2FeatMapCount):
            for ii in xrange(l2FeatMapSize):
                for jj in xrange(l2FeatMapSize):
                    # // 26 is the number of weights per feature map
                    numWeight = fm * kernelWeightCount;
                    n = pLayer2.neurons[ jj + ii*l2FeatMapSize + fm*l2FeatMapArea ]

                    n.AddConnection( self.ULONG_MAX, numWeight )  #// bias weight
                    numWeight += 1

                    for kk in xrange(l2FeatMapArea):
                        # // note: max val of index == 1013,
                        # // corresponding to 1014 neurons in prev layer
                        for l in xrange(featMapCount):
                            neuronIndex = ((l * featMapArea) + move * (jj + featMapSize * ii) +
                                kernelTemplate2[kk])
                            n.AddConnection(neuronIndex , numWeight)
                            numWeight += 1
                            maxNeuronIndex = max(maxNeuronIndex, neuronIndex)

        assert(maxNeuronIndex < neuronCount)

        # // layer three:
        # // This layer is a fully-connected layer
        # // with 100 units.  Since it is fully-connected,
        # // each of the 100 neurons in the
        # // layer is connected to all 1250 neurons in
        # // the previous layer.
        # // So, there are 100 neurons and 100*(1250+1)=125100 weights

        l3NeuronCount = 500
        l3WeightCount = l3NeuronCount * (1 + l2NeuronCount) #// 500 * (1 + 50 * 4 ^ 2)

        pLayer3 = cnn.Layer('Layer03', pLayer2 )
        self.NN.layers.append( pLayer3 )

        for ii in xrange(l3NeuronCount):
            pLayer3.neurons.append(cnn.Neuron() )

        for ii in xrange(l3WeightCount):
            pLayer3.weights.append(0.05 * self.UNIFORM_PLUS_MINUS_ONE())

        # // Interconnections with previous layer: fully-connected

        numWeight = 0;  #// weights are not shared in this layer

        for fm in xrange(l3NeuronCount):
            n = pLayer3.neurons[ fm ]
            n.AddConnection( self.ULONG_MAX, numWeight)  #// bias weight
            numWeight += 1

            for ii in xrange(l2NeuronCount): # // // 50 * 4 ^ 2
                n.AddConnection( ii, numWeight)
                numWeight += 1

        # // layer four, the final (output) layer:
        # // This layer is a fully-connected layer
        # // with 10 units.  Since it is fully-connected,
        # // each of the 10 neurons in the layer
        # // is connected to all 100 neurons in
        # // the previous layer.
        # // So, there are 10 neurons and 10*(100+1)=1010 weights

        lfNeuronCount = 10
        lfWeightCount = lfNeuronCount * (1 + l3NeuronCount) #// 10 * (1 + 500)

        pLayer4 = cnn.Layer('Layer04', pLayer3)
        self.NN.layers.append( pLayer4 )

        for ii in xrange(lfNeuronCount):
            pLayer4.neurons.append(cnn.Neuron())

        for ii in xrange(lfWeightCount):
            pLayer4.weights.append(0.05 * self.UNIFORM_PLUS_MINUS_ONE())

        # // Interconnections with previous layer: fully-connected

        numWeight = 0;  #// weights are not shared in this layer

        for fm in xrange(lfNeuronCount):
            n = pLayer4.neurons[ fm ]
            n.AddConnection( self.ULONG_MAX, numWeight)  #// bias weight
            numWeight += 1

            for ii in xrange(0, l3NeuronCount):
                n.AddConnection( ii, numWeight)
                numWeight += 1

    def UNIFORM_PLUS_MINUS_ONE(self):
        # // random value range: -1.0 ~ 1.0
        return random.uniform(-1, 1)