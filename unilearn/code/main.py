import common as cm
import mnist as mt
import learn_mnist as lm
import train as ta
import cProfile

if __name__ == '__main__':

    pr = cProfile.Profile()
    pr.enable()

    # Constant
    useCache = False
    useTheanoWeight = False

    trainCases = 100 # 60000
    testCases = 100 # 10000

    # // Do any additional setup after loading the view, typically from a nib.

    print('Minist reading==============================')
    mnist = mt.Mnist('/Users/len/Documents/work/unimindService/MNIST') # path
    mnist.load_training(trainCases)
    mnist.load_testing(testCases)

    print('NN init==============================')
    rhwd = lm.RecognizeDigits()
    rhwd.initNN()

    if useCache:

        # let result = (useTheanoWeight && rhwd.NN.loadFromTheano()) || rhwd.NN.loadFromFile()
        # if (!result)
        # {
        #     print("Cannot load cache")
        #     return;
        # }
        raise Exception('useCache is not implemented yet !')
    else:
        train = ta.Train()
        train.run(mnist, rhwd)
        # rhwd.NN.saveToFile()

    print("NN Recall============================== \(NSDate())")


    penalty = 0
    total = 0
    for ins in mnist.iTestInstances:
        nnInput = ins.iImage

        output = rhwd.NN.forward(nnInput)
        outputLabel = cm.getOutputLabel(output)
        print('Test {} > {}'.format(ins.iLabel, outputLabel))

        if ins.iLabel != outputLabel:
            print(outputLabel)
            penalty += 1
    # //            ins.printImage()
        total+=1

    print("Done============================== \(NSDate())")
    print("Penalty: {} / {}".format(penalty, total))

    pr.disable()
    # after your program ends
    pr.print_stats(sort="calls")