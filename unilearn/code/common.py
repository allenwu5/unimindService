def getOutputLabel(aOutput):
    idx = -1
    max = 0.0
    for i in xrange(len(aOutput)):
        if idx == -1 or aOutput[i] >= max:
            idx = i
            max = aOutput[i]

    return idx