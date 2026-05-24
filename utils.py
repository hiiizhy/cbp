import numpy as np
import torch

def generateBatchSamples(dataLoader, batchIdx, config, isEval):

    samples, sampleLen, userhis, itemhis, his, target, userPOP, itemPOP = dataLoader.batchLoader(batchIdx, config.isTrain, isEval)

    maxLenSeq = max([len(userLen) for userLen in sampleLen])

    maxLenBas = max([max(userLen) for userLen in sampleLen])

    paddedSamples = []
    paddedDecays  = []
    paddedOffset  = []
    targetList    = []
    lenList       = []
    for user in samples:

        trainU  = user[:-1]
        testU   = user[-1]
        targetList.append(testU)

        paddedU = []
        decayU  = []
        offsetU = []
        lenList.append([len(trainU) - 1])
        decayNum = len(trainU)-1
        for eachBas in trainU:
            paddedBas = eachBas + [config.padIdx] * (maxLenBas - len(eachBas))
            paddedU.append(paddedBas)
            decayU.append(config.decay ** decayNum)
            decayNum -= 1
            offsetU.append(maxLenBas/float(len(eachBas)))
        paddedU = paddedU + [[config.padIdx] * maxLenBas] * (maxLenSeq - len(paddedU))
        decayU  = decayU + [0] * (maxLenSeq - len(decayU))
        offsetU = offsetU + [0] * (maxLenSeq-len(offsetU))
        #add a sample
        paddedSamples.append(paddedU) # [batch, maxLenSeq]
        paddedDecays.append(decayU)   # [batch, maxLenSeq]
        paddedOffset.append(offsetU)  # [batch, maxLenSeq]
 
    paddedOffset = np.asarray(paddedOffset).reshape(len(samples), -1, 1)  # [batch, maxLenSeq,1]

    #1-hot vectors
    lenTen = torch.tensor(lenList, dtype=torch.long)
    lenX   = torch.FloatTensor(len(samples), maxLenSeq)
    lenX.zero_()
    lenX.scatter_(1,lenTen,1)
    return np.asarray(paddedSamples), np.asarray(paddedDecays).reshape(len(samples), -1, 1), \
           userhis, itemhis, his, target, paddedOffset, lenX, targetList, userPOP, itemPOP