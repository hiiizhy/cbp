import torch
import numpy as np
import random
import time
from utils import *
from evaluation import evaluate_ranking
from model import Model
import torch.nn as nn


def training(dataLoader, config, device, writer):
    if config.isTrain:
        numUsers = dataLoader.numTrain
        numItems = dataLoader.numItemsTrain
    else:
        numUsers = dataLoader.numTrainVal
        numItems = dataLoader.numItemsTest

    if numUsers % config.batch_size == 0:
        numBatch = numUsers // config.batch_size
    else:
        numBatch = numUsers // config.batch_size + 1
    idxList = [i for i in range(numUsers)]

    model = Model(config, numUsers, numItems)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.opt == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    elif config.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                                        centered=False)

    for epoch in range(config.epochs):
        random.seed(1234)
        random.shuffle(idxList)  # userID
        timeEpStr = time.time()
        epochLoss = 0

        for batch in range(numBatch):
            start = config.batch_size * batch
            end = min(numUsers, start + config.batch_size)

            batchList = idxList[start:end]

            samples, decays, \
            userhis, itemhis, his, target, offset, lenX, targetList, \
            userPOP, itemPOP = generateBatchSamples(dataLoader,
                                                    batchList,
                                                    config,
                                                    isEval=0)
            users = torch.from_numpy(np.array(batchList)).type(torch.LongTensor).to(device)
            samples = torch.from_numpy(samples).type(torch.LongTensor).to(device)
            decays = torch.from_numpy(decays).type(torch.FloatTensor).to(device)
            userhis = torch.from_numpy(userhis).type(torch.FloatTensor).to(device)
            itemhis = torch.from_numpy(itemhis).type(torch.FloatTensor).to(device)
            his = torch.from_numpy(his).type(torch.FloatTensor).to(device)
            userPOP = torch.from_numpy(userPOP).type(torch.FloatTensor).to(device)
            itemPOP = torch.from_numpy(itemPOP).type(torch.FloatTensor).to(device)
            target = torch.from_numpy(target).type(torch.FloatTensor).to(device)
            offset = torch.from_numpy(offset).type(torch.FloatTensor).to(device)
            lenX = lenX.to(device)

            scores = model.forward(users, samples, decays, offset,
                                      userhis, itemhis, his, userPOP, itemPOP, device)

            loss_scores = -(torch.log(scores) * target +
                            torch.log(1 - scores) * (1 - target)).sum(-1).mean()

            loss = loss_scores

            epochLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epochLoss = epochLoss / float(numBatch)
        timeEpEnd = time.time()

        if epoch % 1 == 0:
            timeEvalStar = time.time()
            print("start evaluation")

            recall, ndcg = evaluate_ranking(model, dataLoader, config, device, config.isTrain)

            timeEvalEnd = time.time()

            output_str = "Epoch %d \t recall@5=%.6f, recall@10=%.6f" \
                         "ndcg@5=%.6f, ndcg@10=%.6f, [%.1f s]" % (
                             epoch + 1, recall[0], recall[1],
                             ndcg[0], ndcg[1],
                             timeEvalEnd - timeEvalStar)

            print("num_epoch: %d, elaspe: %.1f, loss: %.3f" % (epoch, timeEpEnd - timeEpStr, epochLoss))
            print(output_str)
