import numpy as np
import rforest
import sys
import random
import data_preparation as dataPrep
import matplotlib.pyplot as plt

Ntr = 200
N_ftr = 10
depth = N_ftr + 1
Nbag = 200
Nlabel = 2
Ntotal = 25

def calculate_error(output):
    n_examples = output.shape[0]
    prediction = output[:, N_ftr]
    target = output[:, N_ftr + 1]
    misclass = np.where(target == prediction, 0, 1)
    error = np.mean(misclass)
    precision = 1. - error
    return error, precision


np.random.seed(1)
random.seed(1)

trainData = dataPrep.titanic('train.csv', LABELED=True)
trainData  = [trainData[0][:Ntr, :], trainData[1][:Ntr, :]]
crossValData = [np.concatenate(
        (trainData[0][Ntr + 1:, :],
         trainData[1][Ntr + 1:, :]), axis=0)]



testData = dataPrep.titanic('test.csv', LABELED=False)

finalSurv = np.zeros(419)
for Ntree in range(Ntotal):
    if Ntree % 10 == 0:
        print Ntree
    for lbl in trainData:
        np.random.shuffle(lbl)
    idset = np.random.randint(0, Ntr, Nbag)
    baggedData = [lbl[idset, :] for lbl in data]
    RF = rforest.DecisionTree(depth, 2)
    RF.train(baggedData)

    output = RF.predict_label(baggedData)
    error, precision = calculate_error(output)
    print 'error and precision on training set: {}, {}'.format(
        round(error, 3), round(precision, 3))

    output = RF.predict_label(crossValData)
    error, precision = calculate_error(output)
    print 'error and precision on validation set: {}, {}\n'.format(
        round(error, 3), round(precision, 3))

    output = RF.predict_label(testData)
    output = zip(output[:, N_ftr], output[:, N_ftr + 1])
    output.sort(key=lambda tup: tup[0])
    pId, survival = zip(*output)

    finalSurv += survival

finalSurv = 1. * finalSurv / Ntotal

survival = np.where(finalSurv >= .5, 1, 0)
with open('submit.txt', 'w') as f:
    f.write('PassengerId,Survived\n')
for ii in range(np.shape(pId)[0]):
    if 0 == ii:
        continue
    with open('submit.txt', 'a') as f:
        f.write(str(int(pId[ii])) + ',' + str(int(survival[ii])) + '\n')
