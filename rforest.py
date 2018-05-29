import sys
import math
import scipy
import random
import numpy as np


class DecisionTree:
    def __init__(self, depthIn, nftr, NLabelIn):
        self.n_features = nftr
        self.N_label = NLabelIn
        self.depth = depthIn
        self.shape = []
        self.splitVal = []
        self.labelVal = []
        self.feature = []
        for dd in range(depthIn):
            self.shape.append([0] * int(math.pow(2, dd)))
            self.splitVal.append([0] * int(math.pow(2, dd)))
            self.labelVal.append([0] * int(math.pow(2, dd)))
            self.feature.append([0] * int(math.pow(2, dd)))
        self.shape[0][0] = 1

    def entropy(self, targetArr, targets):
        S = 0.
        modD = np.size(targetArr)[0]
        for target in targets:
            tmp = np.where(target == targetArr, 1, 0)
            modD_j = np.sum(tmp)
            p_j = float(modD_j) / modD
            S -= p_j * math.log(p_j, 2.)
        return S

    def split_featureArr(self, featureArr, xk):
        # featureArr must only contain the feature that should
        # split the dataset at xk
        xx = np.shape(xk)[0]
        yy = np.shape(featureArr)[0]
        tmpFeatureArr = np.tile(featureArr, (xx, 1))
        tmpxk = np.reshape(xk, (xx, 1))
        tmpxk = np.tile(tmpxk, (1, yy))
        featureTag = np.where(tmpFeatureArr <= tmpxk, 1., 0.)
        D_0 = np.sum(featureTag, 1)
        D_1 = yy - D_0
        return D_0, D_1

    def entropy(self, a, b):
        tot = a + b
        s1, s2 = 0., 0.
        if a > 0:
            a1 = 1. * a / tot
            s1 = -a1 * math.log(a1, 2.)
        if b > 0:
            b1 = 1. * b / tot
            s2 = - b1 * math.log(b1, 2.)
        return s1 + s2

    def maximise_entropy_gain(self, featureArrList, xk, mod_D):
        alive, dead = featureArrList[0], featureArrList[1]
        s_t = self.entropy(len(alive), len(dead))
        tot = len(alive) + len(dead)
        gain_max = 0.
        x_best = None
        for xval in xk:
            alive_less = len([item for item in alive if item <= xval])
            alive_greater = len([item for item in alive if item > xval])
            dead_less = len([item for item in dead if item <= xval])
            dead_greater = len([item for item in dead if item > xval])
            p_less = 1. * (alive_less + dead_less) / tot
            p_greater = 1. * (alive_greater + dead_greater) / tot

            s_less = self.entropy(alive_less, dead_less)
            s_greater = self.entropy(alive_greater, dead_greater)

            s_x = p_less * s_less + p_greater * s_greater

            gain = s_t - s_x
            if gain > gain_max:
                gain_max = gain
                x_best = xval
        if gain_max > 0.001:
            return x_best, gain_max
        else:
            return None, None

    def split_labeled_data(self, ftrArrList, layer_id, node_id, ftr_id):
        L1, L2 = [], []
        x_split = self.splitVal[layer_id][node_id]
        for lbl_id in range(self.N_label):
            if 0 == len(ftrArrList[node_id][lbl_id]):
                L1.append([])
                L2.append([])
                continue
            L1.append(ftrArrList[node_id][lbl_id]
                      [ftrArrList[node_id][lbl_id][:, ftr_id] <= x_split])
            L2.append(ftrArrList[node_id][lbl_id]
                      [ftrArrList[node_id][lbl_id][:, ftr_id] > x_split])
        return L1, L2

    def split_unlabeled_data(self, ftrArrList, layer_id, node_id, ftr_id):
        L1, L2 = [], []
        x_split = self.splitVal[layer_id][node_id]
        if 0 == len(ftrArrList[node_id]):
            return L1, L2
        L1.append(ftrArrList[node_id]
                  [ftrArrList[node_id][:, ftr_id] <= x_split])
        L2.append(ftrArrList[node_id]
                  [ftrArrList[node_id][:, ftr_id] > x_split])
        return L1, L2

    def survival_probability(self, ftrArrList):
        N_total = 0.
        for lbl_id in range(self.N_label):
            N_total += np.shape(ftrArrList[lbl_id])[0]
        if 0. == N_total:
            return 0.
        p_survival = float(np.shape(ftrArrList[1])[0]) / N_total
        return p_survival

    def train(self, ftrArrList, NsplitVal=25):
        N_ftr = np.shape(ftrArrList[0][0])[1] - 1
        newFtrList = ftrArrList[:]
        for layer in self.shape:
            ftrArrList = newFtrList[:]
            newFtrList = []
            layer_id = self.shape.index(layer)
            if ((self.depth - 1) == layer_id):
                continue

            for node_id in range(int(math.pow(2., layer_id))):
                if layer_id > 0 \
                        and 0 == self.shape[layer_id - 1][int(node_id / 2)]:
                    split_features = [[], []]
                    for f in split_features:
                        newFtrList.append(f)
                    continue
                fArr, tmpL = [], np.array([])
                splits = []
                for ftr_id_ in range(self.n_features):
                    tmpL = [ftrArrList[node_id][lbl_id][:, ftr_id_]
                            for lbl_id in range(self.N_label)]
                    flattened = [item for l1 in tmpL for item in l1]
                    mod_D = len(flattened)
                    if flattened:
                        fMax, fMin = np.max(flattened), np.min(flattened)
                        xkArr = np.linspace(fMin, fMax, NsplitVal)
                        x_split, dS = self.maximise_entropy_gain(
                            tmpL, xkArr, mod_D)
                        if x_split is None:  # note: it can be 0.
                            continue
                        else:
                            splits.append((ftr_id_, x_split, dS))
                if splits:
                    split = 1
                    dSs = [item[2] for item in splits]
                    maxds = max(dSs)
                    idx = dSs.index(maxds)
                    x_split = splits[idx][1]
                    ftr_id = splits[idx][0]
                else:
                    split = 0
                    x_split = None
                    ftr_id = None
                    split_features = [[], []]
                    for f in split_features:
                        newFtrList.append(f)
                self.feature[layer_id][node_id] = ftr_id
                self.shape[layer_id][node_id] = split
                self.splitVal[layer_id][node_id] = x_split
                if ftr_id is not None:
                    L1, L2 = self.split_labeled_data(
                        ftrArrList, layer_id, node_id, ftr_id)
                    #print len(L1[0]), len(L1[1]), len(L2[0]), len(L2[1])
                    newFtrList.append(L1)
                    newFtrList.append(L2)
                    self.labelVal[layer_id + 1][2 *
                                                node_id] = self.survival_probability(L1)
                    self.labelVal[layer_id + 1][2 * node_id +
                                                1] = self.survival_probability(L2)

#        print 'Fully trained'
#        print 'layer_id, node_id, feature_id, split_val, surivival_probability'
#        for i in range(self.depth):
#            for jj in range(int(math.pow(2., i))):
#                print i, jj, self.feature[i][jj], self.splitVal[i][jj],  self.labelVal[i][jj]

    def predict_label(self, ftrArrList):
        outArr = np.zeros((1, np.shape(ftrArrList[0])[1] + 1))
        newFtrArrList = ftrArrList[:]
        for layer in self.shape:
            ftrArrList = newFtrArrList[:]
            newFtrArrList = []
            layer_id = self.shape.index(layer)
            for jj in range(int(math.pow(2., layer_id))):
                node_id = jj
                ftr_id = self.feature[layer_id][node_id]
                if 0 == np.shape(ftrArrList[node_id])[0]:
                    newFtrArrList.append([])
                    newFtrArrList.append([])
                    continue
                if layer_id > 0 \
                        and 0 == self.shape[layer_id][node_id]:
                    p_survival = self.labelVal[layer_id][node_id]
                    p_survival = 1 if p_survival > 0.5 else 0
                    dim = np.shape(ftrArrList[node_id])[0]
                    rnd = np.random.uniform(0, 1, dim)
                    pred_lbl = np.where(rnd < p_survival, 1, 0)
                    pred_lbl = np.reshape(pred_lbl, (dim, 1))
                    tmp_out = np.concatenate((ftrArrList[node_id],
                                              pred_lbl), axis=1)

                    outArr = np.concatenate((outArr, tmp_out), axis=0)
                    newFtrArrList.append([])
                    newFtrArrList.append([])
                    continue

                x_split = self.splitVal[layer_id][node_id]
                L1, L2 = self.split_unlabeled_data(ftrArrList,
                                                   layer_id, node_id, ftr_id)
                newFtrArrList += L1 + L2

        return outArr
