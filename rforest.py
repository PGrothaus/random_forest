import sys
import math
import scipy
import random
import numpy as np

class DecisionTree:
    def __init__(self, depthIn, NLabelIn):
        self.N_label  = NLabelIn
        self.depth    = depthIn
        self.shape    = []
        self.splitVal = []
        self.labelVal = []
        self.feature  = np.array([])
        for dd in range( depthIn ):
            self.shape.append( [0] * int( math.pow(2, dd ) ) )
            self.splitVal.append( [0] * int( math.pow(2, dd ) ) )
            self.labelVal.append( [0] * int( math.pow(2, dd ) ) )
        self.shape[0][0] = 1

    def entropy(self, targetArr, targets):
        S = 0.
        modD = np.size(targetArr)[0]
        for target in targets:
            tmp    = np.where(target == targetArr, 1, 0)
            modD_j = np.sum( tmp )
            p_j    = float( modD_j ) / modD 
            S     -= p_j * math.log( p_j, 2. )
        return S
    
    def split_featureArr(self, featureArr, xk ):
        #featureArr must only contain the feature that should
        #split the dataset at xk
        xx = np.shape( xk )[0]
        yy = np.shape( featureArr )[0]
        tmpFeatureArr = np.tile( featureArr, ( xx, 1 ) )
        tmpxk         = np.reshape( xk, ( xx, 1 ) )
        tmpxk         = np.tile( tmpxk, ( 1, yy ) )
        featureTag    = np.where( tmpFeatureArr <= tmpxk, 1., 0. )
        D_0           = np.sum( featureTag, 1 )
        D_1           = yy - D_0
        return D_0, D_1
    
    def maximise_entropy_gain(self, featureArrList, xk, mod_D ):
        #featureArrList should be a list of featureArr in which
        #each featureArr a np.array that contains different 
        #feature values, but the same label y
        S_split = np.zeros( len(xk) )
        S       = np.zeros( len(xk) )
        S1, S2  = np.zeros( len(xk) ), np.zeros( len(xk) )
        D_0_sum = 0.
        D_1_sum = 0.
        D_0_List, D_1_List = [], []
        for featureArr in featureArrList:
            D_0, D_1 = self.split_featureArr(featureArr, xk)
            D_0_List.append( D_0 )
            D_1_List.append( D_1 )
            D_0_sum     += D_0
            D_1_sum     += D_1
            p_i = float( len(featureArr) ) / mod_D 
            if p_i > 0.:
                S  -= p_i * math.log( p_i, 2. )
        for D_0, D_1 in zip(D_0_List, D_1_List):
            p_0  = np.where( D_0_sum == 0., 0., D_0 / D_0_sum )
            p_1  = np.where( D_1_sum == 0., 0., D_1 / D_1_sum )
            w1   = D_0_sum / mod_D * p_0 
            w2   = D_1_sum / mod_D * p_1
            p_0  = np.where( p_0 == 0., 1, p_0 )
            p_1  = np.where( p_1 == 0., 1, p_1 )
            S1  -= w1 * np.log2( p_0 )
            S2  -= w2 * np.log2( p_1 )
    
        S_split =  S1 + S2
        dS = S - S_split
        dS_max, id_max = np.max( dS ), np.argmax( dS )
        if dS_max <= 0.:
            return xk[5]
            return 'none'
        x_split = xk[ id_max ]
        return x_split

    def split_labeled_data( self, ftrArrList, layer_id, node_id, ftr_id):
        L1, L2 = [], []
        x_split = self.splitVal[layer_id][node_id]
        for lbl_id in range( self.N_label ):
            if 0 == len(ftrArrList[node_id][lbl_id]):
                L1.append([])
                L2.append([])
                continue
            L1.append(ftrArrList[node_id][lbl_id]\
                    [ftrArrList[node_id][lbl_id][:,ftr_id] <= x_split])
            L2.append(ftrArrList[node_id][lbl_id]\
                    [ftrArrList[node_id][lbl_id][:,ftr_id]  > x_split])
        return L1, L2

    def split_unlabeled_data( self, ftrArrList, layer_id, node_id, ftr_id):
        L1, L2 = [], []
        x_split = self.splitVal[layer_id][node_id]
        if 0 == len(ftrArrList[node_id]):
            return L1, L2
        L1.append(ftrArrList[node_id]\
                [ftrArrList[node_id][:,ftr_id] <= x_split])
        L2.append(ftrArrList[node_id]\
                [ftrArrList[node_id][:,ftr_id]  > x_split])
        return L1, L2

    def survival_probability(self, ftrArrList ):
        N_total = 0.
        for lbl_id in range( self.N_label ):
            N_total += np.shape( ftrArrList[lbl_id] )[0]
        if 0. == N_total: return 0.
        p_survival = float( np.shape( ftrArrList[1] )[0] ) / N_total
        return p_survival
 
    def train(self, ftrArrList, NsplitVal = 2500):
        N_ftr = np.shape(ftrArrList[0][0])[1] - 1
        print 'There are ', self.N_label, ' labels!'
        if( self.depth < N_ftr ):
            print 'Depth of tree is smaller than number of features'
            print 'Only the first ', self.depth, 'features are used'
        if( self.depth > N_ftr + 1 ):
            print 'Depth of tree is greater than number of features'
            print 'We want to avoid over-fitting!'
        #    print 'Exit'
        #    sys.exit()

        newFtrList = ftrArrList[:]
        for layer in self.shape:
            ftrArrList = newFtrList[:]
            newFtrList = []
            layer_id = self.shape.index( layer )
            if 0 == layer_id%5:
                print layer_id
            if ( (self.depth - 1) == layer_id):
                #last layer is output layer -> don't train
                continue
            ftr_id   = int( random.uniform(0,N_ftr) )
            self.feature = np.concatenate( \
                                (self.feature, np.array([int(ftr_id)]) ) )
                     #layer_id #Change this later when looking at RF

            for jj in range( int( math.pow(2., layer_id ) ) ):
                node_id = jj
                if layer_id > 0 \
                  and 0 == self.shape[layer_id - 1][int( node_id / 2 ) ]:
                    for ii in range(2):
                        newFtrList.append( [] )
                        self.shape[layer_id][node_id] = 0
                    continue
                fArr, tmpL = [], np.array([])
                for lbl_id in range( self.N_label ):
                    if (0,) == np.shape(ftrArrList[node_id][lbl_id]):
                        fArr.append(ftrArrList[node_id][lbl_id])
                    else:
                        fArr.append( ftrArrList[node_id][lbl_id][:,ftr_id])
                    tmpL = np.concatenate( (tmpL, fArr[-1]) )
                mod_D = len(tmpL)
                if 0 == mod_D:
                    for ii in range(2):
                        newFtrList.append( [] )
                        self.shape[layer_id][node_id] = 0
                    continue
                fMax, fMin = np.max(tmpL), np.min(tmpL)
                xkArr = np.linspace(fMin, fMax, NsplitVal)
                x_split = self.maximise_entropy_gain(fArr, xkArr, mod_D )
                if 'none' == x_split:
                    for ii in range(2):
                        newFtrList.append( [] )
                        self.shape[layer_id][node_id] = 0
                    continue

                self.shape[layer_id][node_id]    = 1
                self.splitVal[layer_id][node_id] = x_split
               
                L1, L2 = self.split_labeled_data(ftrArrList, \
                                                layer_id, node_id, ftr_id)
                newFtrList.append( L1 )
                newFtrList.append( L2 )
                self.labelVal[layer_id + 1][2 * node_id]      =\
                                                self.survival_probability(L1)
                self.labelVal[layer_id + 1][2 * node_id + 1 ] = \
                                                self.survival_probability(L2)
        print ''
        print 'Tree trained'

    def predict_label(self, ftrArrList):
        outArr =  np.zeros( (1,np.shape(ftrArrList[0])[1]+1) )
        ftrArrList    = ftrArrList
        newFtrArrList = ftrArrList[:]
        for layer in self.shape:
            ftrArrList   = newFtrArrList[:]
            newFtrArrList = []
            layer_id = self.shape.index( layer )
            if ( (self.depth - 1) != layer_id):
                #last layer is output layer -> don't train
                if 0 == layer_id%5:
                    print layer_id
                #ftr_id   = layer_id #Change this later when looking at RF
                ftr_id   = self.feature[ layer_id ]
            for jj in range( int( math.pow(2., layer_id) ) ):
                node_id = jj
                if 0 == np.shape(ftrArrList[node_id])[0]:
                    #print 'empty', layer_id, node_id
                    newFtrArrList.append( [] )
                    newFtrArrList.append( [] )
                    continue
                if layer_id > 0 \
                  and 0 == self.shape[layer_id][node_id]:
                    #print 'predict', layer_id, node_id
                    p_survival = self.labelVal[layer_id][node_id]
                    dim        = np.shape(ftrArrList[node_id])[0]
                    rnd        = np.random.uniform(0,1, dim )
                    pred_lbl   = np.where( rnd < p_survival, 1, 0 )
                    pred_lbl   = np.reshape( pred_lbl, (dim,1))
                    tmp_out    = np.concatenate((ftrArrList[node_id],\
                                               pred_lbl), axis = 1  )

                    outArr = np.concatenate( (outArr, tmp_out), axis = 0 )
                    newFtrArrList.append( [] )
                    newFtrArrList.append( [] )
                    continue

                #print 'split', layer_id, node_id
                x_split = self.splitVal[layer_id][node_id]
                L1, L2  = self.split_unlabeled_data(ftrArrList, \
                                            layer_id, node_id, ftr_id)
                newFtrArrList += L1 + L2

        print 'labels predicted'
        print ''
        return outArr

#N_samples = 25
#N_ftr = 1
#MockData = []
#
#for ii in range( 2 ):
#    MockData.append( np.random.uniform( ii-.6, ii+.6, \
#                        (N_samples, N_ftr + 1) ) )
#    MockData[ ii ][:,1] = ii
#
#Data = np.arange(0.,1.,.1)
#Data = np.reshape(Data, (10,1))
#
#rf =  DecisionTree( 2, 2 )
#rf.train( [MockData] )
#
#print ''
#print 'where tree is splitted: ',rf.shape
#print 'split values: ',rf.splitVal
#print 'survival probabilities: ',rf.labelVal
#print ''
#
#print rf.predict_label( [Data] )
