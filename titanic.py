import numpy as np
import rforest
import sys
import random
import data_preparation as dataPrep
import matplotlib.pyplot as plt

np.random.seed( 1 )
random.seed( 1 )
def bagging( data, idset):
    out = []
    for lbl in data:
        out.append( lbl[idset,:] )
    return out

trainData = dataPrep.titanic( 'train.csv', LABELED = True )

#plt.plot( trainData[0][:,1],\
#           trainData[0][:,2] , 'ro')
#plt.plot( trainData[1][:,1]+.2,\
#           trainData[1][:,2] , 'go')
#plt.show()
#sys.exit()

#print np.shape( trainData[0] )
Ntr = 200
trainData, crossValData = [ trainData[0][:Ntr,:], trainData[1][:Ntr,:] ], \
   [np.concatenate( (trainData[0][Ntr+1:,:], trainData[1][Ntr+1:,:]), axis=0)]

N_ftr  = 8
depth  = 20
Nbag   = 150
Nlabel = 2
Ntotal = 1

finalSurv = np.zeros( 419 )
for Ntree in range( Ntotal ):
    if Ntree%10 ==0:
        print Ntree
    for lbl in trainData:
        np.random.shuffle( lbl )
    idset = np.random.randint( 0, Ntr, Nbag ) 
    baggedData = bagging(trainData, idset)
    RF = rforest.DecisionTree( depth, 2 )
    RF.train( [baggedData] )
    
    output = RF.predict_label( baggedData )
    misclass  = np.where( output[:,N_ftr+1]==output[:,N_ftr], 0, 1 )
    Nmisclass = float( np.sum( misclass ) )
    error     = Nmisclass  / np.shape( misclass )[0]
    precision = (np.shape( misclass )[0]-Nmisclass) / np.shape( misclass )[0]
    print 'error and precision on training set: ', \
            round(error,2), round(precision,2)
    
    output = RF.predict_label( crossValData )
    misclass  = np.where( output[:,N_ftr+1]==output[:,N_ftr], 0, 1 )
    Nmisclass = float( np.sum( misclass ) )
    error     = Nmisclass  / np.shape( misclass )[0]
    precision = ( np.shape( misclass )[0] - Nmisclass ) / np.shape( misclass )[0]
    print 'error and precision on validati set: ',\
            round(error,2), round(precision,2)
    print ''
   
    testData = dataPrep.titanic( 'test.csv', LABELED = False )
    output = RF.predict_label( testData )
    
    output = zip( output[:,N_ftr], output[:,N_ftr+1] )
    output.sort(key = lambda tup: tup[0] )
    output  = zip( *output )
    pId, survival = output

    finalSurv += survival

finalSurv = 1./Ntotal * finalSurv

rr = np.random.uniform(0.,1., 419)
survival = np.where( finalSurv >= .5, 1, 0 )
f = open( 'submit.txt', 'w' )
f.write('PassengerId,Survived\n' )
f.close()
for ii in range( np.shape( pId )[0] ):
    if 0 == ii:
        continue
    f = open( 'submit.txt', 'a' )
    f.write(str(int(pId[ii]))+','+str(int(survival[ii]))+'\n')
    f.close()


