import numpy as np
import rforest
import sys
import data_preparation as dataPrep
import matplotlib.pyplot as plt

def bagging( data, idset):
    out = []
    for lbl in data:
        out.append( lbl[idset,:] )
    return out

trainData = dataPrep.titanic( 'train.csv', labeled = 1 )

#plt.plot( trainData[0][:,1],\
#           trainData[0][:,2] , 'ro')
#plt.plot( trainData[1][:,1]+.2,\
#           trainData[1][:,2] , 'go')
#plt.show()
#sys.exit()

#print np.shape( trainData[0] )
trainData, crossValData = [ trainData[0][:250,:], trainData[1][:250,:] ], \
     [ np.concatenate( (trainData[0][251:,:], trainData[1][251:,:]), axis=0) ]
N_ftr  = 6
depth  = 20
Nlabel = 2
Ntotal = 5

finalSurv = np.zeros( 419 )
for Ntree in range( Ntotal ):
    if Ntree%5 ==0:
        print Ntree
    for lbl in trainData:
        np.random.shuffle( lbl )
    idset = np.random.randint( 0, 250, 150 ) 
    baggedData = bagging(trainData, idset)
    RF = rforest.DecisionTree( depth, 2 )
    RF.train( [baggedData] )
    
    output = RF.predict_label( baggedData )
    misclass  = np.where( output[:,N_ftr+1]==output[:,N_ftr], 0, 1 )
    Nmisclass = float( np.sum( misclass ) )
    error     = Nmisclass  / np.shape( misclass )[0]
    precision = ( np.shape( misclass )[0] - Nmisclass ) / np.shape( misclass )[0]
    print 'error and precision on training set: ',error, precision
    
    output = RF.predict_label( crossValData )
    misclass  = np.where( output[:,N_ftr+1]==output[:,N_ftr], 0, 1 )
    Nmisclass = float( np.sum( misclass ) )
    error     = Nmisclass  / np.shape( misclass )[0]
    precision = ( np.shape( misclass )[0] - Nmisclass ) / np.shape( misclass )[0]
    print 'error and precision on cross validation set: ',error, precision
    print ''
   
    testData = dataPrep.titanic( 'test.csv', labeled = 0 )
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


