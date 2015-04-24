import numpy as np
import rforest
import data_preparation as dataPrep

trainData = dataPrep.titanic( 'train.csv', labeled = 1 )

print np.shape( trainData[0] )
trainData, crossValData = [ trainData[0][:300,:], trainData[1][:300,:] ], \
     [ np.concatenate( (trainData[0][301:,:], trainData[1][301:,:]), axis=0) ]

N_ftr  = 7
depth  = 22
Nlabel = 2

RF = rforest.DecisionTree( depth, 2 )
RF.train( [trainData] )

output = RF.predict_label( trainData )
misclass  = np.where( output[:,7]==output[:,8], 0, 1 )
Nmisclass = float( np.sum( misclass ) )
error     = Nmisclass  / np.shape( misclass )[0]
precision = ( np.shape( misclass )[0] - Nmisclass ) / np.shape( misclass )[0]
print 'error and precision on training set: ',error, precision
print ''

output = RF.predict_label( crossValData )
misclass  = np.where( output[:,7]==output[:,8], 0, 1 )
Nmisclass = float( np.sum( misclass ) )
error     = Nmisclass  / np.shape( misclass )[0]
precision = ( np.shape( misclass )[0] - Nmisclass ) / np.shape( misclass )[0]
print 'error and precision on cross validation set: ',error, precision
print ''

testData = dataPrep.titanic( 'test.csv', labeled = 0 )
output = RF.predict_label( testData )

output = zip( output[:,7], output[:,8] )
output.sort(key = lambda tup: tup[0] )
output  = zip( *output )
pId, survival = output

f = open( 'submit.txt', 'w' )
f.write('PassengerId,Survived\n' )
f.close()
for ii in range( np.shape( pId )[0] ):
    if 0 == ii:
        continue
    f = open( 'submit.txt', 'a' )
    f.write(str(int(pId[ii]))+','+str(int(survival[ii]))+'\n')
    f.close()


