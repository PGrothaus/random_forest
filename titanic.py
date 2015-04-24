import numpy as np
import rforest
import data_preparation as dataPrep

trainData = dataPrep.titanic( 'train.csv', labeled = 1 )

print np.shape( trainData[0] )
#trainData, testData = [ trainData[0][:400,:], trainData[1][:400,:] ], \
#     [ np.concatenate( (trainData[0][401:,:], trainData[1][401:,:]), axis=0) ]

N_ftr  = 7
depth  = N_ftr
Nlabel = 2

RF = rforest.DecisionTree( N_ftr, 2 )
RF.train( [trainData] )

#print ''
#print 'where tree is splitted: ', RF.shape
#print 'split values: ',           RF.splitVal
#print 'survival probabilities: ', RF.labelVal
#print ''
output = RF.predict_label( trainData )
misclass  = np.where( output[:,7]==output[:,8], 0, 1 )
Nmisclass = float( np.sum( misclass ) )
error     = Nmisclass  / np.shape( misclass )[0]
precision = ( np.shape( misclass )[0] - Nmisclass ) / np.shape( misclass )[0]
print 'error and precision on training se: ',error, precision
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


