import numpy as np

def titanic( filename, labeled = 1 ):
    dataIn = open( filename )
    lines  = dataIn.readlines()

    idp, idsu, idpc, idse, idag, idNs, idNp, idfa, idpo = 0,1,2,5,6,7,8,10,12
    if 1 == labeled:
        survival, pclass, sex, age, Nsibsp, Nparch, fare, port = \
            np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([])
    else:
        pID, pclass, sex, age, Nsibsp, Nparch, fare, port = \
            np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([])
        idpc, idse, idag, idNs, idNp, idfa, idpo = 1,4,5,6,7,9,11

    FIRST_LINE_READ = False
    for line in lines:
        if False == FIRST_LINE_READ:
            FIRST_LINE_READ = True
            continue
        content  = line.split( ',' )
        if 1 == labeled:
            survival = np.append( survival, float( content[ idsu ] ) )
        else:
            pID = np.append( pID, int( content[ idp ] ) )
        pclass   = np.append( pclass,   float( content[ idpc ] ) )
    
        tmp      = content[ idse ]
        if 'male' == tmp:
            sex = np.append( sex, 0 )
        elif 'female' == tmp:
            sex = np.append( sex, 1 )
        elif 'Mrs' in tmp:
            sex = np.append( sex, 1 )
        elif 'Miss' in tmp:
            sex = np.append( sex, 1 )
        else:
            sex = np.append( sex, 0 )
    
        tmp      = content[ idag ]
        if tmp == '': #will substitute with average age
            age = np.append( age, -1 )
        else:
            age = np.append( age, float( tmp ) )
    
        tmp      = content[ idNs ]
        if '' == tmp:
            Nsibsp = np.append( Nsibsp, -1 )
        else:
            Nsibsp = np.append( Nsibsp, float( tmp ) )
    
        tmp      = content[ idNp ]
        if tmp == '': #will substitute with average Nparch
            Nparch = np.append( Nparch, -1 )
        else:
            Nparch = np.append( Nparch, float( tmp ) )
    
        tmp      = content[ idfa ]
        if tmp == '': #will substitute with average fare
            fare = np.append( fare, -1 )
        else:
            fare = np.append( fare, float( tmp ) )
    
        tmp      = content[ idpo ]
        if  'C' in tmp:
            port = np.append( port, 0 )
        elif 'Q' in tmp:
            port = np.append( port, 1 )
        elif 'S' in tmp:
            port = np.append( port, 2 )
        else: #assumed to have boarded on busiest port - Southampton
            port = np.append( port, 2 )
    
    av_age     = np.sum( age ) / len( age )
    age        = np.where( age == -1, av_age, age )
    av_fare    = np.sum( fare ) / len( fare )
    fare       = np.where( fare == -1, av_fare, fare )
    av_Nsibsp  = np.sum( Nsibsp ) / len( Nsibsp )
    Nsibsp     = np.where( Nsibsp == -1, av_Nsibsp, Nsibsp )
    av_Nparch  = np.sum( Nparch ) / len( Nparch )
    Nparch     = np.where( Nparch == -1, av_Nparch, Nparch )
    
    #print np.sum(survival)
    
    age      = np.reshape( age, (len(age), 1) )
    if 1 == labeled:
        survival = np.reshape( survival, (len(age), 1) )
    else:
        pID  = np.reshape( pID, (len(age), 1) ) 
    pclass   = np.reshape( pclass, (len(age), 1) )
    sex      = np.reshape( sex, (len(age), 1) )
    fare     = np.reshape( fare, (len(age), 1) )
    Nsibsp   = np.reshape( Nsibsp, (len(age), 1) )
    Nparch   = np.reshape( Nparch, (len(age), 1) )
    port     = np.reshape( port, (len(age), 1) )
    
    if 1 == labeled:
        data = np.concatenate( (pclass, sex, age, fare, \
                                Nsibsp, Nparch, port, survival ), axis = 1 )
    else:
        data = np.concatenate( (pclass, sex, age, fare, \
                                Nsibsp, Nparch, port, pID ), axis = 1 )

    if 1 == labeled:
        dead  = data[data[:,7] == 0]
        alive = data[data[:,7] == 1]
        dataList = [dead, alive ]
        return dataList
    else:
        return [data]
