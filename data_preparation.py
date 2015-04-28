import numpy as np

def update_collection( item, collection, sums, survSums):
    if item in collection:
        idx = collection.index( item )
    else:
        collection.append( item )
        idx = collection.index( item )
        sums = np.insert( sums, np.shape( sums), 0 )
        survSums = np.insert( survSums, np.shape( survSums ), 0 )
    sums[ idx ] += 1
    return collection, sums, survSums

def get_average( collection ):
    tmp     = np.where( -1 == collection, 1, 0 )
    Nmiss   = np.sum( tmp )
    total   = np.sum( collection ) + Nmiss
    average = total / ( np.shape( collection )[0] - Nmiss )
    return average
    
def get_title( item ):
    itemlist = list( item )
    count = 0
    out = ''
    for char in itemlist:
        if char == ' ':
            count += 1
        if count >= 0:
            out = out+char
        if char =='.':
            break
    return out

def fill_array( tmp, array ):
    if tmp == '': #will substitute with average age
        array = np.append( array, -1 )
    else:
        array = np.append( array, float( tmp ) )
    return array


def titanic( filename, LABELED = True ):
    collection, sums, survSums = [], np.array([]), np.array([])
    pclass, sex, age      = np.array([]), np.array([]), np.array([]) 
    Nsibsp, Nparch, fare  = np.array([]), np.array([]), np.array([]) 
    port, survival, title = np.array([]), np.array([]), np.array([])

    idp, idsu, idpc, idtit, idse, idag, idNs, idNp, idfa, idpo = \
      0,    1,    2,     3,    5,    6,    7,    8,   10,   12
    if not LABELED:
        pID = np.array( [] )
        idpc, idtit, idse, idag, idNs, idNp, idfa, idpo = \
           1,     2,    4,    5,    6,    7,    9,   11

    dataIn = open( filename )
    lines  = dataIn.readlines()

    FIRST_LINE_READ = False
    for line in lines:
        if False == FIRST_LINE_READ:
            FIRST_LINE_READ = True
            continue
        content  = line.split( ',' )
        if LABELED:
            survival = np.append( survival, float( content[ idsu ] ) )
        else:
            pID = np.append( pID, int( content[ idp ] ) )
        
        tmp = content[ idpc ]
        if tmp == '':
            pclass = np.append( pclass, -1 )
        else:
            pclass   = np.append( pclass, float( tmp ) )
   
        item = content[idtit+1]
        item = get_title( item )
        collection, sums, survSums = \
                        update_collection( item, collection, sums,survSums)
        idx = collection.index( item )
        survSums[ idx ] += float( content[idsu] )

        tmp = content[ idse ]
        SEX = True
        if 'male' == tmp:
            sex = np.append( sex, 0 )
            SEX = False
        if 'female' == tmp:
            sex = np.append( sex, 1 )
            SEX = False
        if 'Mrs' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 0 )
        elif 'Miss' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 0 )
        elif 'Ms' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 1 )
        elif 'Mme' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 1 )
        elif 'Lady' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 2 )
        elif 'Mlle' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 1 )
        elif 'Countess' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 2 )
        elif 'Dona' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 2 )
        elif 'Jonkheer' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 2 )
        elif 'Mr' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 3 )
        elif 'Master' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 4 )
        elif 'Sir' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 4 )
        elif 'Cap' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 5 )
        elif 'Don' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 5 )
        elif 'Major' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 5 )
        elif 'Dr' in content[idtit]:
            if SEX: sex     = np.append( sex, 1 )
            title = np.append( title, 5 )
        elif 'Col' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 5 )
        elif 'Rev' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 5 )
        else:
            title = np.append( title, 2 )
            if SEX: sex   = np.append( sex, 0 )

        tmp    = content[ idag ]
        age    = fill_array( tmp, age ) 
   
        tmp    = content[ idNs ]
        Nsibsp = fill_array( tmp, Nsibsp )    

        tmp    = content[ idNp ]
        Nparch = fill_array( tmp, Nparch )
    
        tmp    = content[ idfa ]
        fare   = fill_array( tmp, fare )
    
        tmp    = content[ idpo ]
        if  'C' in tmp:
            port = np.append( port, 0 )
        elif 'Q' in tmp:
            port = np.append( port, 1 )
        elif 'S' in tmp:
            port = np.append( port, 2 )
        else: #assumed to have boarded on busiest port - Southampton
            port = np.append( port, 0 )
    
    #print collection
    #print sums
    #print survSums
    av_age     = get_average( age )
    age        = np.where(    age == -1, av_age, age )
    av_fare    = get_average( fare )
    fare       = np.where(   fare == -1, av_fare, fare )
    av_Nsibsp  = get_average( Nsibsp )
    Nsibsp     = np.where( Nsibsp == -1, av_Nsibsp, Nsibsp )
    av_Nparch  = get_average( Nparch )
    Nparch     = np.where( Nparch == -1, av_Nparch, Nparch )
    av_class   = get_average( pclass )
    pclass     = np.where( pclass == -1, av_class, pclass )
    
    if LABELED:
        survival = np.reshape( survival, (len(age), 1) )
    else:
        pID  = np.reshape( pID,    (len(age), 1) ) 
    age      = np.reshape( age,    (len(age), 1) )
    pclass   = np.reshape( pclass, (len(age), 1) )
    sex      = np.reshape( sex,    (len(age), 1) )
    fare     = np.reshape( fare,   (len(age), 1) )
    Nsibsp   = np.reshape( Nsibsp, (len(age), 1) )
    Nparch   = np.reshape( Nparch, (len(age), 1) )
    port     = np.reshape( port,   (len(age), 1) )
    title    = np.reshape( title,  (len(age), 1) )
    family   = Nsibsp + Nparch + 1
   
    if LABELED:
        data = np.concatenate( (sex, pclass, age, fare, age**2, fare**2, \
                                age*fare, family, survival),\
                                #pclass, sex, age, fare, \
                                #Nsibsp, Nparch, port, \
                                #family, title, survival ), 
                                axis = 1 )
    else:
        data = np.concatenate( (sex, pclass, age, fare, age**2, fare**2,\
                                age*fare, family,  \
                                #pclass, sex, age, fare, \
                                #Nsibsp, Nparch, port, \
                                #family, title,\
                                pID ), axis = 1 )

    if LABELED:
        dead  = data[data[:,-1] == 0]
        alive = data[data[:,-1] == 1]
        dataList = [dead, alive ]
        return dataList
    else:
        return [data]
