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

def titanic( filename, labeled = 1 ):
    sums, survSums = np.array ([]), np.array([])
    collection = []
    dataIn = open( filename )
    lines  = dataIn.readlines()

    idp, idsu, idpc, idtit, idse, idag, idNs, idNp, idfa, idpo = \
         0,1,2,3,5,6,7,8,10,12
    if 1 == labeled:
        survival, pclass, sex, age, Nsibsp, Nparch, fare, port = \
            np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([])
    else:
        pID, pclass, sex, age, Nsibsp, Nparch, fare, port = \
            np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([])
        idpc, idtit, idse, idag, idNs, idNp, idfa, idpo = 1,2,4,5,6,7,9,11
    title = np.array( [] )

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
        
        tmp = content[ idpc ]
        if tmp == '':
            pclass = np.append( pclass, 0 )
        else:
            pclass   = np.append( pclass,   float( tmp ) )
   
        item = content[idtit+1]
        #print item
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
        item = out
        collection, sums, survSums = \
                        update_collection( item, collection, sums,survSums)
        idx = collection.index( item  )
        survSums[ idx ] += float( content[idsu] )
        tmp      = content[ idse ]
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
            title = np.append( title, 1 )
        elif 'Mlle' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 1 )
        elif 'Countess' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 1 )
        elif 'Dona' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 1 )
        elif 'Jonkheer' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 1 )
        elif 'Mr' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 2 )
        elif 'Master' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 3 )
        elif 'Sir' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 3 )
        elif 'Cap' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 4 )
        elif 'Don' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 4 )
        elif 'Major' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 4 )
        elif 'Dr' in content[idtit]:
            if SEX: sex     = np.append( sex, 1 )
            title = np.append( title, 4 )
        elif 'Col' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 4 )
        elif 'Rev' in content[idtit]:
            if SEX: sex = np.append( sex, 1 )
            title = np.append( title, 3 )
        else:
            title = np.append( title, 1 )
            if SEX: sex   = np.append( sex, 0 )
        tmp      = content[ idag ]
        if tmp == '': #will substitute with average age
            age = np.append( age, 0 )
        else:
            age = np.append( age, float( tmp ) )
    
        tmp      = content[ idNs ]
        if '' == tmp:
            Nsibsp = np.append( Nsibsp, 0 )
        else:
            Nsibsp = np.append( Nsibsp, float( tmp ) )
    
        tmp      = content[ idNp ]
        if tmp == '': #will substitute with average Nparch
            Nparch = np.append( Nparch, 0 )
        else:
            Nparch = np.append( Nparch, float( tmp ) )
    
        tmp      = content[ idfa ]
        if tmp == '': #will substitute with average fare
            fare = np.append( fare, 0 )
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
    
    #print collection
    #print sums
    #print survSums
    length     = np.shape( np.nonzero(age) )[1]
    av_age     = np.sum( age ) / length
    age        = np.where( age == 0, av_age, age )
    length     = np.shape( np.nonzero(fare) )[1]
    av_fare    = np.sum( fare ) / length
    fare       = np.where( fare == 0, av_fare, fare )
    length     = np.shape( np.nonzero(Nsibsp) )[1]
    av_Nsibsp  = np.sum( Nsibsp ) / length
    Nsibsp     = np.where( Nsibsp == 0, av_Nsibsp, Nsibsp )
    length     = np.shape( np.nonzero(Nparch) )[1]
    av_Nparch  = np.sum( Nparch ) / length
    Nparch     = np.where( Nparch == 0, av_Nparch, Nparch )
    length     = np.shape( np.nonzero(pclass) )[1]
    av_class   = np.sum( pclass ) / length
    pclass     = np.where( pclass == 0 , av_class, pclass )
    
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
    family   = Nsibsp + Nparch
    title    = np.reshape( title, (len(age), 1) )
   
    if 1 == labeled:
        data = np.concatenate( (pclass, sex, age, fare, \
                                #Nsibsp, Nparch, port, \
                                family, title, survival ), axis = 1 )
    else:
        data = np.concatenate( (pclass, sex, age, fare, \
                                #Nsibsp, Nparch, port, \
                                family, title, pID ), axis = 1 )

    if 1 == labeled:
        dead  = data[data[:,6] == 0]
        alive = data[data[:,6] == 1]
        dataList = [dead, alive ]
        return dataList
    else:
        return [data]
