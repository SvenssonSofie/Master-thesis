import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt
start_time = time.time()

recRate2 = 0.4
c = 0.1/(1-recRate2)
d = 0.3/(1-recRate2)
coupon = 0.01#pricing for

tr = np.array([0.0, 0.03, 0.06, 0.12, 1.0])/(1-recRate2)
conRa = [0.83553467, 1.38710938,  1.88051758, 2.209375]

############# S24 main 3Y
pi = np.array([0.00087479504671772457, 0.0029353834489070918, 0.004969393415826806, 0.0069769703635795821, 0.0090249736847164419, 0.011068753219014127, 0.013086171126700852, 0.015077371637674086, 0.017108668751384948, 0.019135776533014792, 0.021188431563949561])
spread = np.array([0.0047228532506136903, 0.0048101666897295524, 0.0048253362059923867, 0.0048316350692150416, 0.0048351736249719465, 0.0048374002858278193, 0.0048389188662819015, 0.0048400205026313961, 0.0048408789399059949, 0.0048415530671146135, 0.0048489188205022606])
discFac = np.array([1.0003661802496806, 1.0010318855144935, 1.0017322165435656, 1.0024681809694533, 1.0031665273659054, 1.0037827350418536, 1.0039805436318214, 1.0039247441142243, 1.0039599948710676, 1.0041599686436489, 1.0043651935486468])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25555555555555554])


conRatio = 0.0 # for new tranche
for i in range(int(100*c), int(100*d)):#2-7
    ind = min(np.where(tr > i/100.0)[0]) #index in tranche min of greater than or equal to i*100, say 2-7
    conRatio = conRatio + conRa[ind-1]
conRatio = conRatio/(d-c)/100.0

print conRatio

nIss = 125#Number of issuers
T = len(delta)
sumDelta = np.cumsum(delta)
nom = 1000000.0

recRate = 0.0
C = math.ceil(c*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(d*nIss)) #nbr of losses detachment
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D

nbrSim = 10000
uniAll = np.random.uniform(size = (nbrSim, nIss))#to optimize over the same random numbers, common random numbers


a = np.zeros(len(pi))
for k in range(0, len(spread)): 
    if k == 0: 
        a[k] = -(1-pi[k]-1)/delta[k]
    else:
        a[k] = -(1 - pi[k] - 1 + pi[k-1])/(1-pi[k-1])/delta[k-1]
print a
print np.mean(a)
a = np.mean(a)

rep = 100
price = np.zeros(rep)
for j in range(0, rep):
    uniAll = np.random.uniform(size = (nbrSim, nIss))#to optimize over the same random numbers, common random numbers
    pvMean = np.zeros((2,nbrSim))
    
    for sim in range(0,nbrSim):
        uni = uniAll[sim,:]
        e = -np.log(1-uni)
        eSort = np.sort(e)
        defTime = np.zeros(nIss)
        defTime[0] = eSort[0]/a
        for nbr in range(1,len(defTime)):             
            defTime[nbr] = defTime[nbr - 1] + (eSort[nbr] - eSort[nbr - 1])/(a*(1+nbr*conRatio))
        defTime = defTime*sumDelta[-1]
    
        
        loss = np.zeros(T)#if no value is inserted on a position, then all has defaulted before that timeStep and thus should the loss of the following time steps be 100%
        for t in range(0,T):
            if sumDelta[t] < defTime[0]: continue
            for tau in range(1,len(defTime)):
                if defTime[tau] > sumDelta[t] and defTime[tau - 1] < sumDelta[t]: 
                    loss[t] = tau
                    break
                else: 
                    loss[t] = nIss
        
    
        #loss=np.sum(np.repeat(defTime.reshape(1,nIss),len(sumDelta),0)<=np.repeat(sumDelta.reshape(len(sumDelta),1),nIss,1),1)    
        
        
        ################ tranche loss
        lossTr = np.zeros(T) #add rows to calculate tranches simultaneously
        for t in range(0,T):
            if loss[t] <= C: continue
            elif loss[t] > C and loss[t] <= D: 
                lossTr[t] = (loss[t] - C)*Lavg
            else: 
                lossTr[t] = (D-C)*Lavg
        #lossTr=Lavg*np.maximum(np.minimum(loss,D)-C,0)
        ################Present value defualt leg
        
          
        pvDl = 0.0 
        for t in range(0,T):
            if t == 0:
                pvDl =  1/discFac[t]*lossTr[0]
            else:
                pvDl = pvDl + 1/discFac[t]*(lossTr[t] - lossTr[t-1])    
            
        #pvDl=1/discFac[0]*lossTr[0]+np.sum(1/discFac[1:len(delta)]*(lossTr[1:len(delta)] - lossTr[0:len(delta)-1]))
        ############ present value premium leg    
        pvPl = 0.0
        for t in range(0,T):
            pvPl = pvPl + delta[t]/discFac[t]*((D-C)*Lavg-lossTr[t])        
        pvPl = coupon*pvPl
        
        #pvPl=coupon*np.sum(delta/discFac*((D-C)*Lavg-lossTr))
        pvMean[0, sim] = pvDl
        pvMean[1, sim] = pvPl
 
    pvMean = np.mean(pvMean,1)
    price[j] = pvMean[0] - pvMean[1]
    print("--- %s seconds ---" % (time.time() - start_time)), j
     
print 'Price: ', np.mean(price), 'std: ', np.std(price)
print("--- %s seconds ---" % (time.time() - start_time))