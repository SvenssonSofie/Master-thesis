import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt

##########Calculates contagion ration for C-D, do this for all tranches and put them into pricingContMod

#test
#spread = 0.00676750939073
marketPrice = 388889.446134
C = 0.0
D = 0.03

#marketPrice = 84493.4899999
#C = 0.03
#D = 0.06

marketPrice = 5476.81999264
C = 0.06
D = 0.12

delta = [0.252777777778, 0.252777777778      , 0.252777777778 , 0.255555555556   , 0.252777777778,      \
         0.25,    0.255555555556      , 0.255555555556    , 0.252777777778      , 0.25, \
              0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
              0.255555555556 , 0.252777777778, 0.252777777778   , 0.261111111111 , 0.252777777778, 0.252777777778]
print len(delta)
discFac = [0.998600956242      ,0.999611180239      ,  1.00047630826       , 1.00112395119       ,\
          1.00180019558       ,  1.00252355848       , 1.00320411939       , 1.0036842191        ,\
           1.00377352897       , 1.00368518481       , 1.00372980688       , 1.0038363132        ,\
          1.00395370531       ,  1.0040649568        , 1.00392689128       , 1.00358551835       , \
          1.00315265354    , 1.00262525655    , 1.0018353653        ,    1.00087710692   ,0.999799458336]
print len(discFac)

pi = [0.0, 0.0, 0.0039230364813, 0.00396739004003, 0.00397766902931, 0.00398224178954, \
      0.00398489559746, 0.00398659905109, 0.00398777551688, 0.00398863618887, 0.00398931119962,\
     0.00398984448104, 0.00399027231814, 0.00454277200564, 0.00501692809448,  0.0054198805189, \
     0.00576302421683, 0.00606175067052, 0.00633221440352,0.00656349712714, 0.00677161234034]
print len(pi)



nIss = 125.0#Number of issuers

nom = 1000000.0
sumDelta = np.cumsum(delta)
a = pi[-1]/sum(delta)
T = len(delta)
#generate 50 uniform standard random values
nbrSim = 1000
uniAll = np.random.uniform(size = (nbrSim, nIss))#to optimize over the same random numbers, common random numbers
coupon = 0.01
start_time = time.time()

recRate = 0.0
C = math.ceil(C*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(D*nIss)) #nbr of losses detachment
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
print C, D, Lavg

def contagion(c):
    print c
    #if c < 0:
    #    return float('inf')
    
    pvMean = np.zeros((2,nbrSim))
    for sim in range(0,nbrSim):
        uni = uniAll[sim,:]
        e = -np.log(1-uni)
        eSort = np.sort(e)
        defTime = np.zeros(nIss)
        defTime[0] = eSort[0]/a
        for nbr in range(1,len(defTime)):             
            defTime[nbr] = defTime[nbr - 1] + (eSort[nbr] - eSort[nbr - 1])/(a*(1+nbr*c))
        #print defTime
    
        loss = np.zeros(T)#if no value is inserted on a position, then all has defaulted before that timeStep and thus should the loss of the following time steps be 100%
        for t in range(0,T):
            if sumDelta[t] < defTime[0]: continue
            for tau in range(1,len(defTime)):
                #print tau
                if defTime[tau] > sumDelta[t] and defTime[tau - 1] < sumDelta[t]: 
                    loss[t] = tau
                    break
                else: 
                    loss[t] = 125.0          
        #print sumDelta
        #print loss
        
        ################ tranche loss
        lossTr = np.zeros(T) #add rows to calculate tranches simultaneously
        for t in range(0,T):
            if loss[t] <= C: continue
            elif loss[t] > C and loss[t] <= D: 
                lossTr[t] = (loss[t] - C)*Lavg
            else: 
                lossTr[t] = (D-C)*Lavg
        #print lossTr #
        
        ################Present value defualt leg    
        pvDl = 0.0 
        for t in range(0,T):
            if t == 0:
                pvDl = pvDl + 1/discFac[t]*lossTr[t]
            else:
                pvDl = pvDl + 1/discFac[t]*(lossTr[t] - lossTr[t-1])    
        #print pvDl
        
        ############ present value premium leg    
        pvPl = 0.0
        for t in range(0,T):
            pvPl = pvPl + delta[t]/discFac[t]*((D-C)*Lavg-lossTr[t])        
        pvPl = coupon*pvPl
        #print pvPl
        
        pvMean[0, sim] = pvDl#pvMean[0, sim - 1] + pvDl
        pvMean[1, sim] = pvPl# pvMean[1, sim - 1] + pvPl

    
    pvMean = np.mean(pvMean,1)#/nbrSim

    print 'Present value, PVdl, PVpl: ', pvMean  
    print (pvMean[0] - pvMean[1])# - marketPrice  
    return abs((pvMean[0] - pvMean[1]) - marketPrice)

#contRatio = contagion(0.4)
#c = opt.fmin(contagion, [0.3])
c = opt.fmin_cobyla(contagion, [5.0], (), args=(), rhoend = 0.00001)
print 'True ratio: ', c
print("--- %s seconds ---" % (time.time() - start_time))