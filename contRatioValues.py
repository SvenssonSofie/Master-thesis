import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt

##########Calculates contagion ration for C-D, do this for all tranches and put them into pricingContMod
nIss = 125.0#Number of issuers
delta = np.array([0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.255555555556 ])
discFac = np.array([0.998789615268, 0.999800018888, 1.02, 1.04,  1.08, 1.12, 1.13, 1.16, 1.20, 1.24, 1.25, 1.27, 1.303])
pi = np.array([0.0, 0.0,0.00163277007592  ,0.00383756939696   ,0.00601361347919   ,0.00816106960817  , 0.010351451799  , 0.0125369967385  ,0.014694037557   ,0.0168227400791  ,0.0189939937924    ,0.0211604524977   ,0.0233509971123 ])


C = 0.03
D = 0.06 
marketPrice = 64336.0
#marketPrice = 229269.0
#marketPrice = 10265.0
nom = 1000000.0
sumDelta = np.cumsum(delta)
#a = pi[-1]/sum(delta)# 0.00379778046231# probability to default the first time step #0.29069705 # probability of default before maturity 
a = 0.002#0.0233509971123/sum(delta)#0.002#0.012/0.25
#print a
#assume all issuers the same intensity, so that lambda i = lambda
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