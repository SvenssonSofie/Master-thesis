import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt
import warnings

warnings.simplefilter(action = "ignore", category = RuntimeWarning)
start_time = time.time()

##########Calculates contagion ration for C-D, do this for all tranches and put them into pricingContMod
nIss = 125 #Number of issuers
delta = np.array([0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.255555555556 ])
discFac = np.array([0.998789615268, 0.999800018888, 1.02, 1.04,  1.08, 1.12, 1.13, 1.16, 1.20, 1.24, 1.25, 1.27, 1.303])
pi = np.array([0.0, 0.0,0.00163277007592  ,0.00383756939696   ,0.00601361347919   ,0.00816106960817  , 0.010351451799  , 0.0125369967385  ,0.014694037557   ,0.0168227400791  ,0.0189939937924    ,0.0211604524977   ,0.0233509971123 ])
sumDelta = np.cumsum(delta)
tr = np.array([0.0, 0.03, 0.06, 0.12])
pr = np.array([229269.0, 63815.0, 10265.0])#prices on market 

nom = 1000000.0
sumDelta = np.cumsum(delta)
a = 0.002# 0.0233509971123/sum(delta)#0.002#0.012/0.25
T = len(delta)
coupon = 0.01
recRate = 0.0

nbrSim = 1
uniAll = np.random.uniform(size = (nbrSim, nIss))#to optimize over the same random numbers, common random numbers

#Generate correlated uniformed random variables 
normRand1 = np.random.normal(size = (nbrSim, nIss))
normRand2 = np.random.normal(size = (nbrSim, nIss))

uniAll = np.random.uniform(size = (nbrSim, nIss))#to optimize over the same random numbers, common random numbers

defTime = np.zeros(nIss)
#Lavg = 1000000.0/12.0
def f(tau, Ek, c, d, k): #Want to solve F(tau) = 0
    
    if tau < defTime[k-1]:
        return float('inf')

    ftau = 0.0
    for i in range(k):
        ftau = ftau + 1-np.exp(-d*(tau - defTime[i]))
    
    ftau = ftau*a*c/d + a*tau
    return abs(ftau - Ek)


def parameters(p):     
    return (contagion(p, tr[0], tr[1], pr[0]), contagion(p, tr[1], tr[2], pr[1]), contagion(p, tr[2], tr[3], pr[2]))


def contagion(p, C, D, marketPrice):
    conRa = p[0]
    decay = p[1]
    rho = p[2]
    
    if abs(rho) >=  1.0:
        rho = 0.99
    if rho < 0: 
        rho = abs(rho)
    if decay < 0.0: 
        decay = abs(decay)     
    
    # From percentage to number of defaults
    C = math.ceil(C*nIss)
    if (C != 0):
        C = C-1
    D = float(math.trunc(D*nIss)) #nbr of losses detachment
    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    print p, C, D, Lavg
    
    pvMean = np.zeros((2,nbrSim))
    for sim in range(0,nbrSim):

        # Generate correlated uniform variables 
        normRand = rho*normRand1[sim,:] + math.sqrt(1-rho**2)*normRand2[sim,:]
        randCDF = sct.norm.cdf(normRand)
        uniRand = sct.uniform.ppf(randCDF)
        e = -np.log(uniRand)
        eSort = np.sort(e)

        defTime[0] = eSort[0]/a #defualt times in vector defTime
        for nbr in range(1,nIss):#len(uniRand)):
            Ek = eSort[nbr]
            
            # want to find that tau that solves the equation F(t) = 0
            defTime[nbr] = opt.fmin_cobyla(f,defTime[nbr -1], (), args = (Ek, conRa, decay, nbr), disp = False)
        print defTime
        
        #find loss for each time step 
        loss = np.zeros(T)
        for t in range(0,T):
            if sumDelta[t] < defTime[0]: continue
            for tau in range(1,len(defTime)):
                #print tau
                if defTime[tau] > sumDelta[t] and defTime[tau - 1] < sumDelta[t]: 
                    loss[t] = tau#0/nIss
                    break
                else: 
                    loss[t] = 125.0
        #print loss   
        
        ################ tranche loss in each time point
        lossTr = np.zeros(T)
        for t in range(0,T):
            if loss[t] <= C: continue
            elif loss[t] > C and loss[t] <= D: 
                lossTr[t] = (loss[t] - C)*Lavg
            else: 
                lossTr[t] = (D-C)*Lavg
        print lossTr #
        
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
    print pvMean[0] - pvMean[1]
    return abs((pvMean[0] - pvMean[1]) - marketPrice)

#contRatio = contagion(0.3, 0.3, 0.3)
c = opt.fsolve(parameters, (0.05,0.03,0.5))
print 'True ratio: ', c
print("--- %s seconds ---" % (time.time() - start_time))