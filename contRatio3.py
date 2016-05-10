import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt
import warnings
#from mahotas.demos.surf_gaussians import rho
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

start_time = time.time()

##########Calculates contagion ration for C-D, do this for all tranches and put them into pricingContMod
nIss = 125.0#Number of issuers
discFac = np.array([0.998567685447      ,0.999577864556          ,  1.00045062864              , 1.00045062864             ,\
          1.00178528504           ,  1.00250401027       , 1.00317902449       , 1.0036796674         ,\
           1.00376829329        , 1.00365008529       , 1.00363960304       , 1.00370342137        ,\
          1.00375936033       ,  1.00380409316        , 1.00361392274        ,1.00320197241        , \
          1.0026924065        ,1.00208149172       , 1.00119465201       ,   1.00011221868  ,0.998900782422])
pi = np.array([0.0, 0.0, 0.0042460839466,0.00429791106969, 0.00430942354938, 0.0043144870735, \
      0.00431740949283, 0.0043192794268, 0.00432056792291, 0.00432150925416, 0.00432224686705,\
     0.00432282896438, 0.00432329565995, 0.00486536500626, 0.0053303458054,  0.00572533837007, \
     0.00606158559488, 0.00635421564115, 0.00661908726913,0.00684552159625, 0.00704935047305])


delta = np.array([0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778,      \
         0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, \
              0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
              0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778])
sumDelta = np.cumsum(delta)

C = 0.03
D = 0.06 
marketPrice = 64336.0
#marketPrice = 229269.0
nom = 1000000.0
sumDelta = np.cumsum(delta)
#a = pi[-1]/sum(delta)# 0.00379778046231# probability to default the first time step #0.29069705 # probability of default before maturity 
a = 0.0233509971123/sum(delta)#0.002#0.012/0.25
#print a
#assume all issuers the same intensity, so that lambda i = lambda
T = len(delta)
#generate 50 uniform standard random values
nbrSim = 1
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

#Generate correlated uniformed random variables 
normRand1 = np.random.normal(size = (nbrSim, nIss))
normRand2 = np.random.normal(size = (nbrSim, nIss))

uniAll = np.random.uniform(size = (nbrSim, nIss))#to optimize over the same random numbers, common random numbers

defTime = np.zeros(nIss)

def f(tau, Ek, c, d, k, prevTau):
    #print Ek
    if tau < prevTau: 
        return float('inf')
    #print k, prevTau, c, d
    ftau = 0.0
    for i in range(k):
        ftau = ftau + a*c/d*(1-np.exp(-d*(tau - defTime[i])))
        #print ftau
    ftau = ftau + a*prevTau
    
    #print ftau - Ek
    return ftau - Ek

def contagion(optArg):
    c = optArg[0] 
    d = optArg[1]
    rho = optArg[2]
    
    print optArg
    
    if rho < 0 or rho >= 1 or c < 0 or d < 0: 
        return float('inf')
    
    pvMean = np.zeros((2,nbrSim))
    for sim in range(0,nbrSim):
        #uni = uniAll[sim,:]
        # Generate correlated uniform variables
        normRand = rho*normRand1[sim,:] + math.sqrt(1 - rho**2)*normRand2[sim,:]
        randCDF = sct.norm.cdf(normRand)
        uniRand = sct.uniform.ppf(randCDF)
        e = -np.log(uniRand)
        eSort = np.sort(e)
                
        
        defTime[0] = eSort[0]/a
        for nbr in range(1,len(uniRand)):
            Ek = eSort[nbr]
            
            #print Ek
            defTime[nbr] = opt.fmin(f,defTime[nbr -1], args = (Ek, c, d, nbr, defTime[nbr -1]), disp = False)#, maxfun = 10000000, maxiter = 1000000)# det Tau som uppfyller F(tau) = 0 for detta d
            #print defTime[nbr]
            #print("--- %s seconds ---" % (time.time() - start_time))
            #ar nog ganska fel here, vill optimera over det tau som 

        #print defTime
        
        loss = np.zeros(T)#if no value is inserted on a position, then all has defaulted before that timeStep and thus should the loss of the following time steps be 100%
        for t in range(0,T):
            if sumDelta[t] < defTime[0]: continue
            for tau in range(1,len(defTime)):
                #print tau
                if defTime[tau] > sumDelta[t] and defTime[tau - 1] < sumDelta[t]: 
                    loss[t] = tau#0/nIss
                    break
                else: 
                    loss[t] = 125.0    
        
        ################ tranche loss
        lossTr = np.zeros(T) #add rows to calculate tranches simultaneously
        for t in range(0,T):
            if loss[t] <= C: continue
            elif loss[t] > C and loss[t] <= D: 
                lossTr[t] = (loss[t] - C)*Lavg
            else: 
                lossTr[t] = (D-C)*Lavg
        #print lossTr #
        
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
        
        pvMean[0, sim] = pvMean[0, sim - 1] + pvDl
        pvMean[1, sim] = pvMean[1, sim - 1] + pvPl
        
    
    pvMean = np.mean(pvMean,1)/nbrSim
    
    print 'Present value, PVdl, PVpl: ', pvMean  
    print (pvMean[0] - pvMean[1]) - marketPrice  
    return abs((pvMean[0] - pvMean[1]) - marketPrice)

#contRatio = contagion(0.3, 0.3, 0.3)
c = opt.fmin_powell(contagion, [0.03, 0.03, 0.05])
print 'True ratio: ', c
print("--- %s seconds ---" % (time.time() - start_time))