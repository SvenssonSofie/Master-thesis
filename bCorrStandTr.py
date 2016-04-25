import math
import numpy as np
import scipy.stats as sct
import scipy.optimize as opt
import time
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

################################Finding correlation for 0-detachment for all standard tranches
start_time = time.time()


# iTrx Europe 24, 3Y, 3-6%
delta = np.array([0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.255555555556 ])
#discFac = np.array([0.998789615268, 0.999800018888, 1.00057366592, 1.00123214037,  1.00194107616, 1.00269593306, 1.0034370463, 1.00386099671, 1.00395019207, 1.00392703588, 1.004093509, 1.00426863106, 1.00449749648])
discFac = np.array([0.998789615268, 0.999800018888, 1.02, 1.04,  1.08, 1.12, 1.13, 1.16, 1.20, 1.24, 1.25, 1.27, 1.303])
pi = np.array([0.0, 0.0,0.00163277007592  ,0.00383756939696   ,0.00601361347919   ,0.00816106960817  , 0.010351451799  , 0.0125369967385  ,0.014694037557   ,0.0168227400791  ,0.0189939937924    ,0.0211604524977   ,0.0233509971123 ])
tr = np.array([0.0, 0.03, 0.06, 0.12])
spr = np.array([0.01, 0.01, 0.01])
parSpr = np.array([0.005191, 0.005191, 0.005191])
pr = np.array([229269.0, 63815.0, 10265.0])#prices on market 
comp = [0.73338844, 0.8554282, 0.41002418]
#Pricing of derivatives
recRate = 0.0
nTime = len(delta)
nom = 1000000.0
nIss = 125 #Number of issuers

#C = math.ceil(tr[0]*nIss) #nbr of losses attachment
#if C != 0: #if we want to insure 12-15, C should be 11
#    C = C-1
C= 0.0
D = float(math.trunc(tr[1]*nIss)) #nbr of losses detachment
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D

def compoundCorr(kPrice, spread):
    return opt.fmin_cobyla(optCorr, [0.5], (), args=([kPrice],[spread]), rhoend = 0.00001)#,  consargs = (), rhoend = 0.001)#, maxfun = 100000000, maxiter = 1000000000)

def optCorr(corr, kPrice, spread):
    if corr >= 1.0 or corr < 0:
        return float('inf')
    p = findp(corr)
    pvDl = defaultLeg(p)
    pvPl = premiumLeg(spread, p) 
    return abs(pvDl - pvPl - kPrice) 
    
def conProb(Y0, corr):# conditional probability of k defaults given Y0 
    piCond = np.zeros(nTime)
    denom = math.sqrt(1-math.pow(corr,2))
    for timeStep in range(0,nTime):   
        nume = sct.norm.ppf(pi[timeStep]) - corr*Y0
        piCond[timeStep] = sct.norm.cdf(nume/denom)#conditional probability for each issuer(one row each) and each time step(one column each)
    #Probabilities of number of defaults for all time steps pk 
    q = np.zeros((nIss+1, nTime)) #prob of number of defaults (rows = nbr of issuers +1 (zero defaults)) and time step (columns)
    for timeStep in range(0, nTime): #for each time step, a probability of number of defaults 
        q[0, timeStep] = 1.0     
        alpha = piCond[timeStep]
        for issuer in range(0, nIss): #from 0 to 49, add one at the time
            for nbrDef in range(0, issuer+1): #update all that are added so far
                if nbrDef == 0:
                    qOld = q[nbrDef, timeStep]
                q[nbrDef, timeStep] = q[nbrDef, timeStep] - qOld*alpha
                temp = q[nbrDef + 1, timeStep] # save this to next step since it is needed for updating after overwritten
                q[nbrDef + 1, timeStep] = q[nbrDef + 1, timeStep] + qOld*alpha
                qOld = temp            
    return q

def defaultLeg(p):
    print C, D
    pvDl = 0.0
    for timeStep in range(0,nTime):
        sumPay = 0.0
        for nbrDef in range(0,nIss+1):#0-50            
            if nbrDef <= C:
                Dl = 0.0
            elif nbrDef > C and nbrDef <= D: 
                Dl = Lavg*(nbrDef-C)
            else:
                Dl = Lavg*(D-C)              
                  
            if timeStep == 0:
                sumPay = sumPay + (p[nbrDef, timeStep])*Dl #when timestep zero, 1y from now, p[nbrDef, timestep-1] = 0 
            else: 
                sumPay = sumPay + (p[nbrDef, timeStep] - p[nbrDef, timeStep-1])*Dl
        
        #discount sum of Payments
        pvDl = pvDl + sumPay/discFac[timeStep]
    return pvDl


def premiumLeg(coupon, p): 
    if type(coupon) is list:
        coupon = float(coupon[0])
    pvNl = 0.0
    for timeStep in range(0,nTime):
        sumPay = 0.0
        for nbrDef in range(0,nIss+1):            
            
            if nbrDef <= C:
                Nl = (D-C)*Lavg
            elif nbrDef > C and nbrDef <= D: 
                Nl = Lavg*(D-nbrDef)
            else: 
                Nl = 0
            sumPay = sumPay + p[nbrDef, timeStep]*Nl
        #discount sum of Payments
        pvNl = pvNl + coupon*delta[timeStep]*sumPay/discFac[timeStep]
    return pvNl    

############## Generate p 
def findp(corr):
    probTot = 0.0
    intPnts = np.linspace(-6,6) #50 points by default between -6 and 6
    p = np.zeros((nIss+1, nTime)) 
    for Y0 in intPnts: 
        probTot = probTot + sct.norm.pdf(Y0)
        p = p + np.multiply(conProb(Y0, corr), sct.norm.pdf(Y0))
    p = np.divide(p,probTot)
    return p




def parRate(coupon, corr): 
    print coupon
    if coupon < 0: 
        return float('inf')
    probTot = 0.0
    intPnts = np.linspace(-6,6) #50 points by default between -6 and 6
    p = np.zeros((nIss+1, nTime)) 
    for Y0 in intPnts: 
        probTot = probTot + sct.norm.pdf(Y0)
        p = p + np.multiply(conProb(Y0, corr), sct.norm.pdf(Y0))
    p = np.divide(p,probTot)
    #print p 

    pvDl = 0.0
    for timeStep in range(0,nTime): #0 to 4
        sumPay = 0.0
        for nbrDef in range(0,nIss+1):#0-50            
            if nbrDef <= C:
                Dl = 0.0
            elif nbrDef > C and nbrDef <= D: 
                Dl = Lavg*(nbrDef-C)
            else:
                Dl = Lavg*(D-C)              
                  
            if timeStep == 0:
                sumPay = sumPay + (p[nbrDef, timeStep])*Dl #when timestep zero, 1y from now, p[nbrDef, timestep-1] = 0 
            else: 
                sumPay = sumPay + (p[nbrDef, timeStep] - p[nbrDef, timeStep-1])*Dl
        
        pvDl = pvDl + sumPay/discFac[timeStep] #discount recieved default payments
    print 'pvDl', pvDl
    
    #present value of payments to insurer
    pvNl = 0.0
    for timeStep in range(0,nTime):
        sumPay = 0.0
        for nbrDef in range(0,nIss+1):    
            
            if nbrDef <= C:
                Nl = (D-C)*Lavg
            elif nbrDef > C and nbrDef <= D: 
                Nl = Lavg*(D-nbrDef)
            else: 
                Nl = 0
            sumPay = sumPay + p[nbrDef, timeStep]*Nl

        #discount sum of Payment
        pvNl = pvNl + coupon*delta[timeStep]*sumPay/discFac[timeStep]
    print 'pvNl',  - pvNl    
    print 'PV', pvDl - pvNl
    
    #Recieve default payments, pay premiums
    return abs(pvDl - pvNl) #compound, returns the difference that we want to be zero, finds optimal correlation





def correlation(corr, coupon, marketPrice): #coupon from pricing in PRIME
    print corr
    if corr >= 1 or corr < 0: 
        return float('inf')
    probTot = 0.0
    intPnts = np.linspace(-6,6) #50 points by default between -6 and 6
    p = np.zeros((nIss+1, nTime)) 
    for Y0 in intPnts: 
        probTot = probTot + sct.norm.pdf(Y0)
        p = p + np.multiply(conProb(Y0, corr), sct.norm.pdf(Y0))
    p = np.divide(p,probTot)
    #print p 

    pvDl = 0.0
    for timeStep in range(0,nTime): #0 to 4
        sumPay = 0.0
        for nbrDef in range(0,nIss+1):#0-50            
            if nbrDef <= C:
                Dl = 0.0
            elif nbrDef > C and nbrDef <= D: 
                Dl = Lavg*(nbrDef-C)
            else:
                Dl = Lavg*(D-C)              
                  
            if timeStep == 0:
                sumPay = sumPay + (p[nbrDef, timeStep])*Dl #when timestep zero, 1y from now, p[nbrDef, timestep-1] = 0 
            else: 
                sumPay = sumPay + (p[nbrDef, timeStep] - p[nbrDef, timeStep-1])*Dl
        
        pvDl = pvDl + sumPay/discFac[timeStep] #discount recieved default payments
    print 'pvDl', pvDl
    
    #present value of payments to insurer
    pvNl = 0.0
    for timeStep in range(0,nTime):
        sumPay = 0.0
        for nbrDef in range(0,nIss+1):    
            
            if nbrDef <= C:
                Nl = (D-C)*Lavg
            elif nbrDef > C and nbrDef <= D: 
                Nl = Lavg*(D-nbrDef)
            else: 
                Nl = 0
            sumPay = sumPay + p[nbrDef, timeStep]*Nl

        #discount sum of Payment
        pvNl = pvNl + coupon*delta[timeStep]*sumPay/discFac[timeStep]
    print 'pvNl',  - pvNl    
    print 'PV', pvDl - pvNl
    
    #Recieve default payments, pay premiums
    return abs(pvDl - pvNl - marketPrice) #compound, returns the difference that we want to be zero, finds optimal correlation







############# correlation for standard tranches
corr = np.zeros(len(tr)-1)
corr[0] = comp[0]# just for running faster in test case compoundCorr(pr[0], spr[0]), compound correlation
print corr
for k in range(1,len(tr)-1): #find base correlation for all tranches
    
    D = float(math.trunc(tr[k+1]*nIss)) #nbr of losses detachment
    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    corrTran = comp[k] # compound correlation opt.fmin_cobyla(correlation, [0.5], (), args=([0.01, pr[k]]), rhoend = 0.00001)
    print corrTran
    
    C = math.ceil(tr[k]*nIss) #For market premium we need tranche with other attachment than zero
    if C != 0:
        C = C-1
    marketPrem = opt.fmin_cobyla(parRate, [0.1], (), args=([corrTran]), rhoend = 0.001)#from correlation find price = 0
    C = 0
    
    scale = tr[k]/tr[k+1] #3/6 for first
    p = findp(corr[k-1]) #correlation for 0-3 find probability of k defaults, changes global variable
    kPrice = (defaultLeg(p) - premiumLeg(marketPrem, p))*scale #spread 3-6 for finding correlation 0-6
    print kPrice, marketPrem
    corr[k] = compoundCorr(kPrice, marketPrem)
    print corr
print("--- %s seconds ---" % (time.time() - start_time))


