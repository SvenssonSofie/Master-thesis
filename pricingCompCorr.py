######### Price the CDS given the compound correlation for all tranches

import math
import numpy as np
import scipy.stats as sct
import time
start_time = time.time()

corr = [0.73, 0.86, 0.41]

tranches = [0, 1, 2] #all tranches that should be prices
tr = np.array([0.0, 0.03, 0.06, 0.12])#tranches


delta = np.array([0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.255555555556 ])
discFac = np.array([0.998789615268, 0.999800018888, 1.02, 1.04,  1.08, 1.12, 1.13, 1.16, 1.20, 1.24, 1.25, 1.27, 1.303])
pi = np.array([0.0, 0.0,0.00163277007592  ,0.00383756939696   ,0.00601361347919   ,0.00816106960817  , 0.010351451799  , 0.0125369967385  ,0.014694037557   ,0.0168227400791  ,0.0189939937924    ,0.0211604524977   ,0.0233509971123 ])

coupons = np.array([0.03, 0.02, 0.03])#coupons we want to price for

#coupon = 0.05
nIss = 125#Number of issuers 

#Pricing of derivatives
recRate = 0.0
nTime = len(delta)#nbr of time points
nom = 1000000.0

C = math.ceil(tr[0]*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(tr[1]*nIss)) #nbr of losses detachment
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    
def defaultLeg():
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
        
        #discount sum of Payments
        pvDl = pvDl + sumPay/discFac[timeStep]
        #print pvDl
    return pvDl

def premiumLeg(coupon): 
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

def findp(corr):
    probTot = 0.0
    intPnts = np.linspace(-6,6) #50 points by default between -6 and 6
    global p
    p = np.zeros((nIss+1, nTime)) 
    for Y0 in intPnts: 
        probTot = probTot + sct.norm.pdf(Y0)
        p = p + np.multiply(conProb(Y0, corr), sct.norm.pdf(Y0))
    p = np.divide(p,probTot)
    
    
def conProb(Y0, corr):# conditional probability of k defaults given Y0 
    piCond = np.zeros(nTime)
    denom = math.sqrt(1-math.pow(corr,2))
    for timeStep in range(0,nTime):   
        nume = sct.norm.ppf(pi[timeStep]) - corr*Y0
        piCond[timeStep] = sct.norm.cdf(nume/denom)#conditional probability for each issuer(one row each) and each time step(one column each)
    #Probabilities of number of defaults for all time steps pk 
    q = np.zeros((nIss+1, nTime)) #prob of number of defaults (rows = nbr of issuers +1 (zero defaults)) and time step (columns)
    for timeStep in range(0, nTime): #for each time step we want a probability of number of defaults 
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

price = np.zeros(len(tranches))
####################main



for k in tranches: 
    
    C = math.ceil(tr[k]*nIss) #nbr of losses attachment
    if C != 0: #if we want to insure 12-15, C should be 11
        C = C-1
    D = float(math.trunc(tr[k+1]*nIss)) #nbr of losses detachment
    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    
    findp(corr[k]) #correlation for 0-3 find probability of k defaults, changes global variable 
    price[k] = defaultLeg() - premiumLeg(coupons[k])  #spread 3-6 for finding correlation 0-6
    
print price
print("--- %s seconds ---" % (time.time() - start_time))




C = math.ceil(0.01*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(0.03*nIss)) #nbr of losses detachment
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    
findp(0.43126441) #correlation for 0-3 find probability of k defaults, changes global variable 
print 'price base corr 1-3:', defaultLeg() - premiumLeg(0.00285129716295)  #spread 3-6 for finding correlation 0-6