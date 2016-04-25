import math
import numpy as np
import scipy.stats as sct
import time
import scipy.optimize as opt
start_time = time.time()

corr = [0.733338844, 0.82111647, 0.90891503]
tranches = [0, 1, 2] #all tranches that should be prices first is 3-6
tr = np.array([0.0, 0.03, 0.06, 0.12])#tranches

# check if all correlations give same value for linear CDS 
#corr = [0.99]
#tranches =[0]
#tr = [0.0, 1.0]

delta = np.array([0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.255555555556 ])
discFac = np.array([0.998789615268, 0.999800018888, 1.02, 1.04,  1.08, 1.12, 1.13, 1.16, 1.20, 1.24, 1.25, 1.27, 1.303])
pi = np.array([0.0, 0.0,0.00163277007592  ,0.00383756939696   ,0.00601361347919   ,0.00816106960817  , 0.010351451799  , 0.0125369967385  ,0.014694037557   ,0.0168227400791  ,0.0189939937924    ,0.0211604524977   ,0.0233509971123 ])

coupons = np.array([0.03, 0.03, 0.03])#coupons we want to price for
nIss = 125#Number of issuers 

#Pricing of derivatives
recRate = 0.0
nTime = len(delta)#nbr of time points
nom = 1000000.0

C = 0.0
D = 0.0
Lavg = 0.0
    
def defaultLeg(p):

    print C, D, Lavg
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

def premiumLeg(p, coupon): 
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
    return p
    
    
def conProb(Y0, corr):# conditional probability of k defaults given Y0 
    piCond = np.zeros(nTime)
    denom = math.sqrt(1-corr**2)
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





def correlation(corr, coupon): 
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
    return abs(pvDl - pvNl) 







price = np.zeros(len(tranches))
####################main

# Price 0-3
scaleLower = 1.0
scaleUpper = 1.0
for k in tranches: #k should be > 0 since the first tranche is the same as for compound correlation k = 1 corresponds to tranche 3-6

    D = float(math.trunc(tr[k+1]*nIss)) #nbr of losses detachment

    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    scaleUpper = tr[k+1]/(tr[k+1]-tr[k])
    scaleLower = tr[k]/(tr[k+1] - tr[k])
    #upper limit
    p = findp(corr[k])
    priceUpper = defaultLeg(p) - premiumLeg(p, coupons[k])#*scaleUpper
    
    #dlLow = 0
    if k != 0: 
        #lower limit
        p = findp(corr[k-1]) #correlation for 0-3 find probability of k defaults, changes global variable
        priceLower = defaultLeg(p) - premiumLeg(p, coupons[k])#*scaleLower
        #dlLow = defaultLeg()
       
        print scaleUpper, scaleLower
    else: 
        priceLower = 0
    print priceLower, priceUpper
    #price[k] = priceUpper - priceLower
    price[k] = (scaleUpper*priceUpper - scaleLower*priceLower) #spread 3-6 for finding correlation 0-6
    #price[k] = scaleUpper - scaleLower
    #scaleUpper = tr[k+1]/(tr[k+1]-tr[k])
    #scaleLower = tr[k]/(tr[k+1] - tr[k])
    #findp(corr[k-1]) #correlation for 0-3 find probability of k defaults, changes global variable
    #dlLow = defaultLeg()
    #priceLower = (dlLow - premiumLeg(coupons[k]))*scaleLower
    #findp(corr[k]) #correlation for 0-3 find probability of k defaults, changes global variable
    #dlUpp = defaultLeg()
    #priceUpper = (defaultLeg() - premiumLeg(coupons[k]))*scaleUpper
    
    #print 'pricelower: ', priceLower, 'priceupper: ', priceUpper, 'dllow: ', dlLow, 'dlUpp: ', dlUpp
    
    #sprUpp = opt.fmin_cobyla(parRate, [0.5], (), args=([corr[k]]), rhoend = 0.00001)
    #sprLow = opt.fmin_cobyla(parRate, [0.5], (), args=([corr[k-1]]), rhoend = 0.00001)
    
    #dviUpp = dlUpp/sprUpp
    #dviLow = dlLow/sprLow
    #print 'dviUpp', dviUpp, 'dviLow', dviLow
    #dvi = dviUpp*scaleUpper - dviLow*scaleLower
    #pvdl = dlUpp*scaleUpper - dlLow*scaleLower
    
    #spread = pvdl/dvi
    #print 'dvi ', dvi, 'pvdl ', pvdl, 'spread', spread
    
    #c = opt.fmin_cobyla(correlation, [0.5], (), args=([spread]), rhoend = 0.00001)
    
    #findp(c)
    #price[k] = defaultLeg() - premiumLeg(coupons[k])
    
    
print 'price ', price
print("--- %s seconds ---" % (time.time() - start_time))