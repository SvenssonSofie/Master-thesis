import math
import numpy as np
import scipy.stats as sct
import scipy.optimize as opt
import time
import warnings
import sys
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
start_time = time.time()

corr = [0.733338844, 0.82111647, 0.90891503]

tranches = [1, 2, 3] #all tranches that should be prices first is 3-6
tr = np.array([0.0, 0.03, 0.06, 0.12])#tranches


delta = np.array([0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.255555555556 ])
discFac = np.array([0.998789615268, 0.999800018888, 1.02, 1.04,  1.08, 1.12, 1.13, 1.16, 1.20, 1.24, 1.25, 1.27, 1.303])
pi = np.array([0.0, 0.0,0.00163277007592  ,0.00383756939696   ,0.00601361347919   ,0.00816106960817  , 0.010351451799  , 0.0125369967385  ,0.014694037557   ,0.0168227400791  ,0.0189939937924    ,0.0211604524977   ,0.0233509971123 ])

coupon = 0.01#coupon we want to price for

#coupon = 0.05
nIss = 125#Number of issuers 

#Pricing of derivatives
recRate = 0.0
nTime = len(delta)#nbr of time points
nom = 1000000.0
#Set C and D to attachment and detachment respectively
c = 0.025
d = 0.03
C = math.ceil(c*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(d*nIss)) #nbr of losses detachment
try: 
    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
except ZeroDivisionError: 
    print 'The interval contains no defaults, use other attachment/detachment'
    sys.exit()
    
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


def impliedSpread(price, corr):
    return opt.fmin_cobyla(optSpread, [0.5], [cons1], args=(corr,price), consargs = ())#, maxfun = 100000000, maxiter = 1000000000)

def cons1(corr):
    return 1 - abs(corr)

def optSpread(spread, corr, kPrice):
    findp(corr)
    pvDl = defaultLeg()
    pvPl = premiumLeg(spread)    
    return abs(pvDl - pvPl - kPrice)


####################main

#Find which to interpolate with for detachment, if C = 0.07 indUpp = 9 indLow = 6
indUppAtt = min(np.where(tr >= c)[0]) #index of the first tranche detachment greater than C
indLowAtt = max(np.where(tr <= c)[0])

#becomes wrong ig indLowAtt = 0
if indUppAtt == 0:
    bCorrAtt = 0.0
elif indLowAtt == 0: #meaning first tranche  
    bCorrAtt = corr[0]
elif indLowAtt == indUppAtt: 
    bCorrAtt = corr[indUppAtt - 1]
else:
    bCorrAtt = (tr[indUppAtt] - c)/(tr[indUppAtt]-tr[indLowAtt]) * corr[indLowAtt-1] + (c - tr[indLowAtt])/(tr[indUppAtt]-tr[indLowAtt]) * corr[indUppAtt-1] #combination of 0-6 and 0-9


indUppDet = min(np.where(tr >= d)[0]) #index of the first tranche detachment greater than D
indLowDet = max(np.where(tr <= d)[0])
print 'd', d, 'induppdet', indUppDet, 'indlowdet', indLowDet, 'corr: ', corr

if indLowDet == 0: 
    bCorrDet = corr[0]
elif indUppDet == indLowDet: 
    bCorrDet = corr[indUppDet -1]
else:    
    bCorrDet = (tr[indUppDet] - d)/(tr[indUppDet]-tr[indLowDet]) * corr[indLowDet-1] + (d - tr[indLowDet])/(tr[indUppDet]-tr[indLowDet]) * corr[indUppDet-1] #combination of 0-6 and 0-9
print bCorrAtt, bCorrDet

findp(bCorrAtt)
pvDlAtt = defaultLeg()
priceAtt = - premiumLeg(coupon) + pvDlAtt#0-5
findp(bCorrDet)
pvDlDet = defaultLeg()
priceDet = - premiumLeg(coupon) + pvDlDet#0-8
#priceAttDet = (d - c)/d * priceDet - (d-c)/c * priceAtt #5-8
priceAttDet = d/(d-c) * priceDet - c/(d-c) * priceAtt #5-8
print 'price: ' , priceAttDet, 'pricedet: ', priceDet, 'priceAtt: ', priceAtt

### Assume spreads are known 0-3 3-6 6-9 9-12, spreads for 0-3 0-6 0-9 0-12 are known from 
### Interpolate spreads

'''
spreadAttUpp = 0.04266875#impliedSpread(0, corr[indUppAtt -1]) # 0-6

spreadDetUpp = 0.2509063#impliedSpread(0, corr[indUppDet -1])#0-12

if indUppAtt == 0:
    spreadAtt = 0.0
elif indLowAtt == 0: #meaning first tranche  
    spreadAtt = spreadAttUpp
elif indLowAtt == indUppAtt: 
    spreadAtt = spreadAttUpp
else:
    spreadAttLow = 0.04266875#impliedSpread(0, corr[indLowAtt -1])#0-3
    spreadAtt = (tr[indUppAtt] - c)/(tr[indUppAtt]-tr[indLowAtt]) * spreadAttLow + (c - tr[indLowAtt])/(tr[indUppAtt]-tr[indLowAtt]) * spreadAttUpp

if indUppDet == 0:
    spreadDet = 0.0
elif indLowDet == 0: #meaning first tranche  
    spreadDet = spreadDetUpp
elif indLowDet == indUppDet: 
    spreadDet = spreadDetUpp
else:
    spreadDetLow = 0.03495625#impliedSpread(0, corr[indLowDet -1])#0-9
    spreadDet = (tr[indUppDet] - d)/(tr[indUppDet]-tr[indLowDet]) * spreadDetLow + (d - tr[indLowDet])/(tr[indUppDet]-tr[indLowDet]) * spreadDetUpp

print spreadAtt, spreadDet
DVatt = pvDlAtt/spreadAtt
DVdet = pvDlDet/spreadDet

#rescale pvDL avd risky DV01
pvDlAttDet = d/(d-c) * pvDlDet - c/(d-c) * pvDlAtt
DVattDet = d/(d-c) * DVdet - c/(d-c) * DVatt

spreadAttDet = pvDlAttDet/DVattDet




print 'Spread for', c, 'to', d, ': ', spreadAttDet*10000.0, 'bps' '''
print("--- %s seconds ---" % (time.time() - start_time))
