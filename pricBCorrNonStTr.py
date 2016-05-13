import math
import numpy as np
import scipy.stats as sct
import scipy.optimize as opt
import time
import warnings
import sys
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
start_time = time.time()



# S24 main 3Y
pi = np.array([0.00087479504671772457, 0.0029353834489070918, 0.004969393415826806, 0.0069769703635795821, 0.0090249736847164419, 0.011068753219014127, 0.013086171126700852, 0.015077371637674086, 0.017108668751384948, 0.019135776533014792, 0.021188431563949561])
spread = np.array([0.0047228532506136903, 0.0048101666897295524, 0.0048253362059923867, 0.0048316350692150416, 0.0048351736249719465, 0.0048374002858278193, 0.0048389188662819015, 0.0048400205026313961, 0.0048408789399059949, 0.0048415530671146135, 0.0048489188205022606])
discFac = np.array([1.0003661802496806, 1.0010318855144935, 1.0017322165435656, 1.0024681809694533, 1.0031665273659054, 1.0037827350418536, 1.0039805436318214, 1.0039247441142243, 1.0039599948710676, 1.0041599686436489, 1.0043651935486468])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25555555555555554])


nIss = 125
coupon = 0.05


recRate2 = 0.4
tr = np.array([0.0, 0.03, 0.06, 0.12, 1.0])/(1-recRate2)
for t in range(0,len(tr)): 
    tr[t] = float(math.trunc(tr[t]*nIss))
print tr

corr = [ 0.68061913,  0.76454688 , 0.85944655 , 0.94466163]





#Pricing of derivatives
recRate = 0.0
nTime = len(delta)#nbr of time points
nom = 1000000.0
#Set C and D to attachment and detachment respectively
c = 0.1
d = 0.30

recRate2 = 0.4
c = c/(1-recRate2)
d= d/(1-recRate2)


C = math.ceil(c*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(d*nIss)) #nbr of losses detachment
c = C
d = D
try: 
    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
except ZeroDivisionError: 
    print 'The interval contains no defaults, use other attachment/detachment'
    sys.exit()
    
    
print C, D
def defaultLeg(p):
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

def findp(corr):
    a = -6.0
    b = 6.0
    deg = 50
    x, w = np.polynomial.legendre.leggauss(deg)
    w = w*(b-a)*0.5 #weights
    t = 0.5*x*(b - a) #sample points
     
    p = np.zeros((nIss+1, nTime))
    for k in range(0,len(t)):
        p = p + np.multiply(w[k]*sct.norm.pdf(t[k]), conProb(t[k], corr))   
    return p 
    
    
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
    p = findp(corr)
    pvDl = defaultLeg(p)
    pvPl = premiumLeg(spread, p)    
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
    #bCorrAtt = (tr[indUppAtt] - c)/(tr[indUppAtt]-tr[indLowAtt]) * corr[indLowAtt-1] + (c - tr[indLowAtt])/(tr[indUppAtt]-tr[indLowAtt]) * corr[indUppAtt-1] #combination of 0-6 and 0-9
    bCorrAtt = (tr[indUppAtt] - c)/(tr[indUppAtt]-tr[indLowAtt]) * corr[indLowAtt-1] + (c - tr[indLowAtt])/(tr[indUppAtt]-tr[indLowAtt]) * corr[indUppAtt-1] #combination of 0-6 and 0-9


indUppDet = min(np.where(tr >= d)[0]) #index of the first tranche detachment greater than D
indLowDet = max(np.where(tr <= d)[0])
print 'd', d, 'induppdet', indUppDet, 'indlowdet', indLowDet, 'corr: ', corr
print 'c', c, 'induppattt', indUppAtt, 'indlowatt', indLowAtt, 'corr: ', bCorrAtt

if indLowDet == 0: 
    bCorrDet = corr[0]
elif indUppDet == indLowDet: 
    bCorrDet = corr[indUppDet -1]
else:    
    bCorrDet = (tr[indUppDet] - d)/(tr[indUppDet]-tr[indLowDet]) * corr[indLowDet-1] + (d - tr[indLowDet])/(tr[indUppDet]-tr[indLowDet]) * corr[indUppDet-1] #combination of 0-6 and 0-9
print bCorrAtt, bCorrDet

#nu har vi korrelationerna for attachment och detachment
# C = 4, D = 7 

scaleUpper = D/(D-C)
scaleLower = C/(D-C)

C = 0.0
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
p = findp(bCorrDet)
priceUpper = defaultLeg(p) - premiumLeg(coupon,p)#*scaleUpper
#print 'corr[0]', corr[k], 'price upper', priceUpper, 'C', C, 'D', D, 'lavg', Lavg

if c != 0: 
    #lower limit
    D = c #nbr of losses detachment
    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    p = findp(bCorrAtt) 
    priceLower = defaultLeg(p) - premiumLeg(coupon,p)
    print scaleUpper, scaleLower
else: 
    priceLower = 0
print priceLower, priceUpper
price = (scaleUpper*priceUpper - scaleLower*priceLower) #spread 3-6 for finding correlation 0-6

print 'price', price















'''



p = findp(bCorrAtt)
pvDlAtt = defaultLeg(p)
priceAtt = - premiumLeg(coupon,p) + pvDlAtt#0-5
p = findp(bCorrDet)
pvDlDet = defaultLeg(p)
priceDet = - premiumLeg(coupon, p) + pvDlDet#0-8
#priceAttDet = (d - c)/d * priceDet - (d-c)/c * priceAtt #5-8
priceAttDet = d/(d-c) * priceDet - c/(d-c) * priceAtt #5-8
print 'price: ' , priceAttDet, 'pricedet: ', priceDet, 'priceAtt: ', priceAtt

### Assume spreads are known 0-3 3-6 6-9 9-12, spreads for 0-3 0-6 0-9 0-12 are known from 
### Interpolate spreads


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

'''

'''
print 'Spread for', c, 'to', d, ': ', spreadAttDet*10000.0, 'bps'
'''
print("--- %s seconds ---" % (time.time() - start_time))
