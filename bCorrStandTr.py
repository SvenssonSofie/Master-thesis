import math
import numpy as np
import scipy.stats as sct
import scipy.optimize as opt
import time
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

################################Finding correlation for 0-detachment for all standard tranches
start_time = time.time()


############# S24 main 3Y
pi = np.array([0.00087479504671772457, 0.0029353834489070918, 0.004969393415826806, 0.0069769703635795821, 0.0090249736847164419, 0.011068753219014127, 0.013086171126700852, 0.015077371637674086, 0.017108668751384948, 0.019135776533014792, 0.021188431563949561])
spread = np.array([0.0047228532506136903, 0.0048101666897295524, 0.0048253362059923867, 0.0048316350692150416, 0.0048351736249719465, 0.0048374002858278193, 0.0048389188662819015, 0.0048400205026313961, 0.0048408789399059949, 0.0048415530671146135, 0.0048489188205022606])
discFac = np.array([1.0003661802496806, 1.0010318855144935, 1.0017322165435656, 1.0024681809694533, 1.0031665273659054, 1.0037827350418536, 1.0039805436318214, 1.0039247441142243, 1.0039599948710676, 1.0041599686436489, 1.0043651935486468])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25555555555555554])


nIss = 125
coupon = 0.01
recRate = 0.0

recRate2 = 0.4
tr = np.array([0.0, 0.03, 0.06, 0.12, 1.0])/(1-recRate2)
pr = [209314.0, 6965.0, -17076.0, -23219.0]
comp = [0.68061913, 0.97273339, 0.43740319, 0.84061195]
coupon = np.array([0.01, 0.01, 0.01, 0.01])



#Pricing of derivatives

recRate = 0.0
nTime = len(delta)
nom = 1000000.0


C = 0.0 #always zero due to base correlations
D = 0.0 #nbr of losses detachment
Lavg = 0.0#hela nominalen ska va avksriven inom omradet C-D

#for funciton findp, do not generete every function call
a = -6.0
b = 6.0
deg = 50
x, w = np.polynomial.legendre.leggauss(deg)
w = w*(b-a)*0.5#weights
t = 0.5*x*(b - a) #sample points 


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
        piCond[timeStep] = sct.norm.cdf(nume/denom)

    q = np.zeros((nIss+1, nTime)) #prob of number of defaults (rows = nbr of issuers +1 (zero defaults)) and time step (columns)
    for timeStep in range(0, nTime): #for each time step, a probability of number of defaults 
        q[0, timeStep] = 1.0     
        alpha = piCond[timeStep]
        for issuer in range(0, nIss): #from 0 to 49, add one at the time
            for nbrDef in range(0, issuer+1): #update all that are added so far
                if nbrDef == 0:
                    qOld = q[nbrDef, timeStep]
                q[nbrDef, timeStep] = q[nbrDef, timeStep] - qOld*alpha
                temp = q[nbrDef + 1, timeStep] 
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
    p = np.zeros((nIss+1, nTime))
    for k in range(0,len(t)):
        p = p + np.multiply(w[k]*sct.norm.pdf(t[k]), conProb(t[k], corr))
    return p 

def parRate(coupon, p): 
    print coupon
    if coupon < 0: 
        return float('inf')
    
    pvDl = defaultLeg(p)
    pvNl = premiumLeg(coupon, p)
    return abs(pvDl - pvNl)
    

def correlation(corr, coupon, marketPrice): #coupon from pricing in PRIME
    print corr
    if corr >= 1 or corr < 0: 
        return float('inf')

    p = findp(corr)
    pvDl  = defaultLeg(p)
    pvNl = premiumLeg(coupon, p)
    
    #Recieve default payments, pay premiums
    return abs(pvDl - pvNl - marketPrice) #compound, returns the difference that we want to be zero, finds optimal correlation


############# correlation for standard tranches
corr = np.zeros(len(tr)-1)
corr[0] = comp[0]# just for running faster in test case compoundCorr(pr[0], spr[0]), compound correlation

for k in range(1,len(tr)-1): #find base correlation for all tranches
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #(0-7) = (0-3) + (3-7), this is for (3-7)
    D = float(math.trunc(tr[k+1]*nIss)) #nbr of losses detachment    
    C = math.ceil(tr[k]*nIss) #For market premium we need tranche with other attachment than zero
    if C != 0:
        C = C-1

    #Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    #p = findp(comp[k])
    #market premimum so that price (3-7) becomes zero  
    #marketPrem = opt.fmin_cobyla(parRate, [0.1], (), args=([p]), rhoend = 0.001)#from correlation find price = 0
    priceUpp = pr[k]*(D-C)/D
    priceLow = pr[k-1]*C/D
    #scale = C/D

    #C = 0
    #D = float(math.trunc(tr[k]*nIss)) #nbr of losses detachment
    #Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D

    #p = findp(corr[k-1]) #correlation for 0-3 find probability of k defaults, changes global variable
    #price for 0-7 is price for 0-3 times this scale
    pr[k] = priceLow + priceUpp #spread 3-6 for finding correlation 0-6
    print pr[k]
    
    #find correlation given price and premium
    C = 0
    D = float(math.trunc(tr[k+1]*nIss)) #nbr of losses detachment
    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-
    corr[k] = compoundCorr(pr[k], coupon[k])

    print corr
print("--- %s seconds ---" % (time.time() - start_time))


