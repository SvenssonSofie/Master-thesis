import math
import numpy as np
import scipy.stats as sct
import scipy.optimize as opt
import time
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

################################Finding correlation for 0-detachment for all standard tranches
start_time = time.time()

delta = [0.252777777778, 0.255555555556, 0.252777777778,      \
         0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, \
              0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
              0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778]

discFac = [1.00047630826       , 1.00112395119       ,\
          1.00180019558       ,  1.00252355848       , 1.00320411939       , 1.0036842191        ,\
           1.00377352897       , 1.00368518481       , 1.00372980688       , 1.0038363132        ,\
          1.00395370531       ,  1.0040649568        , 1.00392689128       , 1.00358551835       , \
          1.00315265354    , 1.00262525655    , 1.0018353653        ,    1.00087710692   ,0.999799458336]


spread = [0.0039230364813, 0.00396739004003, 0.00397766902931, 0.00398224178954, \
      0.00398489559746, 0.00398659905109, 0.00398777551688, 0.00398863618887, 0.00398931119962,\
     0.00398984448104, 0.00399027231814, 0.00454277200564, 0.00501692809448,  0.0054198805189, \
     0.00576302421683, 0.00606175067052, 0.00633221440352,0.00656349712714, 0.00677161234034]

# result 160429 - 0.67596046  0.80258705  0.91920945

sumDelta = np.cumsum(delta)
pi = np.zeros(len(spread))
for s in range(0,len(spread)):
    pi[s] = 1 - math.exp(-spread[s]*sumDelta[s])

tr = np.array([0.0, 0.03, 0.06, 0.12])
coupon = np.array([0.01, 0.01, 0.01])
#pr = np.array([388889.446134, 84493.4899999, 5476.81999264])#prices on market 

pr = [388889.446134, 84493.4899999, 5476.81999264]
comp = [0.65638817, 0.87389747, 0.27502]#removed delta

#Pricing of derivatives
recRate = 0.4
nTime = len(delta)
nom = 1000000.0
nIss = 125 #Number of issuers

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


