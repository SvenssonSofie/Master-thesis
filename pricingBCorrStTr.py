import math
import numpy as np
import scipy.stats as sct
import time
import scipy.optimize as opt
start_time = time.time()


tranches = [0, 1, 2, 3]

# S24 main 3Y
pi = np.array([0.00087479504671772457, 0.0029353834489070918, 0.004969393415826806, 0.0069769703635795821, 0.0090249736847164419, 0.011068753219014127, 0.013086171126700852, 0.015077371637674086, 0.017108668751384948, 0.019135776533014792, 0.021188431563949561])
spread = np.array([0.0047228532506136903, 0.0048101666897295524, 0.0048253362059923867, 0.0048316350692150416, 0.0048351736249719465, 0.0048374002858278193, 0.0048389188662819015, 0.0048400205026313961, 0.0048408789399059949, 0.0048415530671146135, 0.0048489188205022606])
discFac = np.array([1.0003661802496806, 1.0010318855144935, 1.0017322165435656, 1.0024681809694533, 1.0031665273659054, 1.0037827350418536, 1.0039805436318214, 1.0039247441142243, 1.0039599948710676, 1.0041599686436489, 1.0043651935486468])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25555555555555554])


nIss = 125
recRate = 0.0

recRate2 = 0.4
tr = np.array([0.0, 0.03, 0.06, 0.12, 1.0])/(1-recRate2)
pr = [209314.0, 6965.0, -17076.0, -23219.0]
comp = [0.68061913]
coupons = np.array([0.05, 0.05, 0.05, 0.05])



corr = [ 0.68061913,  0.76454688 , 0.85944655 , 0.94466163]



#Pricing of derivatives
recRate = 0.0
nTime = len(delta)#nbr of time points
nom = 1000000.0

C = 0.0
D = 0.0
Lavg = 0.0

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
        piCond[timeStep] = sct.norm.cdf(nume/denom)
        
    q = np.zeros((nIss+1, nTime)) 
    for timeStep in range(0, nTime): 
        q[0, timeStep] = 1.0     
        alpha = piCond[timeStep]
        for issuer in range(0, nIss):
            for nbrDef in range(0, issuer+1): #update all that are added so far
                if nbrDef == 0:
                    qOld = q[nbrDef, timeStep]
                q[nbrDef, timeStep] = q[nbrDef, timeStep] - qOld*alpha
                temp = q[nbrDef + 1, timeStep] 
                q[nbrDef + 1, timeStep] = q[nbrDef + 1, timeStep] + qOld*alpha
                qOld = temp            
    return q

####################main
price = np.zeros(len(tranches))
for k in tranches: 

    D = float(math.trunc(tr[k+1]*nIss)) #nbr of losses detachment
    C = math.ceil(tr[k]*nIss) #For market premium we need tranche with other attachment than zero
    if C != 0:
        C = C-1
    
    scaleUpper = D/(D-C)
    scaleLower = C/(D-C)


    C = 0.0
    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    p = findp(corr[k])
    priceUpper = defaultLeg(p) - premiumLeg(coupons[k],p)#*scaleUpper
    #print 'corr[0]', corr[k], 'price upper', priceUpper, 'C', C, 'D', D, 'lavg', Lavg

    if k != 0: 
        #lower limit
        D = float(math.trunc(tr[k]*nIss)) #nbr of losses detachment
        Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
        p = findp(corr[k-1]) 
        priceLower = defaultLeg(p) - premiumLeg(coupons[k],p)
        print scaleUpper, scaleLower
    else: 
        priceLower = 0
    print priceLower, priceUpper
    price[k] = (scaleUpper*priceUpper - scaleLower*priceLower) #spread 3-6 for finding correlation 0-6

print 'price ', price
print("--- %s seconds ---" % (time.time() - start_time))