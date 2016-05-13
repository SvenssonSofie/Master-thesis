'''
Created on 22 feb 2016

@author: FIHE03E1
'''
import math
import numpy as np
import scipy as sc
import scipy.stats as sct
import scipy.optimize as opt
import time 
import matplotlib.pyplot as plt

start_time = time.time()

############# S24 3Y
pi = np.array([0.00087479504671772457, 0.0029353834489070918, 0.004969393415826806, 0.0069769703635795821, 0.0090249736847164419, 0.011068753219014127, 0.013086171126700852, 0.015077371637674086, 0.017108668751384948, 0.019135776533014792, 0.021188431563949561])
spread = np.array([0.0047228532506136903, 0.0048101666897295524, 0.0048253362059923867, 0.0048316350692150416, 0.0048351736249719465, 0.0048374002858278193, 0.0048389188662819015, 0.0048400205026313961, 0.0048408789399059949, 0.0048415530671146135, 0.0048489188205022606])
discFac = np.array([1.0003661802496806, 1.0010318855144935, 1.0017322165435656, 1.0024681809694533, 1.0031665273659054, 1.0037827350418536, 1.0039805436318214, 1.0039247441142243, 1.0039599948710676, 1.0041599686436489, 1.0043651935486468])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25555555555555554])
#print len(pi), len(spread), len(delta), len(discFac)
c = 0.06
d = 0.12
marketPrice = -17076.0

nIss = 125
coupon = 0.01
recRate = 0.0


sumDelta = np.cumsum(delta)
nTime = len(delta)
nom = 1000000.0

recRate2 = 0.4
c = c/(1-recRate2)
d = d/(1-recRate2)


C = math.ceil(c*nIss) #nbr of losses attachment
if C != 0:
    C = C-1
D = float(math.trunc(d*nIss))
C = C
Lavg = (1.0-recRate)*nom/(D-C)
print C, D, Lavg
if D > nIss: 
    D = nIss
print C, D, Lavg
def conProb (Y0, corr):# conditional probability of k defaults given Y0 

    piCond = np.zeros(nTime)
    denom = math.sqrt(1-corr**2)
    for timeStep in range(0,nTime):   
        nume = sct.norm.ppf(pi[timeStep]) - corr*Y0
        piCond[timeStep] = sct.norm.cdf(nume/denom)#probability of default before timeStep

    q = np.zeros((nIss+1, nTime)) 
    for timeStep in range(0, nTime):
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

# Main program
##########################
def correlation(corr):
    print corr
    if corr >= 1 or corr < 0: 
        return float('inf')
    
    a = -6.0
    b = 6.0
    deg = 50
    x, w = np.polynomial.legendre.leggauss(deg)
    w = w*(b-a)*0.5#weights
    t = 0.5*x*(b - a) #sample points
    
    p = np.zeros((nIss+1, nTime))
    for k in range(0,len(t)):
        p = p + np.multiply(w[k]*sct.norm.pdf(t[k]), conProb(t[k], corr))    
    #print np.sum(p,0)
    
    pvDl = 0.0
    for timeStep in range(0,nTime): #0 to 4
        sumPay = 0.0
        for nbrDef in range(1,nIss+1):#0-50            
            if nbrDef <= C:
                Dl = 0.0
            elif nbrDef > C and nbrDef <= D: 
                Dl = Lavg*(nbrDef-C)
            else:
                Dl = Lavg*(D-C)
            #print Dl     
            if timeStep == 0: 
                sumPay = sumPay + (p[nbrDef, timeStep])*Dl
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
            #print Nl
            sumPay = sumPay + p[nbrDef, timeStep]*Nl

        #discount sum of Payment
        pvNl = pvNl + coupon*delta[timeStep]*sumPay/discFac[timeStep]
    print 'pvNl',  - pvNl    
    print 'PV', pvDl - pvNl
    
    return abs(pvDl - pvNl - marketPrice)

#c = conProb(0, 0)
#print c
#c = correlation(0.680779488973)
c = opt.fmin_cobyla(correlation, [0.7], (), args=(), rhoend = 0.00001)
print "par corr: ", c
print("--- %s seconds ---" % (time.time() - start_time))

# Maybe some plots for the report
#prices = np.zeros(10)
#for k in range(0,10):
#    prices[k] = correlation(0.1*k)
#plt.plot(prices)
#plt.show()
    