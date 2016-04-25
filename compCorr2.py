'''
Created on 22 feb 2016

@author: FIHE03E1
'''
import math
import numpy as np
import scipy.stats as sct
import scipy.optimize as opt
import time 
import matplotlib.pyplot as plt

start_time = time.time()

#delta = np.array([0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25])

nom = 1000000.0

#itrx eur crossover version 1 5Y
spread = 0.03125052929660203
marketPrice = 0.608660831012495*nom
c = 0.00
d = 0.10

#itrx eur crossover version 1 5Y
#spread = 0.03125052929660203
#marketPrice = 0.07442900191774378*nom
#c = 0.10
#d = 0.20

#itrx eur crossover version 1 5Y
spread = 0.03125052929660203
marketPrice = -0.1251604990411281*nom
c = 0.20
d = 0.35

#itrx eur crossover version 1 5Y
spread = 0.03125052929660203
marketPrice = -0.1954961471691892*nom
c = 0.35
d = 1.00

#test
spread = 0.0439641
marketPrice = 67,91#674892
c = 0.0
d = 0.1

delta = [91.0, 91.0,91.0,91.0,91.0,91.0,92.0,92.0,91.0,90.0,92.0,92.0,91.0,90.0,92.0,92.0,91.0,90.0,92.0,94.0,89.0]
delta = np.divide(delta,360.0)
print delta

discFac = [1.0, 1.0,1.0,1.0,1.0,1.0, 1.00048227,  1.00113133, 1.00180821, 1.00253211,\
           1.00321245, 1.00368726, 1.00377284, 1.00368368, 1.00373134, 1.00383709, \
           1.00395518, 1.00406595 , 1.00392370, 1.00357276, 1.00315265]

pi = np.zeros(len(delta))
for k in range(6,len(delta)): 
    pi[k] = spread*(k-6+1)/(len(delta)-6)
print 'pi ', pi, 'marketprice', marketPrice, 'c', c, 'd', d

#Test non standard tranches
#c = 0.03
#d = 0.07
#marketPrice = 0.0
#coupon = 0.0316635199716

# iTrx Europe 24, 3Y, 3-6%
#delta = np.array([0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.255555555556 ])
#discFac = np.array([0.998789615268, 0.999800018888, 1.00057366592, 1.00123214037,  1.00194107616, 1.00269593306, 1.0034370463, 1.00386099671, 1.00395019207, 1.00392703588, 1.004093509, 1.00426863106, 1.00449749648])
#discFac = np.array([0.998789615268, 0.999800018888, 1.00057366592, 1.00123214037,  1.00194107616, 1.00269593306, 1.0034370463, 1.00386099671, \
                   # 1.00395019207, 1.00392703588, 1.004093509, 1.00426863106, 1.00449749648, 1.0048, 1.005, 1.005, 1.0053, 1.0055, 1.0058, 1.058, 1.061])

#discFac = np.array([0.998789615268, 0.999800018888, 1.02, 1.04,  1.08, 1.12, 1.13, 1.16, 1.20, 1.24, 1.25, 1.27, 1.303])
#pi = np.array([0.0, 0.0, 0.00175271248834 , 0.00395723568787 , 0.00613300727422 ,0.00828019451109  ,0.0104703024593  ,0.0126555737865  , 0.0148123445854 , 0.0169407806589 , 0.0191117626216 , 0.0212779502011 ,0.0234683081583  ])


#c = 0.03
#d = 0.06
#marketPrice = 64336.0

#iTrx Europw 24, 3Y, 6-12%
#delta = np.array([0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.255555555556 ])
#discFac = np.array([0.998789615268, 0.999800018888, 1.00057366592, 1.00123214037,  1.00194107616, 1.00269593306, 1.0034370463, 1.00386099671, 1.00395019207, 1.00392703588, 1.004093509, 1.00426863106, 1.00449749648])
#discFac = np.array([0.998789615268, 0.999800018888, 1.02, 1.04,  1.08, 1.12, 1.13, 1.16, 1.20, 1.24, 1.25, 1.27, 1.303])
#pi = np.array([0.0, 0.0, 0.00168074891904 ,0.00388543778954  ,0.0060613728715  , 0.00820872144173 , 0.0103989939335 , 0.0125844294262 ,0.0147413622349  , 0.0168699581757 ,0.0190411031868  ,0.0212074534399  ,0.0233979232675  ])
#c = 0.06
#d = 0.12
#marketPrice = 10370.0

#iTrx Europw 24, 5Y, 0-3%
#delta = np.array([0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, 0.255555555556 , 0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778 ])
#discFac = np.array([0.998789615268, 0.999800018888, 1.00057366592, 1.00123214037,  1.00194107616, 1.00269593306, 1.0034370463, 1.00386099671, 1.00395019207, 1.00392703588, 1.004093509, 1.00426863106, 1.00449749648])
#discFac = np.array([0.998789615268, 0.999800018888, 1.02, 1.04,  1.08, 1.12, 1.13, 1.16, 1.20, 1.24, 1.25, 1.27, 1.303, 1.4, 1.43, 1.5, 1.59, 1.65, 1.67, 1.8, 1.87, ])
#pi = np.array([0.0, 0.0, 0.00168074891904 ,0.00388543778954  ,0.0060613728715  , 0.00820872144173 , 0.0103989939335 , 0.0125844294262 ,0.0147413622349  , 0.0168699581757 ,0.0190411031868  ,0.0212074534399  ,0.0233455499208 , 0.0280479206174 , 0.0328313881374 , 0.0375913137956 ,  0.0422764552814 , 0.0469387888364 , 0.0517309953653 ,0.0563473028831 , 0.0609562820975 ])
#c = 0.0
#d = 0.03
#marketPrice = 438693.0


####################################################################
#coupon = 0.01
coupon = 0.05
nIss = 125#Number of issuers 
recRate = 0.0
nTime = len(delta)#nbr of time points


C = math.ceil(c*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(d*nIss)) #nbr of losses detachment
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
print C, D

def conProb (Y0, corr):# conditional probability of k defaults given Y0 
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
    return q #probability for all time steps and number of defaults

# Main program
####################################################################
def correlation(corr):
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
    #print p, np.sum(p,axis = 0), type(p)

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

#c = correlation(math.sqrt(0.58))
c = opt.fmin_cobyla(correlation, [0.7], (), args=(), rhoend = 0.001)
print "par coupon: ", c
print("--- %s seconds ---" % (time.time() - start_time))

# Maybe some plots for the report
#prices = np.zeros(10)
#for k in range(0,10):
#    prices[k] = correlation(0.1*k)
#plt.plot(prices)
#plt.show()
    