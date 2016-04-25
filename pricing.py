'''
Created on 22 feb 2016

@author: FIHE03E1
'''
import math
import numpy as np
import scipy.stats as sct
import scipy.optimize as opt

pi = np.array([0.0404796532753, 0.0830294094631, 0.135445204365, 0.187331591664, 0.241929290875])
discFac = np.array([1.00255795468,1.00331139094, 1.00345792344, 1.00150540866, 0.997194824632])
delta = np.array([1.01388888889, 1.01388888889, 1.01388888889, 1.02222222222, 1.01388888889])
coupon = 0.05
nIss = 3#Number of issuers 
marketPrice = 0.0

#Pricing of derivatives
recRate = 0.0
nTime = len(pi)#nbr of time points
T = 5.0#years
nom = 1000000.0

C = 0.0*nIss #nbr of losses attachment
D = 0.5*nIss #nbr of losses detachment
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D

def conProb (Y0, corr):# conditional probability of k defaults given Y0 
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
        for issuer in range(0, nIss): #from 0 to 1, add one at the time
            for nbrDef in range(0, issuer+1) :  #update all that are added so far
                #qOld = q[nbrDef, timeStep]
                #q[nbrDef, timeStep] = q[nbrDef, timeStep] - alpha*qOld
                #q[nbrDef + 1, timeStep] = q[nbrDef + 1, timeStep] + alpha*qOld
                if nbrDef == 0:
                    qOld = q[nbrDef, timeStep]
                q[nbrDef, timeStep] = q[nbrDef, timeStep] - qOld*alpha
                temp = q[nbrDef + 1, timeStep] # save this to next step since it is needed for updating after overwritten
                q[nbrDef + 1, timeStep] = q[nbrDef + 1, timeStep] + qOld*alpha
                qOld = temp                
    return q
    #now we have probability distribution for all time steps and all number of defaults, q is conditional probability

####################################################################
def correlation(corr): 
    
    probTot = 0.0
    intPnts = np.linspace(-6,6) #50 points by default between -6 and 6
    p = np.zeros((nIss+1, nTime)) 
    for Y0 in intPnts: 
        probTot = probTot + sct.norm.pdf(Y0)
        p = p + np.multiply(conProb(Y0, corr), sct.norm.pdf(Y0))
    p = np.divide(p,probTot) #p is the 
    #print p
    
    #payment from insurer
    pvDl = 0.0
    for timeStep in range(0,nTime): #1 to 4
        sumPay = 0.0
        for nbrDef in range(0,nIss+1):#0-50
            
            if nbrDef <= C:
                Dl = 0.0
            elif nbrDef > C and nbrDef <= D: 
                Dl = Lavg*(nbrDef-C)
            else:
                Dl = Lavg*(D-C)                
            if timeStep == 0:
                sumPay = sumPay + p[nbrDef, timeStep]*Dl #when timestep zero, 1y from now, p[nbrDef, timestep-1] = 0 
            else: 
                sumPay = sumPay + (p[nbrDef, timeStep] - p[nbrDef, timeStep-1])*Dl
        
        #discount sum of Payments
        pvDl = pvDl + sumPay*discFac[timeStep]
    print 'pvDl', -pvDl
    
    #present value to insurer, premiums
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
        pvNl = pvNl + coupon*delta[timeStep]*sumPay*discFac[timeStep]
    print 'pvNl', pvNl    
    print 'PV', pvNl - pvDl
    
    #compound correlation
    #return abs(pvNl - pvDl - marketPrice) #compound, returns the difference that we want to be zero, finds optimal correlation

c = correlation(math.sqrt(0.75))
#c = opt.fmin(correlation, [0.5])
#print "correlation optimum: ", c