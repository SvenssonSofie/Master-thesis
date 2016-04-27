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

nom = 1000000.0

#test
#spread = 0.00676750939073
#marketPrice = 388889.446134
#c = 0.0
#d = 0.03

#marketPrice = 84493.4899999
#c = 0.03
#d = 0.06
marketPrice = 5476.81999264
c = 0.06
d = 0.12
#
delta = [0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778,      \
         0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, \
              0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
              0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778]
print len(delta)
discFac = [0.998600956242      ,0.999611180239      ,  1.00047630826       , 1.00112395119       ,\
          1.00180019558       ,  1.00252355848       , 1.00320411939       , 1.0036842191        ,\
           1.00377352897       , 1.00368518481       , 1.00372980688       , 1.0038363132        ,\
          1.00395370531       ,  1.0040649568        , 1.00392689128       , 1.00358551835       , \
          1.00315265354    , 1.00262525655    , 1.0018353653        ,    1.00087710692   ,0.999799458336]
print len(discFac)

pi = [0.0, 0.0, 0.0039230364813, 0.00396739004003, 0.00397766902931, 0.00398224178954, \
      0.00398489559746, 0.00398659905109, 0.00398777551688, 0.00398863618887, 0.00398931119962,\
     0.00398984448104, 0.00399027231814, 0.00454277200564, 0.00501692809448,  0.0054198805189, \
     0.00576302421683, 0.00606175067052, 0.00633221440352,0.00656349712714, 0.00677161234034]
print len(pi)

#delta = [1.0, 1.0,1.0,1.0,1.0, ]
#discFac = [1.0,1.0,1.0,1.0,1.0,]
#pi = [0.01, 0.02, 0.03, 0.04, 0.05]


coupon = 0.01
nIss = 125#Number of issuers 
recRate = 0.0
nTime = len(delta)#nbr of time points


C = math.ceil(c*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(d*nIss)) #nbr of losses detachment
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
print C, D, Lavg

def conProb (Y0, corr):# conditional probability of k defaults given Y0 
    #print Y0
    piCond = np.zeros(nTime)
    denom = math.sqrt(1-corr**2)
    for timeStep in range(0,nTime):   
        nume = sct.norm.ppf(pi[timeStep]) - corr*Y0
        piCond[timeStep] = sct.norm.cdf(nume/denom)#conditional probability for each issuer(one row each) and each time step(one column each)
        if piCond[timeStep] < piCond[timeStep -1]:
            print timeStep
    #print piCond
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
                #qOld = q[nbrDef + 1, timeStep]
                qOld = temp
                
    return q #probability for all time steps and number of defaults
np.set_printoptions(threshold=np.inf)
# Main program
####################################################################
def correlation(corr):
    print corr
    if corr >= 1 or corr < 0: 
        return float('inf')
    a = -6.0
    b = 6.0
    deg = 50
    x, w = np.polynomial.legendre.leggauss(deg)
    #print len(x), len(w)
    # Translate x values from the interval [-1, 1] to [a, b]
    t = 0.5*(x + 1)*(b - a) + a
    #print conProb(0.2,0.2) #returns a matrix
    p = np.zeros((nIss+1, nTime)) 
    for k in range(0,len(t)):
        p = p + np.multiply(w[k], conProb(t[k], corr))
    p = np.multiply(0.5*(b-a), p)
    p = np.divide(p, p[0,0])
    #print sum(w)
    #gauss = sum(w * conProb(x, corr)) * 0.5*(b - a)
    #print p 
    # x sample points, w weights
    
    
    #probTot = 0.0
    #intPnts = np.linspace(-6,6) #50 points by default between -6 and 6
    #p = np.zeros((nIss+1, nTime)) 
    #sc.integrate.quadrature(conProb, -6, 6, args=(corr,))
    '''for Y0 in intPnts: 
        probTot = probTot + sct.norm.pdf(Y0)
        if Y0 == -6: 
            prev = conProb(Y0, corr)
        else:
            prevTemp = conProb(Y0, corr)
            #prevTemp = np.multiply(conProb(Y0, corr), sct.norm.pdf(Y0))
            p = p + (prev - prevTemp)
            prev = prevTemp
    p = np.divide(p,probTot)'''
    #print p 
    #print np.sum(p,axis = 0), type(p)

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
            #print 'Dl', Dl
            if timeStep == 0: 
                sumPay = sumPay + (p[nbrDef, timeStep])*Dl
            else:
                #if p[nbrDef, timeStep] - p[nbrDef, timeStep-1] < 0:
                    #print nbrDef, timeStep
                sumPay = sumPay + (p[nbrDef, timeStep] - p[nbrDef, timeStep-1])*Dl
                #print (p[nbrDef, timeStep] - p[nbrDef, timeStep-1])*Dl
        #print sumPay    
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
            #print 'NL ', Nl
            sumPay = sumPay + p[nbrDef, timeStep]*Nl

        #discount sum of Payment
        pvNl = pvNl + coupon*delta[timeStep]*sumPay/discFac[timeStep]
    print 'pvNl',  - pvNl    
    print 'PV', pvDl - pvNl
    
    #Recieve default payments, pay premiums
    return abs(pvDl - pvNl - marketPrice) #compound, returns the difference that we want to be zero, finds optimal correlation

#c = conProb(0, 0)
#print c
#c = correlation(0.5)
c = opt.fmin_cobyla(correlation, [0.7], (), args=(), rhoend = 0.00001)
#print "par coupon: ", c
#print("--- %s seconds ---" % (time.time() - start_time))

# Maybe some plots for the report
#prices = np.zeros(10)
#for k in range(0,10):
#    prices[k] = correlation(0.1*k)
#plt.plot(prices)
#plt.show()
    