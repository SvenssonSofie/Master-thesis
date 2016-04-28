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

#thursday
marketPrice = -4709.1206101
c = 0.06
d =0.12

discFac = [0.998567685447      ,0.999577864556          ,  1.00045062864              , 1.00045062864             ,\
          1.00178528504           ,  1.00250401027       , 1.00317902449       , 1.0036796674         ,\
           1.00376829329        , 1.00365008529       , 1.00363960304       , 1.00370342137        ,\
          1.00375936033       ,  1.00380409316        , 1.00361392274        ,1.00320197241        , \
          1.0026924065        ,1.00208149172       , 1.00119465201       ,   1.00011221868  ,0.998900782422]
pi = [0.0, 0.0, 0.0042460839466,0.00429791106969, 0.00430942354938, 0.0043144870735, \
      0.00431740949283, 0.0043192794268, 0.00432056792291, 0.00432150925416, 0.00432224686705,\
     0.00432282896438, 0.00432329565995, 0.00486536500626, 0.0053303458054,  0.00572533837007, \
     0.00606158559488, 0.00635421564115, 0.00661908726913,0.00684552159625, 0.00704935047305]


delta = [0.252777777778, 0.252777777778, 0.252777777778, 0.255555555556, 0.252777777778,      \
         0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, \
              0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
              0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778]
'''
discFac = [0.998600956242      ,0.999611180239      ,  1.00047630826       , 1.00112395119       ,\
          1.00180019558       ,  1.00252355848       , 1.00320411939       , 1.0036842191        ,\
           1.00377352897       , 1.00368518481       , 1.00372980688       , 1.0038363132        ,\
          1.00395370531       ,  1.0040649568        , 1.00392689128       , 1.00358551835       , \
          1.00315265354    , 1.00262525655    , 1.0018353653        ,    1.00087710692   ,0.999799458336]


pi = [0.0, 0.0, 0.0039230364813, 0.00396739004003, 0.00397766902931, 0.00398224178954, \
      0.00398489559746, 0.00398659905109, 0.00398777551688, 0.00398863618887, 0.00398931119962,\
     0.00398984448104, 0.00399027231814, 0.00454277200564, 0.00501692809448,  0.0054198805189, \
     0.00576302421683, 0.00606175067052, 0.00633221440352,0.00656349712714, 0.00677161234034]

'''
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


#np.set_printoptions(threshold=np.inf)
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
    w = w*(b-a)*0.5
    # Translate x values from the interval [-1, 1] to [a, b]
    t = 0.5*x*(b - a) 
    p = np.zeros((nIss+1, nTime))
    for k in range(0,len(t)):
        p = p + np.multiply(w[k]*sct.norm.pdf(t[k]), conProb(t[k], corr))
    #gauss = sum(w * conProb(x, corr)) * 0.5*(b - a) 
    #print np.sum(p, 0)
    
    
    '''
    def getGaussHermitePointsandWeights(n):
        [x,w]=np.polynomial.hermite.hermgauss(n)
        x=x*math.sqrt(2)
        w=w/math.sqrt(math.pi)
        return [x,w]

    #def testfun(x):
    #   return x*np.exp(3*x)

    [x,w]=getGaussHermitePointsandWeights(50)
    #approxint=np.sum(testfun(x)*w)
    w = w*(b-a)*0.5
    x = 0.5*x*(b - a)
    p = np.zeros((nIss+1, nTime)) 
    for k in range(0,len(x)):
        p = p + np.multiply(w[k]*sct.norm.pdf(x[k]), conProb(x[k], corr))
    #p = np.divide(p, p[0,0])
    #p = np.multiply(0.5*(b-a), p)
    '''
    
    #trueint=math.exp(4.5)*3.0
    #print 'approximative: %.12f' % approxint, 'exact: %.12f' % trueint 
    
    

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
c = opt.fmin_cobyla(correlation, [0.4], (), args=(), rhoend = 0.00001)
print "par coupon: ", c
#print("--- %s seconds ---" % (time.time() - start_time))

# Maybe some plots for the report
#prices = np.zeros(10)
#for k in range(0,10):
#    prices[k] = correlation(0.1*k)
#plt.plot(prices)
#plt.show()
    