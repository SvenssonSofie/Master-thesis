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
'''
#test
marketPrice = 388889.446134
c = 0.0
d = 0.03

#marketPrice = 84493.4899999
#c = 0.03
#d = 0.06

#marketPrice = 5476.81999264
#c = 0.06
#d = 0.12

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





sumDelta = np.cumsum(delta)
pi = np.zeros(len(spread))
for s in range(0,len(spread)):
    pi[s] = 1 - math.exp(-spread[s]*sumDelta[s])

coupon = 0.01
nIss = 125 #Number of issuers 
recRate = 0.0

'''


############# xover 
#pi = np.array([0.0054613538589327604, 0.017607791695479436, 0.029476260356662176, 0.041073269614748997, 0.052784774205969498, 0.064353244574194668, 0.075656972454807292, 0.086702157863696949, 0.097856390541134042, 0.10887439503442109, 0.11964025435853443, 0.13563164728614596, 0.15167821333151854, 0.1674268829059713, 0.18271673524872123, 0.19772579593502271, 0.21294030436312905, 0.22739432201532073, 0.24159546220851535])
#spread = np.array([0.028147118285289249, 0.028634939976430237, 0.028723255753864623, 0.028760259933442979, 0.028781104758606655, 0.028794264502661968, 0.028803257244908714, 0.028809783040567572, 0.02881486421967942, 0.028818864331903976, 0.028822059381286458, 0.030048444153557351, 0.031093429118851375, 0.031977030520288628, 0.032726160277256104, 0.033375748224999592, 0.033961755674568911, 0.034461192631016067, 0.034906824184687317])
#discFac = np.array([1.0003635153416293, 1.0010199296089801, 1.0017009237372976, 1.0024044576014735, 1.0030619651536226, 1.0035857631689056, 1.0036885676120733, 1.0035626340471684, 1.0035425703254608, 1.0036384904666531, 1.0037402530029267, 1.0038644452392391, 1.0038167791152117, 1.0035736824892858, 1.0032519368338813, 1.0028494417398197, 1.0022296412972789, 1.0014260482045545, 1.000516346531535])
#delta = np.array([0.252777777778, 0.255555555556, 0.252777777778,      \
 #        0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, \
 #             0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
#              0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778])


#pi = np.zeros(len(spread))
#for s in range(0,len(spread)):
#    pi[s] = 1 - math.exp(-spread[s]*sumDelta[s])
#print pi
#pi = spread
#c = 0.0
#d = 0.10
#marketPrice = 631782.070252

#nIss = 75
#coupon = 0.05
#recRate = 0.0

#c = 0.10
#d = 0.20
#marketPrice = 102100.123354

#c = 0.20
#d = 0.35
#marketPrice = -115046.97444

######## 10 juni 

discFac = np.array([1.0003635153416293, 1.0010199296089801, 1.0017009237372976, 1.0024044576014735, 1.0030619651536226, 1.0035857631689056, 1.0036885676120733, 1.0035626340471684, 1.0035425703254608, 1.0036384904666531, 1.0037402530029267, 1.0038644452392391, 1.0038167791152117, 1.0035736824892858, 1.0032519368338813, 1.0028494417398197, 1.0022296412972789, 1.0014260482045545, 1.000516346531535])
spread = np.array([0.0045244341749098834, 0.0046027012776632429, 0.0046168697365552604, 0.0046228051609962933, 0.0046261520984201077, 0.0046282640722665226, 0.004629706789672363, 0.0046307540958324912, 0.0046315707195910133, 0.0046322129058806545, 0.0046327260745146577, 0.0051941879063954243, 0.0056748678580730676, 0.0060825418584728808, 0.0064291123565602994, 0.0067303788304509445, 0.0070028012033693026, 0.007235496127153785, 0.0074448698161273434])
pi = np.array([0.00087988747682610224, 0.0028514524517694317, 0.004797760055717859, 0.0067189430820959739, 0.0086789858412678056, 0.010635160845597347, 0.012566275637129642, 0.014472461972708572, 0.016417204703299659, 0.018358109870613926, 0.020274150444318573, 0.024840910419199558, 0.029487153021399259, 0.034111258137983769, 0.038663426774489862, 0.043194141346570847, 0.04785180082622742, 0.052339211209776693, 0.056821960151918427])
delta = np.array([0.252777777778, 0.255555555556, 0.252777777778,      \
         0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, \
              0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
              0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778])


marketPrice = -1612.42055845
c = 0.06
d = 0.12

#marketPrice = 69276.9371794
#c = 0.03
#d = 0.06

#marketPrice = 406421.037154
#C = 0.0
#D = 0.03

nIss = 125
coupon = 0.01
recRate = 0.4




sumDelta = np.cumsum(delta)
nTime = len(delta)
nom = 1000000.0

C = math.ceil(c*nIss) #nbr of losses attachment
if C != 0:
    C = C-1
D = float(math.trunc(d*nIss)) 
Lavg = (1.0-recRate)*nom/(D-C)
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
    print np.sum(p,0)
    
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
c = opt.fmin_cobyla(correlation, [0.5], (), args=(), rhoend = 0.00001)
print "par corr: ", c
print("--- %s seconds ---" % (time.time() - start_time))

# Maybe some plots for the report
#prices = np.zeros(10)
#for k in range(0,10):
#    prices[k] = correlation(0.1*k)
#plt.plot(prices)
#plt.show()
    