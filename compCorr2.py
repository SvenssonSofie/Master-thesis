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

############# S24 5Y
#pi = np.array([0.00078406209223425005, 0.0029025609697107546, 0.0049936140589693645, 0.0070573752975346959, 0.0091625737396605622, 0.01126330882169857, 0.013336828250815169, 0.015383284673755715, 0.017470830845738194, 0.019553951083291277, 0.021610083873515373, 0.026332554067412195, 0.031136409136407472, 0.035916563070461871, 0.040621557327197189, 0.045303589908513464, 0.050115984541564074, 0.054751681687788323, 0.059381210714061772])
#spread = np.array([0.004836493469134874, 0.0049435372779453425, 0.0049601248471501267, 0.0049668580686195287, 0.0049706006697175092, 0.0049729420073176926, 0.004974532788767738, 0.0049756834329529771, 0.0049765783406867312, 0.004977280256108641, 0.0049778402528445065, 0.005547448502903618, 0.0060345496155683274, 0.0064472591877582164, 0.0067978163208867641, 0.0071023305563781714, 0.0073775257164661887, 0.0076124602657073777, 0.0078237390368579487])
#discFac = np.array([1.0003274275132721, 1.0009892081227751, 1.0016754525209999, 1.0023863451292354, 1.0030604094581885, 1.0036701738001863, 1.0038532663691215, 1.0037706753812285, 1.0037488378459536, 1.0038969889780374, 1.004032911854813, 1.0042120175574905, 1.0042659255614939, 1.0041048481631689, 1.0038701302138373, 1.0035605025093022, 1.0030749067583251, 1.002380718501074, 1.0015862998749887])
#delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25277777777777777, 0.26111111111111113, 0.25277777777777777, 0.25277777777777777])
#print len(pi), len(spread), len(delta), len(discFac), sum(delta)
#c = 0.06
#d = 0.12
#marketPrice = 5826.0
#0.28814256

#c = 0.00
#d = 0.03
#marketPrice = 426604.0
#0.67738816

#c = 0.03
#d = 0.06
#marketPrice = 85594.0
#0.94294213

#c = 0.12
#d = 1.0
#marketPrice = -30899.0
# 0.70222558


#nIss = 125
#coupon = 0.01

# Xover S24 5Y
pi = np.array([0.0045969163165063831, 0.016929957100967785, 0.028978612372667967, 0.040749613362203285, 0.052634722354568009, 0.064372574887380463, 0.075839766253202989, 0.087042702835787966, 0.098354240608738985, 0.10952562846757441, 0.12043941717304518, 0.13677406893502542, 0.15315821249913075, 0.1692313828282247, 0.18482970908523089, 0.20013516464756798, 0.21564351953391681, 0.23037042186414414, 0.24483318854606917])
spread = np.array([0.028410302368569135, 0.029040051636199059, 0.029137644999354142, 0.029177266645545812, 0.029199269402997122, 0.029213039864198809, 0.029222398790065638, 0.029229166172768623, 0.029234422458971971, 0.029238549101640702, 0.029241840079856647, 0.030522666546099732, 0.031612704567184229, 0.032533337725092398, 0.033313100248131521, 0.03398868424654531, 0.034597712065773778, 0.035116430341139065, 0.035578919026911168])
discFac = np.array([1.0003274275132721, 1.0009892081227751, 1.0016754525209999, 1.0023863451292354, 1.0030604094581885, 1.0036701738001863, 1.0038532663691215, 1.0037706753812285, 1.0037488378459536, 1.0038969889780374, 1.004032911854813, 1.0042120175574905, 1.0042659255614939, 1.0041048481631689, 1.0038701302138373, 1.0035605025093022, 1.0030749067583251, 1.002380718501074, 1.0015862998749887])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25277777777777777, 0.26111111111111113, 0.25277777777777777, 0.25277777777777777])
print len(pi), len(spread), len(delta), len(discFac), sum(delta)

#c = 0.0
#d = 0.10
#marketPrice = 654168.0
# 0.56695777
#c = 0.1
#d = 0.2
#marketPrice = 113513.0
# 0.95892555
#c = 0.2
#d = 0.35
#marketPrice = -100989.0
c = 0.35
d = 1.0
marketPrice = -196503.0


nIss = 75
coupon =  0.05


recRate = 0.0


sumDelta = np.cumsum(delta)
nTime = len(delta)
nom = 1000000.0

recRate2 = 0.4
c = c/(1-recRate2)
d = d/(1-recRate2)


C = math.ceil(c*nIss) #nbr of losses attachment
if C != 0 and C != c*nIss:
    C = C-1
D = float(math.trunc(d*nIss))

print C, D
if D > nIss: 
    D = nIss
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

#c = correlation(0.94466163)
c = opt.fmin_cobyla(correlation, [0.1], (), args=(), rhoend = 0.00001)
print "par corr: ", c
print("--- %s seconds ---" % (time.time() - start_time))

# Maybe some plots for the report
#prices = np.zeros(10)
#for k in range(0,10):
#    prices[k] = correlation(0.1*k)
#plt.plot(prices)
#plt.show()
    