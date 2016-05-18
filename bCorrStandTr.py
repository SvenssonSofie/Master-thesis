import math
import numpy as np
import scipy.stats as sct
import scipy.optimize as opt
import time
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

################################Finding correlation for 0-detachment for all standard tranches
start_time = time.time()
############# S24 5Y
#pi = np.array([0.00078406209223425005, 0.0029025609697107546, 0.0049936140589693645, 0.0070573752975346959, 0.0091625737396605622, 0.01126330882169857, 0.013336828250815169, 0.015383284673755715, 0.017470830845738194, 0.019553951083291277, 0.021610083873515373, 0.026332554067412195, 0.031136409136407472, 0.035916563070461871, 0.040621557327197189, 0.045303589908513464, 0.050115984541564074, 0.054751681687788323, 0.059381210714061772])
#spread = np.array([0.004836493469134874, 0.0049435372779453425, 0.0049601248471501267, 0.0049668580686195287, 0.0049706006697175092, 0.0049729420073176926, 0.004974532788767738, 0.0049756834329529771, 0.0049765783406867312, 0.004977280256108641, 0.0049778402528445065, 0.005547448502903618, 0.0060345496155683274, 0.0064472591877582164, 0.0067978163208867641, 0.0071023305563781714, 0.0073775257164661887, 0.0076124602657073777, 0.0078237390368579487])
#discFac = np.array([1.0003274275132721, 1.0009892081227751, 1.0016754525209999, 1.0023863451292354, 1.0030604094581885, 1.0036701738001863, 1.0038532663691215, 1.0037706753812285, 1.0037488378459536, 1.0038969889780374, 1.004032911854813, 1.0042120175574905, 1.0042659255614939, 1.0041048481631689, 1.0038701302138373, 1.0035605025093022, 1.0030749067583251, 1.002380718501074, 1.0015862998749887])
#delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25277777777777777, 0.26111111111111113, 0.25277777777777777, 0.25277777777777777])

#nIss = 125


#recRate2 = 0.4
#tr = np.array([0.0, 0.03, 0.06, 0.12, 1.0])/(1-recRate2)
#pr = [426604.0, 85594.0, 5826.0, -30899.0]

# omstart compcorr max 125 
#comp = [0.67738816]


# Xover S24 5Y
pi = np.array([0.0045969163165063831, 0.016929957100967785, 0.028978612372667967, 0.040749613362203285, 0.052634722354568009, 0.064372574887380463, 0.075839766253202989, 0.087042702835787966, 0.098354240608738985, 0.10952562846757441, 0.12043941717304518, 0.13677406893502542, 0.15315821249913075, 0.1692313828282247, 0.18482970908523089, 0.20013516464756798, 0.21564351953391681, 0.23037042186414414, 0.24483318854606917])
spread = np.array([0.028410302368569135, 0.029040051636199059, 0.029137644999354142, 0.029177266645545812, 0.029199269402997122, 0.029213039864198809, 0.029222398790065638, 0.029229166172768623, 0.029234422458971971, 0.029238549101640702, 0.029241840079856647, 0.030522666546099732, 0.031612704567184229, 0.032533337725092398, 0.033313100248131521, 0.03398868424654531, 0.034597712065773778, 0.035116430341139065, 0.035578919026911168])
discFac = np.array([1.0003274275132721, 1.0009892081227751, 1.0016754525209999, 1.0023863451292354, 1.0030604094581885, 1.0036701738001863, 1.0038532663691215, 1.0037706753812285, 1.0037488378459536, 1.0038969889780374, 1.004032911854813, 1.0042120175574905, 1.0042659255614939, 1.0041048481631689, 1.0038701302138373, 1.0035605025093022, 1.0030749067583251, 1.002380718501074, 1.0015862998749887])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25277777777777777, 0.26111111111111113, 0.25277777777777777, 0.25277777777777777])

recRate2 = 0.4
nIss = 75
comp = [0.56695777]
tr = np.array([0.0, 0.10, 0.20, 0.35])/(1-recRate2)
pr = [654168.0, 113513.0, -100989.0]







coupon = np.array([0.05, 0.05, 0.05])



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
    print corr
    if corr >= 1.0 or corr < 0:
        return float('inf')
    
    p = findp(corr)
    pvDl = defaultLeg(p)
    pvPl = premiumLeg(spread, p)
    print pvDl - pvPl
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
    if D > nIss: 
        D = nIss  
    C = math.ceil(tr[k]*nIss) #For market premium we need tranche with other attachment than zero
    if C != 0 and C != tr[k]*nIss:
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
    print pr[k], priceLow, priceUpp
    
    #find correlation given price and premium
    C = 0
    D = float(math.trunc(tr[k+1]*nIss)) #nbr of losses detachment
    if D > nIss: 
        D = nIss
    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-
    print C, D
    corr[k] = compoundCorr(pr[k], coupon[k])

    print corr
print("--- %s seconds ---" % (time.time() - start_time))


