import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt

start_time = time.time()
############# S24 5Y
#pi = np.array([0.00078406209223425005, 0.0029025609697107546, 0.0049936140589693645, 0.0070573752975346959, 0.0091625737396605622, 0.01126330882169857, 0.013336828250815169, 0.015383284673755715, 0.017470830845738194, 0.019553951083291277, 0.021610083873515373, 0.026332554067412195, 0.031136409136407472, 0.035916563070461871, 0.040621557327197189, 0.045303589908513464, 0.050115984541564074, 0.054751681687788323, 0.059381210714061772])
#spread = np.array([0.004836493469134874, 0.0049435372779453425, 0.0049601248471501267, 0.0049668580686195287, 0.0049706006697175092, 0.0049729420073176926, 0.004974532788767738, 0.0049756834329529771, 0.0049765783406867312, 0.004977280256108641, 0.0049778402528445065, 0.005547448502903618, 0.0060345496155683274, 0.0064472591877582164, 0.0067978163208867641, 0.0071023305563781714, 0.0073775257164661887, 0.0076124602657073777, 0.0078237390368579487])
#discFac = np.array([1.0003274275132721, 1.0009892081227751, 1.0016754525209999, 1.0023863451292354, 1.0030604094581885, 1.0036701738001863, 1.0038532663691215, 1.0037706753812285, 1.0037488378459536, 1.0038969889780374, 1.004032911854813, 1.0042120175574905, 1.0042659255614939, 1.0041048481631689, 1.0038701302138373, 1.0035605025093022, 1.0030749067583251, 1.002380718501074, 1.0015862998749887])
#delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25277777777777777, 0.26111111111111113, 0.25277777777777777, 0.25277777777777777])
#nIss = 125
#coupon = 0.01

#C1 = 0.00
#D = 0.03
#marketPrice = 426604.0
#C1 = 0.03
#D = 0.06
#marketPrice = 85594.0
#C1 = 0.06
#D = 0.12
#marketPrice = 5826
#C1 = 0.12
#D = 1.0
#marketPrice = -30899.0




# Xover S24 5Y
pi = np.array([0.0045969163165063831, 0.016929957100967785, 0.028978612372667967, 0.040749613362203285, 0.052634722354568009, 0.064372574887380463, 0.075839766253202989, 0.087042702835787966, 0.098354240608738985, 0.10952562846757441, 0.12043941717304518, 0.13677406893502542, 0.15315821249913075, 0.1692313828282247, 0.18482970908523089, 0.20013516464756798, 0.21564351953391681, 0.23037042186414414, 0.24483318854606917])
spread = np.array([0.028410302368569135, 0.029040051636199059, 0.029137644999354142, 0.029177266645545812, 0.029199269402997122, 0.029213039864198809, 0.029222398790065638, 0.029229166172768623, 0.029234422458971971, 0.029238549101640702, 0.029241840079856647, 0.030522666546099732, 0.031612704567184229, 0.032533337725092398, 0.033313100248131521, 0.03398868424654531, 0.034597712065773778, 0.035116430341139065, 0.035578919026911168])
discFac = np.array([1.0003274275132721, 1.0009892081227751, 1.0016754525209999, 1.0023863451292354, 1.0030604094581885, 1.0036701738001863, 1.0038532663691215, 1.0037706753812285, 1.0037488378459536, 1.0038969889780374, 1.004032911854813, 1.0042120175574905, 1.0042659255614939, 1.0041048481631689, 1.0038701302138373, 1.0035605025093022, 1.0030749067583251, 1.002380718501074, 1.0015862998749887])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25277777777777777, 0.26111111111111113, 0.25277777777777777, 0.25277777777777777])
nIss = 75
coupon = 0.05

#C1 = 0.0
#D = 0.1
#marketPrice = 654168.0
#C1 = 0.1
#D = 0.2
#marketPrice = 113513.0
C1 = 0.2
D = 0.35
marketPrice = -100989


recRate = 0.0
sumDelta = np.cumsum(delta)


nom = 1000000.0

T = len(delta)


a = np.zeros(len(spread))
for k in range(0, len(spread)): 
    if k == 0: 
        a[k] = -(1-pi[k]-1)/delta[k]
    else:
        a[k] = -(1 - pi[k] - 1 + pi[k-1])/(1-pi[k-1])/delta[k-1]
print a
print np.mean(a)
a = np.mean(a)

#generate uniform standard random values
nbrSim = 10000
uniAll = np.random.uniform(size = (nbrSim, nIss))#to optimize over the same random numbers, common random numbers

recRate2 = 0.4
C1 = C1/(1-recRate2)
D = D/(1-recRate2)
print C1, D

C = math.ceil(C1*nIss) #nbr of losses attachment
print C
if C != 0 and C != C1*nIss: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(D*nIss)) #nbr of losses detachment
if D > nIss: 
    D = nIss
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
print C, D, Lavg

def contagion(c):
    print c
    if c < 0:
        return float('inf')
    
    pvMean = np.zeros((2,nbrSim))
    for sim in range(0,nbrSim):
        uni = uniAll[sim,:]
        e = -np.log(uni)
        eSort = np.sort(e)
        #print eSort
        defTime = np.zeros(nIss)
        defTime[0] = eSort[0]/a
        for nbr in range(1,len(defTime)):             
            defTime[nbr] = defTime[nbr - 1] + (eSort[nbr] - eSort[nbr - 1])/(a*(1+nbr*c))
        defTime = defTime*sumDelta[-1] #obs, default time i procent, kanske
        
        ##print defTime
        loss = np.zeros(T)#if no value is inserted on a position, then all has defaulted before that timeStep and thus should the loss of the following time steps be 100%
        for t in range(0,T):
            if sumDelta[t] < defTime[0]: continue
            for tau in range(1,len(defTime)):
                #print tau
                if defTime[tau] > sumDelta[t] and defTime[tau - 1] < sumDelta[t]: 
                    loss[t] = tau
                    break
                else: 
                    loss[t] = nIss  
               
        #print 'loss', loss
        ################ tranche loss
        lossTr = np.zeros(T) #add rows to calculate tranches simultaneously
        for t in range(0,T):
            if loss[t] <= C: continue
            elif loss[t] > C and loss[t] <= D: 
                lossTr[t] = (loss[t] - C)*Lavg
                
            else: 
                lossTr[t] = (D-C)*Lavg
        ################Present value defualt leg    
        pvDl = 0.0 
        for t in range(0,T):
            if t == 0:
                pvDl = pvDl + 1/discFac[t]*lossTr[t]
            else:
                pvDl = pvDl + 1/discFac[t]*(lossTr[t] - lossTr[t-1])    

        ############ present value premium leg    
        pvPl = 0.0
        for t in range(0,T):
            pvPl = pvPl + delta[t]/discFac[t]*((D-C)*Lavg-lossTr[t])        
        pvPl = coupon*pvPl

        pvPl=coupon*np.sum(delta/discFac*((D-C)*Lavg-lossTr))
        
        pvMean[0, sim] = pvDl
        pvMean[1, sim] = pvPl

    
    pvMean = np.mean(pvMean,axis=1)

    print 'Present value, PVdl, PVpl: ', pvMean  
    print pvMean[0] - pvMean[1]
    return abs(pvMean[0] - pvMean[1] - marketPrice)

#contRatio = contagion(0.4)
c = opt.fmin_cobyla(contagion, [1.6], (), args=(), rhoend = 0.00001)
print 'True ratio: ', c
print("--- %s seconds ---" % (time.time() - start_time))