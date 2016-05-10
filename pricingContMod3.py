import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt
import warnings
import scipy.special as spec
#from mahotas.demos.surf_gaussians import rho
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

start_time = time.time()

##########Calculates contagion ration for C-D, do this for all tranches and put them into pricingContMod
nIss = 125.0#Number of issuers
delta = [0.252777777778, 0.255555555556, 0.252777777778,      \
         0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, \
              0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
              0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778]

discFac = np.array([1.00047630826       , 1.00112395119       ,\
          1.00180019558       ,  1.00252355848       , 1.00320411939       , 1.0036842191        ,\
           1.00377352897       , 1.00368518481       , 1.00372980688       , 1.0038363132        ,\
          1.00395370531       ,  1.0040649568        , 1.00392689128       , 1.00358551835       , \
          1.00315265354    , 1.00262525655    , 1.0018353653        ,    1.00087710692   ,0.999799458336])


spread = [0.0039230364813, 0.00396739004003, 0.00397766902931, 0.00398224178954, \
      0.00398489559746, 0.00398659905109, 0.00398777551688, 0.00398863618887, 0.00398931119962,\
     0.00398984448104, 0.00399027231814, 0.00454277200564, 0.00501692809448,  0.0054198805189, \
     0.00576302421683, 0.00606175067052, 0.00633221440352,0.00656349712714, 0.00677161234034]


sumDelta = np.cumsum(delta)

C = 0.00
D = 0.03 
conRa = 0.0000000503733948919
decay = 0.0803633950326
rho = 0.540023195865



recRate = 0.0
sumDelta = np.cumsum(delta)
d = np.mean(delta)
a = np.zeros(len(spread))
for k in range(0, len(spread)):
    a[k] = 365.0/360.0*spread[k]/(1-recRate + spread[k]*d/2.0)
a = np.mean(a)
print a

T = len(delta)
nbrSim = 100000
uniAll = np.random.uniform(size = (nbrSim, nIss))#to optimize over the same random numbers, common random numbers
coupon = 0.01
nom = 1000000.0


C = math.ceil(C*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(D*nIss)) #nbr of losses detachment
Lavg = (1-recRate)*nom/(D-C)
print C, D, Lavg

#Generate correlated uniformed random variables 
normRand1 = np.random.normal(size = (nbrSim, 1))#nIss))
normRand2 = np.random.normal(size = (nbrSim, nIss))


defTime = np.zeros(nIss)
'''
def f(tau, Ek, c, d, k, prevTau):

    if tau < prevTau: 
        return float('inf')

    ftau = 0.0
    for i in range(k):
        ftau = ftau + a*c/d*(1-np.exp(-d*(prevTau - tau)))
    ftau = ftau + a*prevTau
    
    return ftau - Ek
'''
def solvfun(x,y,z): # help function to simulate default times
    return np.real((x*z-y*z+spec.lambertw(y*z*np.exp(-x*z+y*z)))/z);

def simdeftimes(par): # efficient simuation of default times
    a=par[0]
    c=par[1]
    d=par[2]
    rho=par[3]
    tau=np.zeros((nbrSim,nIss))
    # form correlated uniform random var
    U=sct.norm.cdf(rho*np.tile(normRand1,(1,nIss))+math.sqrt(1-rho*rho)*normRand2)
    # form correlated exponential random var
    E=np.sort(-np.log(U),axis=1)
    # solve int(lambda)=Ek k=0,...,nIss-1
    tau[:,0]=E[:,0]/a
    for i in range(1,int(nIss)):
        x=(E[:,i]-E[:,i-1])/a
        y=c*np.exp(-d*tau[:,i-1])*np.sum(np.exp(d*tau[:,0:i]),axis=1)/d
        tau[:,i]=tau[:,i-1]+solvfun(x,y,d)
    #print tau
    return tau

    
pvMean = np.zeros((2,nbrSim))
defTime=simdeftimes([a, conRa, decay,rho])
for sim in range(0,nbrSim):

    # Generate correlated uniform variables
    #normRand = rho*normRand1[sim,:] + math.sqrt(1 - rho*rho)*normRand2[sim,:]
    #randCDF = sct.norm.cdf(normRand)
    #uniRand = sct.uniform.ppf(randCDF)
    #e = -np.log(uniRand)
    #eSort = np.sort(e)                
        
    #defTime[0] = eSort[0]/a
    #for nbr in range(1,len(uniRand)):
    #    Ek = eSort[nbr]
    #    defTime[nbr] = opt.fmin(f,defTime[nbr -1], args = (Ek, conRa, decay, nbr, defTime[nbr -1]), disp = False)#, maxfun = 10000000, maxiter = 1000000)# det Tau som uppfyller F(tau) = 0 for detta d
    
    
    #print defTime
    '''
    loss = np.zeros(T)#if no value is inserted on a position, then all has defaulted before that timeStep and thus should the loss of the following time steps be 100%
    for t in range(0,T):
        if sumDelta[t] < defTime[0]: continue
        for tau in range(1,len(defTime)):
            #print tau
            if defTime[tau] > sumDelta[t] and defTime[tau - 1] < sumDelta[t]: 
                loss[t] = tau#0/nIss
                break
            else: 
                loss[t] = 125.0    
    
    ################ tranche loss
    lossTr = np.zeros(T) #add rows to calculate tranches simultaneously
    for t in range(0,T):
        if loss[t] <= C: continue
        elif loss[t] > C and loss[t] <= D: 
            lossTr[t] = (loss[t] - C)*Lavg
        else: 
            lossTr[t] = (D-C)*Lavg
    #print lossTr #
     
    pvDl = 0.0 
    for t in range(0,T):
        if t == 0:
            pvDl = pvDl + 1/discFac[t]*lossTr[t]
        else:
            pvDl = pvDl + 1/discFac[t]*(lossTr[t] - lossTr[t-1])    
    #print pvDl
     
    ############ present value premium leg    
    pvPl = 0.0
    for t in range(0,T):
        pvPl = pvPl + delta[t]/discFac[t]*((D-C)*Lavg-lossTr[t])        
    pvPl = coupon*pvPl
    #print pvPl
    '''
    loss=np.sum(np.repeat(defTime[sim,:].reshape(1,nIss),len(sumDelta),0)<=np.repeat(sumDelta.reshape(len(sumDelta),1),nIss,1),1)
    #print 'loss',loss
       
    ################ tranche loss in each time point
    lossTr=Lavg*np.maximum(np.minimum(loss,D)-C,0)
    #print lossTr
    # sum up discounted losses
    pvDl=1/discFac[0]*lossTr[0]+np.sum(1/discFac[1:T]*(lossTr[1:T] - lossTr[0:T-1])) 
       
       
    ############ present value premium leg    
    pvPl=coupon*np.sum(delta/discFac*((D-C)*Lavg-lossTr))
     
    pvMean[0, sim] = pvDl
    pvMean[1, sim] = pvPl     
    
pvMean = np.mean(pvMean,1)
    
print 'Present value, PVdl, PVpl: ', pvMean  
print 'Price: ', (pvMean[0] - pvMean[1]) 
print("--- %s seconds ---" % (time.time() - start_time))