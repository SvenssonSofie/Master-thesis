import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt
import warnings
import scipy.special as spec
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
start_time = time.time()

############# S24 main 3Y
pi = np.array([0.00087479504671772457, 0.0029353834489070918, 0.004969393415826806, 0.0069769703635795821, 0.0090249736847164419, 0.011068753219014127, 0.013086171126700852, 0.015077371637674086, 0.017108668751384948, 0.019135776533014792, 0.021188431563949561])
spread = np.array([0.0047228532506136903, 0.0048101666897295524, 0.0048253362059923867, 0.0048316350692150416, 0.0048351736249719465, 0.0048374002858278193, 0.0048389188662819015, 0.0048400205026313961, 0.0048408789399059949, 0.0048415530671146135, 0.0048489188205022606])
discFac = np.array([1.0003661802496806, 1.0010318855144935, 1.0017322165435656, 1.0024681809694533, 1.0031665273659054, 1.0037827350418536, 1.0039805436318214, 1.0039247441142243, 1.0039599948710676, 1.0041599686436489, 1.0043651935486468])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25555555555555554])


nIss = 125
coupon = 0.01

recRate2 = 0.4
tr = np.array([0.0, 0.03, 0.06, 0.12, 1.0])/(1-recRate2)
pr = np.array([209314.0, 6965.0, -17076.0, -23219.0])#prices on market 


sumDelta = np.cumsum(delta)
recRate = 0.0

nom = 1000000.0
sumDelta = np.cumsum(delta)

a = np.zeros(len(pi))
for k in range(0, len(spread)): 
    if k == 0: 
        a[k] = -(1-pi[k]-1)/delta[k]
    else:
        a[k] = -(1 - pi[k] - 1 + pi[k-1])/(1-pi[k-1])/delta[k-1]
print a
print np.mean(a)
a = np.mean(a)


T = len(delta)

nbrSim = 10000

#Generate correlated normal random variables 
normRand1 = np.random.normal(size = (nbrSim,1))
# obs storlek ska var (nbrSim,1) och samma for alla foretag  for varje simul
normRand2 = np.random.normal(size = (nbrSim, nIss))



def parameters(p):     
    return (contagion(p, tr[0], tr[1], pr[0]),contagion(p, tr[1], tr[2], pr[1]),contagion(p, tr[2], tr[3], pr[2]), contagion(p, tr[3], tr[4], pr[3]))

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
    for i in range(1,nIss):
        x=(E[:,i]-E[:,i-1])/a
        y=c*np.exp(-d*tau[:,i-1])*np.sum(np.exp(d*tau[:,0:i]),axis=1)/d
        tau[:,i]=tau[:,i-1]+solvfun(x,y,d)
    #print tau
    return tau

def contagion(p, C, D, marketPrice):
    ##### transform parameters so that they fit into required regions
    conRa = math.exp(p[0])
    decay = math.exp(p[1])
    rho = math.tanh(p[2])

    # From percentage to number of defaults
    C = math.ceil(C*nIss)
    if (C != 0):
        C = C-1
    D = float(math.trunc(D*nIss)) #nbr of losses detachment
    Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    print conRa,decay,rho, C, D, Lavg
    
    pvMean = np.zeros((2,nbrSim))
   
    ## simulate default times
    defTime=simdeftimes([a, conRa, decay,rho])
    #print defTime
    defTime = defTime*sumDelta[-1]
    
    for sim in range(0,nbrSim):
        
        
        #find loss for each time step using summation of logic matrices
        
        loss=np.sum(np.repeat(defTime[sim,:].reshape(1,nIss),len(sumDelta),0)<=np.repeat(sumDelta.reshape(len(sumDelta),1),nIss,1),1)
        #print 'loss',loss
       
        ################ tranche loss in each time point
        lossTr=Lavg*np.maximum(np.minimum(loss,D)-C,0)
        #print lossTr
        # sum up discounted losses
        pvDl=1/discFac[0]*lossTr[0]+np.sum(1/discFac[1:T]*(lossTr[1:T] - lossTr[0:T-1])) 
       
       
        ############ present value premium leg    
        pvPl=coupon*np.sum(delta/discFac*((D-C)*Lavg-lossTr))
        #print pvPl
        pvMean[0, sim] = pvDl
        pvMean[1, sim] = pvPl
        
    
    pvMean = np.mean(pvMean,1)#/nbrSim
    
    print 'Present value, PVdl, PVpl: ', pvMean  
    print (pvMean[0] - pvMean[1])# - marketPrice  
    return abs((pvMean[0] - pvMean[1]) - marketPrice)

#contRatio = contagion(0.3, 0.3, 0.3)
c = opt.fsolve(parameters, (0.5,0.5,0.5, 0.5))
#### print result in parameters tranformed back 
print 'contagion ratio, decay, rho: ', math.exp(c[0]), math.exp(c[1]),math.tanh(c[2])
print("--- %s seconds ---" % (time.time() - start_time))

