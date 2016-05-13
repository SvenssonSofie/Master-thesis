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

############# S24 main 3Y
pi = np.array([0.00087479504671772457, 0.0029353834489070918, 0.004969393415826806, 0.0069769703635795821, 0.0090249736847164419, 0.011068753219014127, 0.013086171126700852, 0.015077371637674086, 0.017108668751384948, 0.019135776533014792, 0.021188431563949561])
spread = np.array([0.0047228532506136903, 0.0048101666897295524, 0.0048253362059923867, 0.0048316350692150416, 0.0048351736249719465, 0.0048374002858278193, 0.0048389188662819015, 0.0048400205026313961, 0.0048408789399059949, 0.0048415530671146135, 0.0048489188205022606])
discFac = np.array([1.0003661802496806, 1.0010318855144935, 1.0017322165435656, 1.0024681809694533, 1.0031665273659054, 1.0037827350418536, 1.0039805436318214, 1.0039247441142243, 1.0039599948710676, 1.0041599686436489, 1.0043651935486468])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25555555555555554])


nIss = 125
coupon = 0.05

recRate2 = 0.4
pr = np.array([631782.070252, 102100.123354, -115046.97444 ])#prices on market 

conRa = 1.38746723906
decay = 2.2953625984
rho = 0.224901644514
C = 0.10
D = 0.30


recRate = 0.0
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
uniAll = np.random.uniform(size = (nbrSim, nIss))#to optimize over the same random numbers, common random numbers

nom = 1000000.0

C = C/(1-recRate2)
D = D/(1-recRate2)

C = math.ceil(C*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(D*nIss)) #nbr of losses detachment
Lavg = (1-recRate)*nom/(D-C)
print C, D, Lavg



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

rep = 100
price = np.zeros(rep)
for j in range(0, rep):
        
    #Generate correlated uniformed random variables 
    normRand1 = np.random.normal(size = (nbrSim, 1))
    normRand2 = np.random.normal(size = (nbrSim, nIss))    
    defTime = np.zeros(nIss)    
        
    pvMean = np.zeros((2,nbrSim))
    defTime=simdeftimes([a, conRa, decay,rho])
    defTime = defTime*sumDelta[-1]
    
    for sim in range(0,nbrSim):
    
        loss=np.sum(np.repeat(defTime[sim,:].reshape(1,nIss),len(sumDelta),0)<=np.repeat(sumDelta.reshape(len(sumDelta),1),nIss,1),1)
           
        ################ tranche loss in each time point
        lossTr=Lavg*np.maximum(np.minimum(loss,D)-C,0)
    
        # sum up discounted losses
        pvDl=1/discFac[0]*lossTr[0]+np.sum(1/discFac[1:T]*(lossTr[1:T] - lossTr[0:T-1]))   
           
        ############ present value premium leg    
        pvPl=coupon*np.sum(delta/discFac*((D-C)*Lavg-lossTr))
         
        pvMean[0, sim] = pvDl
        pvMean[1, sim] = pvPl     
        
    pvMean = np.mean(pvMean,1)
        
    price[j] = pvMean[0]-pvMean[1] 
    print("--- %s seconds ---" % (time.time() - start_time)), j  
    
print 'Present value, PVdl, PVpl: ', pvMean  
print 'Price: ', np.mean(price)
print 'price standard deviation: ', np.std(price) 
print("--- %s seconds ---" % (time.time() - start_time))