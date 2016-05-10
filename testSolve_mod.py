import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt
import warnings
import scipy.special as spec
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
start_time = time.time()

##########Calculates contagion ration for C-D, do this for all tranches and put them into pricingContMod
nIss = 125 #Number of issuers
sumDelta = np.cumsum(delta)
tr = np.array([0.0, 0.03, 0.06, 0.12])
pr = np.array([229269.0, 63815.0, 10265.0])#prices on market 

nom = 1000000.0
sumDelta = np.cumsum(delta)
a = 0.0233509971123/sum(delta)#0.002#0.012/0.25
T = len(delta)
coupon = 0.01
recRate = 0.0


nbrSim = 1000


#Generate correlated normal random variables 
normRand1 = np.random.normal(size = (nbrSim,1))
# obs storlek ska var (nbrSim,1) och samma för alla företag  för varje simul
normRand2 = np.random.normal(size = (nbrSim, nIss))

Lavg = 1000000.0/15.0 # fattar inte detta, varför 15
def f(tau, Ek, c, d, k): #Want to solve F(tau) = 0
    
    if tau < defTime[k-1]:#+ 1: 
        return float('inf')

    ftau = 0.0
    for i in range(k):
        ftau = ftau + 1-np.exp(-d*(tau - defTime[i]))
        #print ftau
    
    ftau = ftau*a*c/d + a*tau
    #print ftau - Ek
    return abs(ftau - Ek)


def parameters(p):     
    return (contagion(p, tr[0], tr[1], pr[0]),contagion(p, tr[1], tr[2], pr[1]),contagion(p, tr[2], tr[3], pr[2]))

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
 return tau

def contagion(p, C, D, marketPrice):
    #nom = (D-C)/0.03*1000000.0
    ##### transform parameters so that they fit into required regions
    conRa = math.exp(p[0])
    decay = math.exp(p[1])
    rho = math.tanh(p[2])
    
    # From percentage to number of defaults
    C = math.ceil(C*nIss)
    if (C != 0):
        C = C-1
    D = float(math.trunc(D*nIss)) #nbr of losses detachment
    #Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
    print conRa,decay,rho, C, D
    
    pvMean = np.zeros((2,nbrSim))
   
    ## simulate default times
    defTime=simdeftimes([a, conRa, decay,rho])
    
    for sim in range(0,nbrSim):
        
        
        #find loss for each time step using summation of logic matrices
        
        loss=np.sum(np.repeat(defTime[sim,:].reshape(1,nIss),len(sumDelta),0)<=np.repeat(sumDelta.reshape(len(sumDelta),1),nIss,1),1)
        #print loss
       
        ################ tranche loss in each time point
        lossTr=Lavg*np.maximum(np.minimum(loss,D)-C,0)
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
c = opt.fsolve(parameters, (-2,0.03,0.07))
#### print result in parameters tranformed back 
print 'True ratio: ', math.exp(c[0]), math.exp(c[1]),math.tanh(c[2])
print("--- %s seconds ---" % (time.time() - start_time))

