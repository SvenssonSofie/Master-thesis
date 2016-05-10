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
coupon = 0.01
discFac = np.array([1.00047630826       , 1.00112395119       ,\
          1.00180019558       ,  1.00252355848       , 1.00320411939       , 1.0036842191        ,\
           1.00377352897       , 1.00368518481       , 1.00372980688       , 1.0038363132        ,\
          1.00395370531       ,  1.0040649568        , 1.00392689128       , 1.00358551835       , \
          1.00315265354    , 1.00262525655    , 1.0018353653        ,    1.00087710692   ,0.999799458336])


spread = np.array([0.0039230364813, 0.00396739004003, 0.00397766902931, 0.00398224178954, \
      0.00398489559746, 0.00398659905109, 0.00398777551688, 0.00398863618887, 0.00398931119962,\
     0.00398984448104, 0.00399027231814, 0.00454277200564, 0.00501692809448,  0.0054198805189, \
     0.00576302421683, 0.00606175067052, 0.00633221440352,0.00656349712714, 0.00677161234034])


delta = np.array([0.252777777778, 0.255555555556, 0.252777777778,      \
         0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, \
              0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
              0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778])

tr = np.array([0.0, 0.03, 0.06, 0.12])
pr = np.array([388889.446134, 84493.4899999, 5476.81999264])#prices on market 

############# xover 
pi = np.array([0.0054613538589327604, 0.017607791695479436, 0.029476260356662176, 0.041073269614748997, 0.052784774205969498, 0.064353244574194668, 0.075656972454807292, 0.086702157863696949, 0.097856390541134042, 0.10887439503442109, 0.11964025435853443, 0.13563164728614596, 0.15167821333151854, 0.1674268829059713, 0.18271673524872123, 0.19772579593502271, 0.21294030436312905, 0.22739432201532073, 0.24159546220851535])
spread = np.array([0.028147118285289249, 0.028634939976430237, 0.028723255753864623, 0.028760259933442979, 0.028781104758606655, 0.028794264502661968, 0.028803257244908714, 0.028809783040567572, 0.02881486421967942, 0.028818864331903976, 0.028822059381286458, 0.030048444153557351, 0.031093429118851375, 0.031977030520288628, 0.032726160277256104, 0.033375748224999592, 0.033961755674568911, 0.034461192631016067, 0.034906824184687317])
discFac = np.array([1.0003635153416293, 1.0010199296089801, 1.0017009237372976, 1.0024044576014735, 1.0030619651536226, 1.0035857631689056, 1.0036885676120733, 1.0035626340471684, 1.0035425703254608, 1.0036384904666531, 1.0037402530029267, 1.0038644452392391, 1.0038167791152117, 1.0035736824892858, 1.0032519368338813, 1.0028494417398197, 1.0022296412972789, 1.0014260482045545, 1.000516346531535])
delta = np.array([0.252777777778, 0.255555555556, 0.252777777778,      \
         0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, \
              0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
              0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778])



C=0.0
D =0.10
marketPrice = 631782.070252
nIss = 75
coupon = 0.05

##C = 0.10
#D = 0.20
#marketPrice = 102100.123354

tr = np.array([0.0, 0.1, 0.2, 0.35])
pr = np.array([631782.070252, 102100.123354, -115046.97444])#prices on market 


sumDelta = np.cumsum(delta)
recRate = 0.4

nom = 1000000.0
sumDelta = np.cumsum(delta)
#d = np.mean(delta)
#pi = np.zeros(len(spread))
#for s in range(0,len(spread)):
#    pi[s] = 1 - math.exp(-spread[s]*sumDelta[s])
    
#a = np.zeros(len(pi))
#for k in range(0, len(pi)):
#    a[k] = 365.0/360.0*spread[k]/(1-recRate + spread[k]*d/2.0)
#a = np.mean(a)
#print a
#a = 0.00235886981172
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

nbrSim = 1000

#Generate correlated normal random variables 
normRand1 = np.random.normal(size = (nbrSim,1))
# obs storlek ska var (nbrSim,1) och samma for alla foretag  for varje simul
normRand2 = np.random.normal(size = (nbrSim, nIss))



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
c = opt.fsolve(parameters, (2,0.03,0.07))
#### print result in parameters tranformed back 
print 'True ratio: ', math.exp(c[0]), math.exp(c[1]),math.tanh(c[2])
print("--- %s seconds ---" % (time.time() - start_time))

