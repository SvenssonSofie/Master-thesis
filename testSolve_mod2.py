import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt
import warnings
import scipy.special as spec
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
start_time = time.time()

############# S24 5Y
#pi = np.array([0.00078406209223425005, 0.0029025609697107546, 0.0049936140589693645, 0.0070573752975346959, 0.0091625737396605622, 0.01126330882169857, 0.013336828250815169, 0.015383284673755715, 0.017470830845738194, 0.019553951083291277, 0.021610083873515373, 0.026332554067412195, 0.031136409136407472, 0.035916563070461871, 0.040621557327197189, 0.045303589908513464, 0.050115984541564074, 0.054751681687788323, 0.059381210714061772])
#spread = np.array([0.004836493469134874, 0.0049435372779453425, 0.0049601248471501267, 0.0049668580686195287, 0.0049706006697175092, 0.0049729420073176926, 0.004974532788767738, 0.0049756834329529771, 0.0049765783406867312, 0.004977280256108641, 0.0049778402528445065, 0.005547448502903618, 0.0060345496155683274, 0.0064472591877582164, 0.0067978163208867641, 0.0071023305563781714, 0.0073775257164661887, 0.0076124602657073777, 0.0078237390368579487])
#discFac = np.array([1.0003274275132721, 1.0009892081227751, 1.0016754525209999, 1.0023863451292354, 1.0030604094581885, 1.0036701738001863, 1.0038532663691215, 1.0037706753812285, 1.0037488378459536, 1.0038969889780374, 1.004032911854813, 1.0042120175574905, 1.0042659255614939, 1.0041048481631689, 1.0038701302138373, 1.0035605025093022, 1.0030749067583251, 1.002380718501074, 1.0015862998749887])
#delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25277777777777777, 0.26111111111111113, 0.25277777777777777, 0.25277777777777777])

#nIss = 125
#coupon = 0.01

#recRate2 = 0.4
#tr = np.array([0.0, 0.03, 0.06, 0.12])/(1-recRate2)
#pr = np.array([426604.0, 85594.3, 5825.7])#prices on market 



# Xover S24 5Y
pi = np.array([0.0045969163165063831, 0.016929957100967785, 0.028978612372667967, 0.040749613362203285, 0.052634722354568009, 0.064372574887380463, 0.075839766253202989, 0.087042702835787966, 0.098354240608738985, 0.10952562846757441, 0.12043941717304518, 0.13677406893502542, 0.15315821249913075, 0.1692313828282247, 0.18482970908523089, 0.20013516464756798, 0.21564351953391681, 0.23037042186414414, 0.24483318854606917])
spread = np.array([0.028410302368569135, 0.029040051636199059, 0.029137644999354142, 0.029177266645545812, 0.029199269402997122, 0.029213039864198809, 0.029222398790065638, 0.029229166172768623, 0.029234422458971971, 0.029238549101640702, 0.029241840079856647, 0.030522666546099732, 0.031612704567184229, 0.032533337725092398, 0.033313100248131521, 0.03398868424654531, 0.034597712065773778, 0.035116430341139065, 0.035578919026911168])
discFac = np.array([1.0003274275132721, 1.0009892081227751, 1.0016754525209999, 1.0023863451292354, 1.0030604094581885, 1.0036701738001863, 1.0038532663691215, 1.0037706753812285, 1.0037488378459536, 1.0038969889780374, 1.004032911854813, 1.0042120175574905, 1.0042659255614939, 1.0041048481631689, 1.0038701302138373, 1.0035605025093022, 1.0030749067583251, 1.002380718501074, 1.0015862998749887])
delta = np.array([0.25277777777777777, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25, 0.25555555555555554, 0.25555555555555554, 0.25277777777777777, 0.25277777777777777, 0.26111111111111113, 0.25277777777777777, 0.25277777777777777])
nIss = 75
coupon = 0.05

recRate2 = 0.4
pr = np.array([654168.0, 113513.0, -100989])
tr = np.array([0.0, 0.1, 0.2, 0.35])/(1-recRate2)



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
    if (C != 0) and C != C*nIss:
        C = C-1
    D = float(math.trunc(D*nIss)) #nbr of losses detachment
    if D > nIss: 
        D = nIss
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
c = opt.fsolve(parameters, (0.5,0.5,0.5))
#### print result in parameters tranformed back 
print 'contagion ratio, decay, rho: ', math.exp(c[0]), math.exp(c[1]),math.tanh(c[2])
print("--- %s seconds ---" % (time.time() - start_time))

