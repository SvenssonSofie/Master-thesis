import math
import numpy as np
import scipy.stats as sct
import time 
import scipy.optimize as opt

start_time = time.time()
##########Calculates contagion ration for C-D, do this for all tranches and put them into pricingContMod

marketPrice = 388889.446134
C = 0.0
D = 0.03

#marketPrice = 84493.4899999
#C = 0.03
#D = 0.06

#marketPrice = 5476.81999264
#C = 0.06
#D = 0.12

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


#### 9 majy38322, 1.00102760219, 1.00170852363, 1.00241212434,1.0030695185, 1.00359297642, 1.00369484159, 1.00356789028, 1.00354717324, 1.00364271423, 1.00374418093, 1.00386815397, 1.00382014537, 1.00357659785, 1.0032544063, 1.0028514655, 1.00223112542, 1.00142694292, 1.00051665287])
#print len(delta), len(spread), len(discFac)
#marketPrice = -1572.1921378
#C = 0.06
#D = 0.12

#C = 0.0
#D = 0.03
#marketPrice = 406586.239243

### 10 maj

discFac = np.array([1.0003635153416293, 1.0010199296089801, 1.0017009237372976, 1.0024044576014735, 1.0030619651536226, 1.0035857631689056, 1.0036885676120733, 1.0035626340471684, 1.0035425703254608, 1.0036384904666531, 1.0037402530029267, 1.0038644452392391, 1.0038167791152117, 1.0035736824892858, 1.0032519368338813, 1.0028494417398197, 1.0022296412972789, 1.0014260482045545, 1.000516346531535])
spread = np.array([0.0045244341749098834, 0.0046027012776632429, 0.0046168697365552604, 0.0046228051609962933, 0.0046261520984201077, 0.0046282640722665226, 0.004629706789672363, 0.0046307540958324912, 0.0046315707195910133, 0.0046322129058806545, 0.0046327260745146577, 0.0051941879063954243, 0.0056748678580730676, 0.0060825418584728808, 0.0064291123565602994, 0.0067303788304509445, 0.0070028012033693026, 0.007235496127153785, 0.0074448698161273434])
pi = np.array([0.00087988747682610224, 0.0028514524517694317, 0.004797760055717859, 0.0067189430820959739, 0.0086789858412678056, 0.010635160845597347, 0.012566275637129642, 0.014472461972708572, 0.016417204703299659, 0.018358109870613926, 0.020274150444318573, 0.024840910419199558, 0.029487153021399259, 0.034111258137983769, 0.038663426774489862, 0.043194141346570847, 0.04785180082622742, 0.052339211209776693, 0.056821960151918427])
marketPrice = -1612.42055845
C = 0.06
D = 0.12

marketPrice = 69276.9371794
C = 0.03
D = 0.06

#marketPrice = 406421.037154
#C = 0.0
#D = 0.03

nIss = 125
coupon = 0.01
recRate = 0.4

############# xover 
#pi = np.array([0.0054613538589327604, 0.017607791695479436, 0.029476260356662176, 0.041073269614748997, 0.052784774205969498, 0.064353244574194668, 0.075656972454807292, 0.086702157863696949, 0.097856390541134042, 0.10887439503442109, 0.11964025435853443, 0.13563164728614596, 0.15167821333151854, 0.1674268829059713, 0.18271673524872123, 0.19772579593502271, 0.21294030436312905, 0.22739432201532073, 0.24159546220851535])
#spread = np.array([0.028147118285289249, 0.028634939976430237, 0.028723255753864623, 0.028760259933442979, 0.028781104758606655, 0.028794264502661968, 0.028803257244908714, 0.028809783040567572, 0.02881486421967942, 0.028818864331903976, 0.028822059381286458, 0.030048444153557351, 0.031093429118851375, 0.031977030520288628, 0.032726160277256104, 0.033375748224999592, 0.033961755674568911, 0.034461192631016067, 0.034906824184687317])
#discFac = np.array([1.0003635153416293, 1.0010199296089801, 1.0017009237372976, 1.0024044576014735, 1.0030619651536226, 1.0035857631689056, 1.0036885676120733, 1.0035626340471684, 1.0035425703254608, 1.0036384904666531, 1.0037402530029267, 1.0038644452392391, 1.0038167791152117, 1.0035736824892858, 1.0032519368338813, 1.0028494417398197, 1.0022296412972789, 1.0014260482045545, 1.000516346531535])
#delta = np.array([0.252777777778, 0.255555555556, 0.252777777778,      \
#         0.25, 0.255555555556, 0.255555555556, 0.252777777778, 0.25, \
#              0.255555555556, 0.255555555556 , 0.252777777778 , 0.25, 0.255555555556, \
#              0.255555555556 , 0.252777777778, 0.252777777778, 0.261111111111 , 0.252777777778, 0.252777777778])



#C=0.0
#D =0.10
#marketPrice = 631782.070252
##nIss = 75
#coupon = 0.05
#
#C = 0.10
#D = 0.20
#marketPrice = 102100.123354

#C = 0.20
#D = 0.35
#marketPrice = -115046.97444


recRate = 0.4
sumDelta = np.cumsum(delta)

#pi = np.zeros(len(spread))
#for s in range(0,len(spread)):
#    pi[s] = (1 - math.exp(-spread[s]*sumDelta[s]))/(1-recRate)#/marketPrice
#print pi
#pi = spread/0.6
#print spread/0.6
#print np.mean(pi)

#nIss = 125.0#Number of issuers

#recRate = 0.0

nom = 1000000.0
#coupon = 0.01
T = len(delta)


###find default prob
#s1 = 0.0 #default 
#s2 = 0.0
#for k in range(0, len(delta)):
#    if k == 0: 
#        s1 = (1-recRate)/discFac[k]*(1-denvivillhitta)



#d = np.mean(delta)
a = np.zeros(len(spread))
#for k in range(0, len(spread)):
#    a[k] = (365.0/360.0)*spread[k]/(1-recRate + spread[k]*d/2.0)#/sumDelta[k]
#a = np.mean(a)
#a = 0.003
#print a
#sumLambda = 0.0
#for k in range(0, len(spread)):
##    a[k] =  (-np.log(1-pi[k]) - sumLambda)/delta[k]
#   sumLambda = sumLambda + a[k]*delta[k] 
#a = np.mean(a)*d
#print a

#a = np.mean(pi*delta)
#print a 

#a[0] = pi[0]/delta[0]
#for k in range(1, len(pi)): 
#    a[k] = (pi[k]- pi[k-1])/delta[k]
#print np.mean(a)
#print a
#a = np.mean(a)

for k in range(0, len(spread)): 
    if k == 0: 
        a[k] = -(1-pi[k]-1)/delta[k]
    else:
        a[k] = -(1 - pi[k] - 1 + pi[k-1])/(1-pi[k-1])/delta[k-1]
print a
print np.mean(a)
a = np.mean(a)

#generate uniform standard random values
nbrSim = 1000
uniAll = np.random.uniform(size = (nbrSim, nIss))#to optimize over the same random numbers, common random numbers

C = math.ceil(C*nIss) #nbr of losses attachment
if C != 0: #if we want to insure 12-15, C should be 11
    C = C-1
D = float(math.trunc(D*nIss)) #nbr of losses detachment
Lavg = (1.0-recRate)*nom/(D-C)#hela nominalen ska va avksriven inom omradet C-D
print C, D, Lavg
#print sumDelta
#sumDelta = sumDelta/sumDelta[-1]
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
            #print 1.0/(a*(1+nbr*c))
        #print defTime
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
               
        #print 'deftim', defTime
        #loss=np.sum(np.repeat(defTime.reshape(1,nIss),len(sumDelta),0)<=np.repeat(sumDelta.reshape(len(sumDelta),1),nIss,1),1)
        #print 'loss:'
        #print loss
        #print ''
        #print 'loss', loss
        ################ tranche loss
        lossTr = np.zeros(T) #add rows to calculate tranches simultaneously
        for t in range(0,T):
            if loss[t] <= C: continue
            elif loss[t] > C and loss[t] <= D: 
                lossTr[t] = (loss[t] - C)*Lavg
                
            else: 
                lossTr[t] = (D-C)*Lavg
        #lossTr=Lavg*np.maximum(np.minimum(loss,D)-C,0)
        #print 'lossTr:'
        #print lossTr
        #print ''
        #print 'losstr', lossTr
        ################Present value defualt leg    
        pvDl = 0.0 
        for t in range(0,T):
            if t == 0:
                pvDl = pvDl + 1/discFac[t]*lossTr[t]
            else:
                pvDl = pvDl + 1/discFac[t]*(lossTr[t] - lossTr[t-1])    
        #print 'lossTr[0]: ', lossTr[0]
        #print '(lossTr[1:len(delta)] - lossTr[0:len(delta)-1]):', (lossTr[1:len(delta)] - lossTr[0:len(delta)-1])
        #print 'lossTr[1:len(delta)]:', lossTr[1:len(delta)]
        #print 'lossTr[0:len(delta)-1]:', lossTr[0:len(delta)-1]
        #pvDl=(1/discFac[0])*lossTr[0]+np.sum(1/discFac[1:len(delta)]*(lossTr[1:len(delta)] - lossTr[0:len(delta)-1])) 
        #print 'pvDl:', pvDl
        ############ present value premium leg    
        pvPl = 0.0
        for t in range(0,T):
            pvPl = pvPl + delta[t]/discFac[t]*((D-C)*Lavg-lossTr[t])        
        pvPl = coupon*pvPl
        #print ''
        #print 'np.sum(delta/discFac*((D-C)*Lavg-lossTr)):', np.sum(delta/discFac*((D-C)*Lavg-lossTr))
        #print ''
        pvPl=coupon*np.sum(delta/discFac*((D-C)*Lavg-lossTr))
        
        pvMean[0, sim] = pvDl
        pvMean[1, sim] = pvPl

    
    pvMean = np.mean(pvMean,axis=1)

    print 'Present value, PVdl, PVpl: ', pvMean  
    print pvMean[0] - pvMean[1]
    return abs(pvMean[0] - pvMean[1] - marketPrice)

#contRatio = contagion(0.4)
c = opt.fmin_cobyla(contagion, [0.2], (), args=(), rhoend = 0.0000001)
print 'True ratio: ', c
print("--- %s seconds ---" % (time.time() - start_time))