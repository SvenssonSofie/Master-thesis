
'''
Created on 4 mars 2016

@author: FIHE03E1
'''
import math
import numpy as np
import scipy.stats as sct
import scipy.optimize as opt

print sct.norm.cdf(sct.norm.ppf(0.02))

#func = np.poly1d([1.3, 4.0, 0.6])
#def func(y, *kPrice):
#    print 1.3*math.pow(y[0],2) + 4.0*y[0] + 0.6*kPrice[0]
#    return 1.3*math.pow(y[0],2) + 4.0*y[0] + 0.6*kPrice[0]
#print func

#def cons1(corr):
 #   return 5 - abs(corr)

#"kPrice = [10.0]
#x = opt.fmin_cobyla(func, [0.5], [cons1], args = (kPrice), consargs=())
#print "solved x = ", x


 
#tr = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.22, 1.0])#tranches
 
#C = 0.06
 
#print min(np.where(tr >= C)[0]) #index of the first tranche detachment greater than C
#print type(np.where(tr >= C))
#rho = 0.5
#normRand1 = np.random.normal(size = (5, 5))
#normRand = np.random.normal(size = (5, 5))
#normRand2 = rho*normRand1 + math.sqrt(1 - rho**2)*normRand

#print sct.norm.ppf(normRand)

#print normRand1
#print normRand 
#print normRand2

#print sct.norm.cdf(normRand)

#randCDF = sct.norm.cdf(normRand)
#uniRand = sct.uniform.ppf(randCDF)
#print uniRand