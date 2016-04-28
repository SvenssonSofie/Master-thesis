import numpy as np
import math

def getGaussHermitePointsandWeights(n):
 [x,w]=np.polynomial.hermite.hermgauss(n)
 x=x*math.sqrt(2)
 w=w/math.sqrt(math.pi)
 return [x,w]

def testfun(x):
  return x*np.exp(3*x)

[x,w]=getGaussHermitePointsandWeights(50)
approxint=np.sum(testfun(x)*w)
trueint=math.exp(4.5)*3.0
print 'approximative: %.12f' % approxint, 'exact: %.12f' % trueint 
# error is one unit in 12 decimal place



