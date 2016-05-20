#!/usr/bin/env python

"""
We consider a queuing simulation based on New York City's Bike system,
in which system users may remove an available bike from a station at one
location within the city, and ride it to a station with an available dock
in some other location within the city. The optimization problem that we
consider is the allocation of a constrained number of bikes (6000) to available
docks within the city at the start of rush hour, so as to minimize, in simulation,
the expected number of potential trips in which the rider could not find an
available bike at their preferred origination station, or could not find an
available dock at their preferred destination station. We call such trips
"negatively affected trips".

To use the SBO algorithm, we need to create 6 objets:

Objobj: Objective object (See InterfaceSBO).
miscObj: Miscellaneous object (See InterfaceSBO).
VOIobj: Value of Information function object (See VOIGeneral).
optObj: Opt object (See InterfaceSBO).
statObj: Statistical object (See statGeneral).
dataObj: Data object (See InterfaceSBO).

"""
import sys
sys.path.append("..")
import numpy as np
from math import *
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import norm,poisson
import statsmodels.api as sm
import multiprocessing as mp
import os
from scipy.stats import poisson
import json
from BGO.Source import *
import time
from pmf import cross_validation,PMF

##################

nTemp=int(sys.argv[1])  #random seed 
nTemp2=int(sys.argv[2]) #number of training points
nTemp3=int(sys.argv[3]) #number of samples to estimate F
nTemp4=int(sys.argv[4]) #number of iterations
nTemp5=sys.argv[5] #True if code is run in parallel; False otherwise.
nTemp6=int(sys.argv[6]) #number of restarts for the optimization method

if nTemp5=='F':
    nTemp5=False
    nTemp6=1
elif nTemp5=='T':
    nTemp5=True

n1=4
n2=1

###rate leraning, regularizing parameter, rank, epoch
lowerX=[0.01,0.01,1,1]
upperX=[1.01,1.01,21,201]


nGrid=[6,6,11,6]

domainX=[]
for i in range(n1):
    domainX.append(np.linspace(lowerX[i],upperX[i],nGrid[i]))
    
domain=[[a,b,c,d] for a in domainX[0] for b in domainX[1] for c in domainX[2] for d in domainX[3]]


randomSeed=nTemp
np.random.seed(randomSeed)

##############


numberSamplesForF=nTemp3


"""
We define the objective object.
"""
num_user=943
num_item=1682

train=[]
validate=[]

for i in range(1,6):
    data=np.loadtxt("ml-100k/u%d.base"%i)
    test=np.loadtxt("ml-100k/u%d.test"%i)
    train.append(data)
    validate.append(test)


def g(x,w1):
    val=PMF(num_user,num_item,train[w1],validate[w1],x[0],x[1],int(x[3]),int(x[2]))
    return -val*100
    

def noisyF(XW,n):
    """Estimate F(x,w)=E(f(x,w,z)|w)
      
       Args:
          XW: Vector (x,w)
          n: Number of samples to estimate F
    """
    
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]

    w=int(w)
    return np.sum(x),0.0



def sampleFromXAn(n):
    """Chooses n points in the domain of x at random
      
       Args:
          n: Number of points chosen
    """
    s1=np.random.uniform(lowerX[0:2],upperX[0:2],(n,2))
    a=np.random.randint(lowerX[2],upperX[2],n).reshape((n,1))
    b=np.random.randint(lowerX[3],upperX[3],n).reshape((n,1))
    
    
    return np.concatenate((s1,a,b),1)

sampleFromXVn=sampleFromXAn



def simulatorW(n):
    """Simulate n vectors w
      
       Args:
          n: Number of vectors simulated
    """
    return np.random.randint(0,n1,n).reshape((n,n2))


    



def estimationObjective(x,N=1):
    """Estimate g(x)=E(f(x,w,z))
      
       Args:
          x
          N: number of samples used to estimate g(x)
    """

    sol=0
    x=np.reshape(x,(1,n1))
    print x
    for i in range(n1):
	temp=np.concatenate((x,np.array([[i]])),1)
        sol+=noisyF(temp,0)[0]
    return sol/n1,0



Objective=inter.objective(None,n1,noisyF,numberSamplesForF,sampleFromXVn,
                          simulatorW,estimationObjective,sampleFromXAn)


"""
We define the miscellaneous object.
"""
parallel=nTemp5

trainingPoints=nTemp2


misc=inter.Miscellaneous(randomSeed,parallel,nF=numberSamplesForF,tP=trainingPoints,
                         prefix="recommendation")

"""
We define the data object.
"""

"""
Generate the training data
"""

Xtrain=sampleFromXVn(trainingPoints).reshape((trainingPoints,n1))

dt=trainingPoints/n1
Wtrain=[]
for i in range(n1):
    Wtrain+=[i]*dt
Wtrain=np.array(Wtrain).reshape((trainingPoints,1))

XWtrain=np.concatenate((Xtrain,Wtrain),1)
print XWtrain
dataObj=inter.data(XWtrain,yHist=None,varHist=None)

dataObj.getTrainingDataSBO(trainingPoints,noisyF,numberSamplesForF,False)



"""
We define the statistical object.
"""

dimensionKernel=n1+n2
scaleAlpha=np.array([1,1,20,200,4])
#kernel=SK.SEK(n1+n2,X=XWtrain,y=yTrain[:,0],noise=NoiseTrain,scaleAlpha=scaleAlpha)



def expectation(z,alpha):
    num=0
    for i in range(n1):
        num+=np.exp(-alpha*((z-i)**2))
    return num/(n1)

def B(x,XW,n1,n2,kernel,logproductExpectations=None):
    """Computes B(x)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
      
       Args:
          x: Vector of points where B is evaluated
          XW: Point (x,w)
          n1: Dimension of x
          n2: Dimension of w
          kernel
          logproductExpectations: Vector with the logarithm
                                  of the product of the
                                  expectations of
                                  np.exp(-alpha2[j]*((z-W[i,j])**2))
                                  where W[i,:] is a point in the history.
          
    """
    x=np.array(x).reshape((x.shape[0],n1))
    results=np.zeros(x.shape[0])
    #parameterLamb=parameterSetsPoisson
    X=XW[0:n1]
    inda=n1+n2
    W=XW[n1:inda]
    alpha2=0.5*((kernel.alpha[n1:n1+n2])**2)/scaleAlpha[n1:n1+n2]**2
    alpha1=0.5*((kernel.alpha[0:n1])**2)/scaleAlpha[0:n1]**2
    variance0=kernel.variance
    
    if logproductExpectations is None:
        logproductExpectations=0.0
        for j in xrange(n2):
	    temp=expectation(W[j],alpha2[j])
            logproductExpectations+=np.log(temp)
    for i in xrange(x.shape[0]):
        results[i]=logproductExpectations+np.log(variance0)-np.sum(alpha1*((x[i,:]-X)**2))
    return np.exp(results)

def computeLogProductExpectationsForAn(W,N,kernel):
    """Computes the logarithm of the product of the
       expectations of np.exp(-alpha2[j]*((z-W[i,j])**2))
        where W[i,:] is a point in the history.
      
       Args:
          W: Matrix where each row is a past random vector used W[i,:]
          N: Number of observations
          kernel: kernel
    """
    alpha2=0.5*((kernel.alpha[n1:n1+n2])**2)/scaleAlpha[n1:n1+n2]**2
    logproductExpectations=np.zeros(N)

    for i in xrange(N):
        logproductExpectations[i]=0.0
        for j in xrange(n2):
	    temp=expectation(W[i,j],alpha2[j])
            logproductExpectations[i]+=np.log(temp)
    return logproductExpectations

stat=stat.SBOGP(B=B,dimNoiseW=n2,dimPoints=n1,trainingData=dataObj,
                dimKernel=n1+n2, numberTraining=trainingPoints,
                computeLogProductExpectationsForAn=
                computeLogProductExpectationsForAn,scaledAlpha=scaleAlpha)


"""
We define the VOI object.
"""

pointsVOI=np.array(domain) #Discretization of the domain of X


def expectation2(z,alpha):
    num=-alpha*(z**2)+(((z*alpha)**2)*(1.0/(alpha+0.5)))
    num=np.exp(num)
    quotient=np.sqrt(2*alpha+1)
    a1=num/quotient
    a2=z-((z*alpha)/(alpha+0.5))
    return a1*a2

def gradWB(new,kern,BN,keep,points):
    """Computes the vector of gradients with respect to w_{n+1} of
	B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
	where x_{p} is a point in the discretization of the domain of x.
        
       Args:
          new: Point (x_{n+1},w_{n+1})
          kern: Kernel
          keep: Indexes of the points keeped of the discretization of the domain of x,
                after using AffineBreakPoints
          BN: Vector B(x_{p},n+1), where x_{p} is a point in the discretization of
              the domain of x.
          points: Discretization of the domain of x
    """
    alpha1=0.5*((kern.alpha[0:n1])**2)/scaleAlpha[0:n1]**2
    alpha2=0.5*((kern.alpha[n1:n1+n2])**2)/scaleAlpha[n1:n1+n2]**2
    variance0=kern.variance
    wNew=new[0,n1:n1+n2].reshape((1,n2))
    gradWBarray=np.zeros([len(keep),n2])
    M=len(keep)
   # parameterLamb=parameterSetsPoisson
    X=new[0,0:n1]
    W=new[0,n1:n1+n2]
    
    num=0
    for i in range(n1):
        num+=(2.0*alpha2*(i-wNew))*np.exp(-alpha2*((i-wNew)**2))
    num=num/n1
    for j in range(M):
        gradWBarray[j,0]=num*(variance0)*np.exp(np.sum(alpha1*((points[keep[j],:]-X)**2)))
    return gradWBarray

VOIobj=VOI.VOISBO(dimX=n1, pointsApproximation=pointsVOI,
                  gradWBfunc=gradWB,dimW=n2,
                  numberTraining=trainingPoints)


"""
We define the Opt object.

"""

dimXsteepestAn=n1 #Dimension of x when the VOI and a_{n} are optimized.


def functionGradientAscentVn(x,VOI,i,L,temp2,a,kern,XW,scratch,Bfunc,onlyGradient=False,grad=None):
    """ Evaluates the VOI and it can compute its derivative. It evaluates the VOI,
        when grad and onlyGradient are False; it evaluates the VOI and computes its
        derivative when grad is True and onlyGradient is False, and computes only its
        gradient when gradient and onlyGradient are both True.
    
        Args:
            x: VOI is evaluated at (x,numberBikes-sum(x)).Note that we reduce the dimension
               of the space of x.
            grad: True if we want to compute the gradient; False otherwise.
            i: Iteration of the SBO algorithm.
            L: Cholesky decomposition of the matrix A, where A is the covariance
               matrix of the past obsevations (x,w).
            Bfunc: Computes B(x,XW)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
            temp2: temp2=inv(L)*B.T, where B is a matrix such that B(i,j) is
                   \int\Sigma_{0}(x_{i},w,x_{j},w_{j})dp(w)
                   where points x_{p} is a point of the discretization of
                   the space of x; and (x_{j},w_{j}) is a past observation.
            a: Vector of the means of the GP on g(x)=E(f(x,w,z)). The means are evaluated on the
               discretization of the space of x.
            VOI: VOI object
            kern: kernel
            XW: Past observations
            scratch: matrix where scratch[i,:] is the solution of the linear system
                     Ly=B[j,:].transpose() (See above for the definition of B and L)
            onlyGradient: True if we only want to compute the gradient; False otherwise.
    """
    grad=onlyGradient
    x=np.array(x).reshape([1,n1+n2])

    tempX=x[0:1,0:n1]

    tempW=x[0:1,n1:n1+n2]
    xFinal=np.concatenate((tempX,tempW),1)
    temp=VOI.VOIfunc(i,xFinal,L=L,temp2=temp2,a=a,grad=grad,scratch=scratch,onlyGradient=onlyGradient,
                          kern=kern,XW=XW,B=Bfunc)

    

    if onlyGradient:

        return temp
        

    if grad==True:

        return temp[0],temp[1]
    else:

        return temp
    

def functionGradientAscentAn(x,grad,stat,i,L,dataObj,onlyGradient=False,logproductExpectations=None):
    """ Evaluates a_{i} and its derivative, which is the expectation of the GP on g(x).
        It evaluates a_{i}, when grad and onlyGradient are False; it evaluates the a_{i}
        and computes its derivative when grad is True and onlyGradient is False, and
        computes only its gradient when gradient and onlyGradient are both True.
    
        Args:
            x: a_{i} is evaluated at (x,numberBikes-sum(x)).Note that we reduce the dimension
               of the space of x.
            grad: True if we want to compute the gradient; False otherwise.
            i: Iteration of the SBO algorithm.
            L: Cholesky decomposition of the matrix A, where A is the covariance
               matrix of the past obsevations (x,w).
            dataObj: Data object.
            stat: Statistical object.
            onlyGradient: True if we only want to compute the gradient; False otherwise.
            logproductExpectations: Vector with the logarithm of the product of the
                                    expectations of np.exp(-alpha2[j]*((z-W[i,j])**2))
                                    where W[i,:] is a point in the history.
    """
   
    x=np.array(x).reshape([1,n1])

  #  x4=np.array(numberBikes-np.sum(x)).reshape((1,1))
   # x=np.concatenate((x,x4),1)
   
    if onlyGradient:
        temp=stat.aN_grad(x,L,i,dataObj,grad,onlyGradient,logproductExpectations)

        return temp

    temp=stat.aN_grad(x,L,i,dataObj,gradient=grad,logproductExpectations=logproductExpectations)
    if grad==False:
        return temp
    else:
        return temp[0],temp[1]
    
    




def const1(x):
    return x[0]-lowerX[0]

def jac1(x):
    return np.array([1,0,0,0])

def const2(x):
    return x[1]-lowerX[1]

def jac2(x):
    return np.array([0,1,0,0])

def const3(x):
    return x[2]-lowerX[2]

def jac3(x):
    return np.array([0,0,1,0])

def const4(x):
    return x[3]-lowerX[3]

def jac4(x):
    return np.array([0,0,0,1])




def const5(x):
    return upperX[0]-x[0]

def jac5(x):
    return np.array([-1,0,0,0])

def const6(x):
    return upperX[1]-x[1]

def jac6(x):
    return np.array([0,-1,0,0])

def const7(x):
    return upperX[2]-x[2]

def jac7(x):
    return np.array([0,0,-1,0])

def const8(x):
    return upperX[3]-x[3]

def jac8(x):
    return np.array([0,0,0,-1])



consA=({'type':'ineq',
        'fun': const1,
       'jac': jac1},
    {'type':'ineq',
        'fun': const2,
       'jac': jac2},
    {'type':'ineq',
        'fun': const3,
       'jac': jac3},
    {'type':'ineq',
        'fun': const4,
       'jac': jac4},
    {'type':'ineq',
        'fun': const5,
       'jac': jac5},
    {'type':'ineq',
        'fun': const6,
       'jac': jac6},
    {'type':'ineq',
        'fun': const7,
       'jac': jac7},
    {'type':'ineq',
        'fun': const8,
       'jac': jac8})


def transformationDomainXAn(x):
    """ Transforms the point x given by the steepest ascent method to
        the right domain of x.
        
       Args:
          x: Point to be transformed.
    """
    x[0,2:4]=np.rint(x[0,2:4])
    return x

transformationDomainXVn=transformationDomainXAn

def transformationDomainW(w):
    """ Transforms the point w given by the steepest ascent method to
        the right domain of w.
        
       Args:
          w: Point to be transformed.
    """
    
    return np.rint(w)



def const1b(x):
    return x[0]-lowerX[0]

def jac1b(x):
    return np.array([1,0,0,0,0])

def const2b(x):
    return x[1]-lowerX[1]

def jac2b(x):
    return np.array([0,1,0,0,0])

def const3b(x):
    return x[2]-lowerX[2]

def jac3b(x):
    return np.array([0,0,1,0,0])

def const4b(x):
    return x[3]-lowerX[3]

def jac4b(x):
    return np.array([0,0,0,1,0])




def const5b(x):
    return upperX[0]-x[0]

def jac5b(x):
    return np.array([-1,0,0,0,0])

def const6b(x):
    return upperX[1]-x[1]

def jac6b(x):
    return np.array([0,-1,0,0,0])

def const7b(x):
    return upperX[2]-x[2]

def jac7b(x):
    return np.array([0,0,-1,0,0])

def const8b(x):
    return upperX[3]-x[3]

def jac8b(x):
    return np.array([0,0,0,-1,0])

def const9b(x):
    return 4-x[4]

def jac9b(x):
    return np.array([0,0,0,0,-1])

def const10b(x):
    return x[4]

def jac10b(x):
    return np.array([0,0,0,0,1])



cons=({'type':'ineq',
        'fun': const1b,
       'jac': jac1b},
    {'type':'ineq',
        'fun': const2b,
       'jac': jac2b},
    {'type':'ineq',
        'fun': const3b,
       'jac': jac3b},
    {'type':'ineq',
        'fun': const4b,
       'jac': jac4b},
    {'type':'ineq',
        'fun': const5b,
       'jac': jac5b},
    {'type':'ineq',
        'fun': const6b,
       'jac': jac6b},
    {'type':'ineq',
        'fun': const7b,
       'jac': jac7b},
    {'type':'ineq',
        'fun': const8b,
       'jac': jac8b},
        {'type':'ineq',
        'fun': const9b,
       'jac': jac9b},
            {'type':'ineq',
        'fun': const10b,
       'jac': jac10b})


opt=inter.opt(nTemp6,n1,n1,transformationDomainXVn,transformationDomainXAn,
              transformationDomainW,None,functionGradientAscentVn,
              functionGradientAscentAn,None,1.0,cons,consA,"SLSQP","SLSQP")


"""
We define the SBO object.
"""
l={}
l['VOIobj']=VOIobj
l['Objobj']=Objective
l['miscObj']=misc
l['optObj']=opt
l['statObj']=stat
l['dataObj']=dataObj


sboObj=SBO.SBO(**l)


"""
We run the SBO algorithm.
"""

sboObj.SBOAlg(nTemp4,nRepeat=10,Train=True,plots=False)

