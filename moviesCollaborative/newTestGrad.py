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
from BGOis.Source import *
import time
from pmf import cross_validation,PMF


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

rS=int(1111)

file1="recommendationResults5AveragingSamples10TrainingPoints/SBO/%drun/%dXHist.txt"% (rS, rS)

file2="recommendationResults5AveragingSamples10TrainingPoints/SBO/%drun/%dyhist.txt"% (rS, rS)


n1=4
n2=1

###rate leraning, regularizing parameter, rank, epoch
lowerX=[0.01,0.1,1,1]
upperX=[1.01,2.1,21,201]



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
    return g(x,w),0.0



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
    return np.random.randint(0,5,n).reshape((n,n2))


    



def estimationObjective(x,N=1):
    """Estimate g(x)=E(f(x,w,z))
      
       Args:
          x
          N: number of samples used to estimate g(x)
    """

    sol=0
    x=np.reshape(x,(1,n1))
    print x
    try:
	
	pool = mp.Pool()
	jobs = []
	for j in range(5):
	    temp=np.concatenate((x,np.array([[j]])),1)
	    job = pool.apply_async(noisyF, (temp,0))
	    jobs.append(job)
	pool.close()  # signal that no more data coming in
	pool.join()  # wait for all the tasks to complete
    except KeyboardInterrupt:
	print "Ctrl+c received, terminating and joining pool."
	pool.terminate()
	pool.join()
    sols=[]
    for j in range(5):
	try:
	   sols.append(jobs[j].get()[0])
	except Exception as e:
	    print "Error computing CV"
    if len(sols)==5:
	return np.mean(sols),0
 


numberIS=5

Objective=inter.objective(None,n1,noisyF,numberSamplesForF,sampleFromXVn,
                          simulatorW,estimationObjective,sampleFromXAn,numberIS=numberIS)


"""
We define the miscellaneous object.
"""
parallel=nTemp5

trainingPoints=nTemp2


misc=inter.Miscellaneous(randomSeed,parallel,nF=numberSamplesForF,tP=trainingPoints,
                         prefix="recommendationTest")

"""
We define the data object.
"""

"""
Generate the training data
"""

XWtrain=np.loadtxt(file1)
yHist=np.loadtxt(file2)
yHist=yHist.reshape((len(yHist),1))


trainingPoints*=numberIS

dataObj=inter.data(XWtrain[0:trainingPoints,:],yHist=yHist[0:trainingPoints,0:1],varHist=np.zeros(trainingPoints))


"""
We define the statistical object.
"""

dimensionKernel=n1
scaleAlpha=np.array([1,1,20,200])

#kernel=mattern52.MATTERN52(n1+n2,X=XWtrain,y=yTrain[:,0],noise=NoiseTrain,scaleAlpha=scaleAlpha)



def expectation(z,alpha):
    num=0
    for i in range(n1):
        num+=np.exp(-alpha*((z-i)**2))
    return num/(n1)

def B(x,XW,n1,n2,kernels,logproductExpectations=None):
    """Computes B(x)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
      
       Args:
          x: Vector of points where B is evaluated
          XW: Point (x,w)
          n1: Dimension of x
          n2: Dimension of w
          kernels: list with the kernels for each IS
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
    W=int(XW[n1:inda])
    kernel=kernels[W]
    alpha1=0.5*((kernel.alpha[0:n1])**2)/(scaleAlpha[0:n1]**2)
    variance0=kernel.variance
    
    for i in xrange(x.shape[0]):
        results[i]=np.log(variance0)-np.sum(alpha1*((x[i,:]-X)**2))
    return np.exp(results)/numberIS

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
                dimKernel=n1, numberTraining=trainingPoints,
                computeLogProductExpectationsForAn=
                None,IS=numberIS,scaledAlpha=scaleAlpha)


"""
We define the VOI object.
"""

pointsVOI=np.array(domain) #Discretization of the domain of X




VOIobj=VOI.VOISBO(dimX=n1, pointsApproximation=pointsVOI,
                  gradWBfunc=None,dimW=n2,
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
   
    x=np.array(x).reshape([1,n1+n2])

    tempX=x[0:1,0:n1]

    tempW=x[0:1,n1:n1+n2]
    xFinal=np.concatenate((tempX,tempW),1)

    temp=VOI.VOIfunc(i,xFinal,L=L,temp2=temp2,a=a,grad=grad,scratch=scratch,onlyGradient=onlyGradient,
                          kerns=kern,XW=XW,B=Bfunc)
   
    

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
    return np.array([1,0,0,0])

def const2b(x):
    return x[1]-lowerX[1]

def jac2b(x):
    return np.array([0,1,0,0])

def const3b(x):
    return x[2]-lowerX[2]

def jac3b(x):
    return np.array([0,0,1,0])

def const4b(x):
    return x[3]-lowerX[3]

def jac4b(x):
    return np.array([0,0,0,1])




def const5b(x):
    return upperX[0]-x[0]

def jac5b(x):
    return np.array([-1,0,0,0])

def const6b(x):
    return upperX[1]-x[1]

def jac6b(x):
    return np.array([0,-1,0,0])

def const7b(x):
    return upperX[2]-x[2]

def jac7b(x):
    return np.array([0,0,-1,0])

def const8b(x):
    return upperX[3]-x[3]

def jac8b(x):
    return np.array([0,0,0,-1])

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
       'jac': jac8b})

def projectGradient(x,direction,xo,step):
    alphL=[]
    alphU=[]
    st=step
    if (any(x[0:2]<0)):
	ind=np.where(x[0:2]<0)[0]
	

	if (any(direction[ind]>=0)):

	    return xo
	quotient=(-xo[ind].astype(float))/direction[ind]
        alp=np.min(quotient)

        st=min(st,alp)

    if (any(x[2:4]<1)):
    
       	ind=np.where(x[2:n1]<1)[0]
	ind=ind+2
	if (any(direction[ind]>=0)):

	    return xo
	quotient=(-xo[ind].astype(float)+1.0)/direction[ind]
        alp=np.min(quotient)
        st=min(st,alp)

	
    if (any(x[0:2]>2)):
       	ind=np.where(x[0:2]>2)[0]
	if (any(direction[ind]<=0)):
	    return xo
	quotient=(-xo[ind].astype(float)+2.0)/direction[ind]
        alp=np.min(quotient)
        st=min(st,alp)

    
    return xo+direction*st


def stopFunction(x):
    s=np.rint(abs(x[0,2:n1]))
    t=np.max(s)
    d=np.max(abs(x[0,0:2]))
    return np.max([d,t])


opt=inter.opt(nTemp6,n1,n1,transformationDomainXVn,transformationDomainXAn,
              transformationDomainW,projectGradient,functionGradientAscentVn,
              functionGradientAscentAn,stopFunction,1e-2,1e-8/np.sqrt(numberIS),cons,consA,"GRADIENT","GRADIENT")




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
#sboObj.optVOIParal(0,1,0)
#sboObj.trainModel(numStarts=1)
#sboObj.optAnnoParal(0)

sboObj.SBOAlg(nTemp4,nRepeat=10,Train=True,plots=False)

