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
from scipy import linalg
from numpy import linalg as LA
from scipy.spatial.distance import cdist


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

file1="asasdasdasd"
#file1="recommendationTestCorregionalMatternResults5AveragingSamples250TrainingPoints/SBO/2000run/2000XHist.txt"

#file2="recommendationTestCorregionalMatternResults5AveragingSamples250TrainingPoints/SBO/2000run/2000yhist.txt"

#file1="recommendationTestCorregionalMatternResults5AveragingSamples250TrainingPoints/SBO/4001run/4001XHist.txt"
#file2="recommendationTestCorregionalMatternResults5AveragingSamples250TrainingPoints/SBO/4001run/4001yhist.txt"


#file1="recommendationTestMatternResults5AveragingSamples50TrainingPoints/SBO/999run/999XHist.txt"
#file2="recommendationTestMatternResults5AveragingSamples50TrainingPoints/SBO/999run/999yhist.txt"

n1=4
n2=1

###rate leraning, regularizing parameter, rank, epoch
lowerX=[0.0001,0,1,1]
upperX=[4.0,5.0,21,201]



nGrid=[11,11,11,6]

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

def logistic(x):
    return 1.0/(1.0+np.exp(x))

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
    res=g(x,w)

    return g(x,w),0.0



def sampleFromXAn(n,Train=False):
    """Chooses n points in the domain of x at random
      
       Args:
          n: Number of points chosen
    """
    if Train:
	s=np.array([lowerX[0]*10,upperX[1]])
    
	M=10
	d=np.array([lowerX[0],lowerX[1]])
	t1=np.random.uniform(d,s,(M,2))
	
	low=upperX[0]-0.01
	d=np.array([low,lowerX[1]])
	s=np.array([upperX[0],upperX[1]])
	t6=np.random.uniform(d,s,(M,2))
	
	d=np.array([lowerX[0]*10,lowerX[1]])
	s=np.array([low,upperX[1]])
	t7=np.random.uniform(d,s,(n-2*M,2))
	s1=np.concatenate((t1,t6,t7),0)
    
	a=np.random.randint(lowerX[2],upperX[2],n).reshape((n,1))
	b=np.random.randint(lowerX[3],upperX[3],n).reshape((n,1))
	
	
	return np.concatenate((s1,a,b),1)
    else:
	if n>2:
	    s=np.array([lowerX[0]*10,upperX[1]])
	
	    M=1
	    d=np.array([lowerX[0],lowerX[1]])
	    t1=np.random.uniform(d,s,(M,2))
	    
	    low=upperX[0]-0.01
	    d=np.array([low,lowerX[1]])
	    s=np.array([upperX[0],upperX[1]])
	    t6=np.random.uniform(d,s,(M,2))
	    
	    d=np.array([lowerX[0]*100.0,lowerX[1]])
	    s=np.array([low,upperX[1]])
	    t7=np.random.uniform(d,s,(n-2*M,2))
	    s1=np.concatenate((t1,t6,t7),0)
	else:
	    e=np.random.randint(lowerX[0],upperX[0],n).reshape((n,1))
	    f=np.random.randint(lowerX[1],upperX[1],n).reshape((n,1))
	    s1=np.concatenate((e,f),1)
    
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
                          simulatorW,estimationObjective,sampleFromXAn,nIS=numberIS,
			  corregional=True)


"""
We define the miscellaneous object.
"""
parallel=nTemp5

trainingPoints=nTemp2


misc=inter.Miscellaneous(randomSeed,parallel,nF=numberSamplesForF,tP=trainingPoints*numberIS,
                         prefix="recommendationTestCorregionalMattern")

"""
We define the data object.
"""

"""
Generate the training data
"""

if os.path.isfile(file1):
    print "bien"
    XWtrain=np.loadtxt(file1)
    yHist=np.loadtxt(file2)
    yHist=yHist.reshape((len(yHist),1))
#    XWtrain[0:trainingPoints*numberIS,0]=np.log((1.01/XWtrain[0:trainingPoints*numberIS,0])-1.0)
 #   XWtrain[0:trainingPoints*numberIS,1]=np.log((2.1/XWtrain[0:trainingPoints*numberIS,1])-1.0)
    dataObj=inter.data(XWtrain[0:trainingPoints*numberIS,:],yHist=yHist[0:trainingPoints*numberIS,0:1],varHist=np.zeros(trainingPoints*numberIS))

else:

    ###
    numberIS=5
    
    Xtrain=sampleFromXVn(trainingPoints,Train=True).reshape((trainingPoints,n1))
    Xtrain=np.repeat(Xtrain,5,axis=0)
    
    Wtrain=[]
    #for i in range(numberIS):
    Wtrain=[0,1,2,3,4]*trainingPoints
    Wtrain=np.array(Wtrain).reshape((trainingPoints*numberIS,1))
    
    XWtrain=np.concatenate((Xtrain,Wtrain),1)
    
  
    dataObj=inter.data(XWtrain,yHist=None,varHist=None)
    
    numberSamplesForF=1

    dataObj.getTrainingDataSBO(trainingPoints*numberIS,noisyF,numberSamplesForF,True)
   # trainingPoints*=numberIS
    

#index=[]
#for j in range(numberIS):
#    index.append([i+50*j for i in range(trainingPoints)])
#index=np.concatenate((index[0],index[1],index[2],index[3],index[4]))
#trainingPoints*=numberIS

#XWtrain[index,0]=np.log((1.01/XWtrain[index,0])-1.0)
#XWtrain[index,1]=np.log((2.1/XWtrain[index,1])-1.0)

#trainingPoints+=1
#XWtrain[0:trainingPoints*numberIS,0]=np.log((1.01/XWtrain[0:trainingPoints*numberIS,0])-1.0)
#XWtrain[0:trainingPoints*numberIS,1]=np.log((2.1/XWtrain[0:trainingPoints*numberIS,1])-1.0)


trainingPoints*=numberIS

#trainingPoints*=numberIS
"""
We define the statistical object.
"""

dimensionKernel=n1
scaleAlpha=np.array([1.0,1.0,20.0,200.0])

#kernel=mattern52.MATTERN52(n1+n2,X=XWtrain,y=yTrain[:,0],noise=NoiseTrain,scaleAlpha=scaleAlpha)


def expectation(x,X,z,alpha1,alpha2):
    num=0
    temp=alpha1*(np.sum((X-x)**2))
    for i in range(n1):
	r=temp
	num+=(1+np.sqrt(5*r)+(5.0/3.0)*r)*np.exp(-np.sqrt(5*r))
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
   # kernel=kernels[int(W)]
    alpha1=((kernel.alpha[0:n1])**2)/kernel.scaleAlpha[0:n1]**2
    variance0=kernel.variance
   # print alpha1
   # print variance0
   # print kernel.matrix
    
    for i in xrange(x.shape[0]):
	r=(np.sum(alpha1*((X-x[i,:])**2)))
	a=(1.0+np.sqrt(5.0*r)+(5.0/3.0)*r)*np.exp(-np.sqrt(5.0*r))
	sum1=0
	
	#sum2=0
	for j in range(numberIS):
	    sum1+=kernel.matrix[int(W),j]
	   # sum2+=kernel.A(np.concatenate((x[i:i+1,:],np.array([[j]])),1),XW.reshape((1,n1+n2)))
	results[i]=variance0*a*sum1

    return results/float(numberIS)





stat=stat.SBOGP(B=B,dimNoiseW=n2,dimPoints=n1,trainingData=dataObj,
                dimKernel=n1, numberTraining=trainingPoints,
                computeLogProductExpectationsForAn=
                None,scaledAlpha=scaleAlpha,SEK=False,mat52=False,
		corregMat52=True,nIS=numberIS)



"""
We define the VOI object.
"""

pointsVOI=np.array(domain) #Discretization of the domain of X



VOIobj=VOI.VOISBO(dimX=n1, pointsApproximation=pointsVOI,
                  gradWBfunc=None,dimW=n2,
                  numberTraining=trainingPoints,SEK=False,mat52=False,
		  corregMat52=True)


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
    
    



def transformationDomainXAn(x):
    """ Transforms the point x given by the steepest ascent method to
        the right domain of x.
        
       Args:
          x: Point to be transformed.
    """
  #  if (x[0]<=0):
#	x[0]=1e-8
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

#lowerX=[0.01,0,1,1]
#upperX=[1.01,1.0,21,201]



def projectGradient(x,direction,xo,step):
    alphL=[]
    alphU=[]
    st=step
    
    if (x[0]<lowerX[0]):
	ind=0
	if (direction[ind]>=0):
	
	    return xo
	quotient=(-xo[ind].astype(float)+lowerX[0])/direction[ind]
        alp=np.min(quotient)
        st=min(st,alp)
	
    if (x[1]<lowerX[1]):
	ind=1
	if (direction[ind]>=0):
	
	    return xo
	quotient=(-xo[ind].astype(float)+lowerX[1])/direction[ind]
        alp=np.min(quotient)
        st=min(st,alp)
	
    if (x[1]>upperX[1]):
       	ind=1
	if (direction[ind]<=0):
	
	    return xo
	quotient=(-xo[ind].astype(float)+upperX[1])/direction[ind]
        alp=np.min(quotient)
        st=min(st,alp)
	
    if (x[0]>upperX[0]):
       	ind=0
	if (direction[ind]<=0):
	
	    return xo
	quotient=(-xo[ind].astype(float)+upperX[0])/direction[ind]
        alp=np.min(quotient)
        st=min(st,alp)
	
	
#    if (any(x[0:2]>5)):
 
 #     	ind=np.where(x[0:2]>2)[0]
#	if (any(direction[ind]<=0)):
#	  
#	    return xo

#	quotient=(-xo[ind].astype(float)+2.0)/direction[ind]
 #       alp=np.min(quotient)

  #      st=min(st,alp)


    if (any(x[2:4]<1)):
	
       	ind=np.where(x[2:n1]<1)[0]
	ind=ind+2
	if (any(direction[ind]>=0)):
	
	    return xo
	quotient=(-xo[ind].astype(float)+1.0)/direction[ind]
        alp=np.min(quotient)
        st=min(st,alp)
	

	
#    if (any(x[0:2]>5)):
 
 #     	ind=np.where(x[0:2]>2)[0]
#	if (any(direction[ind]<=0)):
#	  
#	    return xo

#	quotient=(-xo[ind].astype(float)+2.0)/direction[ind]
 #       alp=np.min(quotient)

  #      st=min(st,alp)
    temp=xo+direction*st
    if (any(temp<lowerX) or any(temp>upperX)):
	return xo

    return xo+direction*st
#1.01,2.1,21,201]

def projectGradient2(x,direction,xo,step):
    if (any(x<lowerX)):
	ind=np.where(x<lowerX)[0]
	x[ind]=lowerX[ind]
    
    if (any(x>upperX)):
	ind=np.where(x>upperX)[0]
	x[ind]=upperX[ind]
    
    return x
	

def stopFunction(x):
    s=np.rint(abs(x[0,2:n1]))
    t=np.max(s)
    d=np.max(abs(x[0,0:2]))
    return np.max([d,t])


opt=inter.opt(nTemp6,n1,n1,transformationDomainXVn,transformationDomainXAn,
              transformationDomainW,projectGradient2,functionGradientAscentVn,
              functionGradientAscentAn,stopFunction,1e-2,1e-2,None,None,"GRADIENT","GRADIENT")




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


#sboObj.stat._k.trainnoParallel2(np.ones(4),numStarts=1)
#sboObj.runISvoi(0,True)
#sboObj.optVOInoParal(0,1,0)
#sboObj.trainModel(numStarts=10)
#sboObj.optAnnoParal(0)
#print "ok"
sboObj.SBOAlg(nTemp4,nRepeat=10,Train=True,plots=False)



