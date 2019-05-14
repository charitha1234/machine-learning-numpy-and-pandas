import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1+np.exp(-z))

#mapping features in to polynormial terms
def featuremapping(X1,X2):
    X=np.ones(np.shape(X1[:]))
    degree=6
    for i in range(1,degree):
        for j in range(i+1):
            X= np.append(X,(X1**(i-j))*(X2**j),axis=1)
    return X

#loss calculation function
def calculateLoss(theta,X,y,Lambda):
    m=np.shape(y)[0]
    loss=(1/m)*(-y.transpose().dot(np.log(sigmoid(X.dot(theta))))-(1-y).transpose().dot(np.log(1-sigmoid(X.dot(theta)))))+ (Lambda/(2*m))*sum(theta[1:]**2)
    return loss[0][0] 

#gradient calculation function
def gradient(theta,X,y,Lambda):
    m=np.shape(y)[0]
    theta0=theta[1:]
    theta0=np.insert(theta0,0,0,axis=0)
    grad=(1/m)*((sigmoid(X.dot(theta))-y).transpose().dot(X)) +(Lambda/m)*theta0.transpose()
    return grad

#gradient descent optimization algorithm
def gradientdescent(theta,alpha,Lambda,no_of_iterations):
    lossmat=[]
    for i in range(no_of_iterations):
        theta=theta-alpha*gradient(theta,X,y,Lambda).transpose()
        lossmat.append(calculateLoss(theta,X,y,Lambda))
    iter=range(no_of_iterations)
    plt.plot(iter,lossmat)
    plt.xlabel('No of Iterations')
    plt.ylabel('Loss')
    plt.show()
    
def main():
    #input data file 
    dataset=pd.read_csv('logistic-dataset.txt',sep=",",header=None)
    
    #separate features 
    X1=dataset.iloc[:,0:1].values
    X2=dataset.iloc[:,1:2].values
    y=dataset.iloc[:,2:3].values
    
    #plot data
    ones=np.where(y==1)
    zeros=np.where(y==0)
    fig = plt.figure(figsize=(10,10))
    plt.scatter(X1[ones[0]],X2[ones[0]],c='r',s=20,label='y=1')
    plt.scatter(X1[zeros[0]],X2[zeros[0]],c='g',s=20,label='y=0')
    plt.legend()
    plt.show()

    X=featuremapping(X1,X2)

    theta=np.zeros(np.shape(X[1:2,:])).transpose()

    gradientdescent(theta,0.03,1,50000)

    
if __name__ == '__main__':
    main()

