import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def calculateLoss(theta,X,y):
    m=np.shape(y)[0]
    loss=(1/m)*(-y.transpose().dot(np.log(sigmoid(X.dot(theta))))-(1-y).transpose().dot(np.log(1-sigmoid(X.dot(theta)))))
    return loss[0][0]

def gradient(theta,X,y):
    m=np.shape(y)[0]
    theta0=theta[1:]
    theta0=np.insert(theta0,0,0,axis=0)
    grad=(1/m)*((sigmoid(X.dot(theta))-y).transpose().dot(X))
    return grad

#gradient descent optimizer
def gradientdescent(X,y,alpha,theta,no_of_iterations):
    lossmat=[]
    for i in range(no_of_iterations):
        theta=theta-alpha*gradient(theta,X,y).transpose()
        lossmat.append(calculateLoss(theta,X,y))
    #plot loss over iteration
    iter=range(no_of_iterations)
    plt.plot(iter,lossmat)
    plt.xlabel('No of Iterations')
    plt.ylabel('Loss')
    plt.show()
    return theta


def main():
    #input dataset
    dataset=pd.read_csv('logistic-dataset.txt',sep=",",header=None)

    X1=dataset.iloc[:,0:1].values
    X2=dataset.iloc[:,1:2].values
    y=dataset.iloc[:,2:3].values
    
    #plot dataset
    ones=np.where(y==1)
    zeros=np.where(y==0)
    fig = plt.figure(figsize=(10,10))
    plt.scatter(X1[ones[0]],X2[ones[0]],c='r',s=20,label='y=1')
    plt.scatter(X1[zeros[0]],X2[zeros[0]],c='g',s=20,label='y=0')
    plt.legend()
    plt.show()

    #make feature matrix
    X=np.append(X1,X2,axis=1)
    X=np.insert(X,0,1,axis=1)
    
    
    theta=np.zeros(np.shape(X[1:2,:])).transpose()

    theta=gradientdescent(X,y,0.001,theta,1000000)

if __name__ == '__main__':
    main()
