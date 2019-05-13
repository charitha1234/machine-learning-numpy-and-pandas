import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


no_of_features=2 #enter no of features

#read the file
data=pd.read_csv('dataset.txt',sep=",",header=None)

#split training set
X=data.iloc[:,0:no_of_features].values
y=data.iloc[:,no_of_features:no_of_features+1].values

#feature normalization
mu_X=np.mean(X,axis=0)
mu_y=np.mean(y)
s_X=np.std(X,axis=0)
s_y=np.std(y)
s_X=np.reshape(s_X,(1,no_of_features))
s_y=np.reshape(s_y,(1,1))
mu_X=np.reshape(mu_X,(1,no_of_features))
mu_y=np.reshape(mu_y,(1,1))
t=np.ones((len(X),1))
X_norm=(X-t.dot(mu_X))/t.dot(s_X)
y_norm=(y-t.dot(mu_y))/t.dot(s_y)

#insert a colomn of ones
X_norm=np.insert(X_norm,0,1, axis=1)

#training the model
alpha=0.03
no_of_iter=1000
theta=np.zeros((no_of_features+1,1))
m=len(X)
lossmat=[]
for i in range(no_of_iter):
    theta=theta-alpha*(1/m)*((X_norm.dot(theta)-y_norm).transpose().dot(X_norm).transpose())
    loss=(1/(2*m))*((X_norm.dot(theta)-y_norm).transpose().dot((X_norm.dot(theta)-y_norm)))
    lossmat.append(loss[0][0])

#plot loss over iterations
iterations=range(no_of_iter)  
plt.plot(iterations,lossmat)
plt.xlabel('No of iterations')
plt.ylabel('loss')
