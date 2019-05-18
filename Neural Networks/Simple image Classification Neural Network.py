#this is a simple image classification neural network 
#this neural network has 3 layers 
#1 input layer, 1 hidden layer and 1 output layer
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio



def sigmoid(z):
    return (1/(1+np.exp(-z)))

def relu(z):
    return (np.maximum(0,z))

#initialize parameters using he initialization
def parameter_initializtion(layers_dims):
    no_of_layers=len(layers_dims)
    parameters={}
    for i in range(no_of_layers-1):
        parameters["W"+str(i+1)]=np.random.randn(layers_dims[i+1],layers_dims[i])*np.sqrt(2/(layers_dims[i]))
        parameters["b"+str(i+1)]=np.zeros((layers_dims[i+1],1))
    return parameters


# linear ->relu ->linear->sigmoid
def forward_propergation(X,parameters):
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    z1=np.dot(W1,X)+b1
    a1=relu(z1)
    z2=np.dot(W2,a1)+b2
    a2=sigmoid(z2)
    return (a1,a2,z1,z2,W1,W2,b1,b2)


def back_proporgation(cache,X,y_new):
    a1,a2,z1,z2,W1,W2,b1,b2=cache
    m=len(X[0])
    dz2 = a2 - y_new
    dW2 = 1./m * np.dot(dz2, a1.T)
    db2 = 1./m * np.sum(dz2, axis=1, keepdims = True)
    
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = 1./m * np.dot(dz1, X.T)
    db1 = 1./m * np.sum(dz1, axis=1, keepdims = True)
    grad={"dW1":dW1,"dW2":dW2,"db1":db1,"db2":db2,"da1":da1,"dz1":dz1,"dz2":dz2}
    return grad

def calculate_cost(a2,y_new):
    m=len(a2[0])
    logprobs = np.multiply(-np.log(a2),y_new) + np.multiply(-np.log(1 - a2), 1 - y_new)
    cost = 1./m * np.nansum(logprobs)
    return cost

#gradient descent optimizer
def gradient_descent(X,y_new,y,alpha,parameters,no_of_iterations):
    lossmat=[]
    
    for i in range(no_of_iterations):
        cache=forward_propergation(X,parameters)
        a1,a2,z1,z2,W1,W2,b1,b2=cache
        grad=back_proporgation(cache,X,y_new)
        for t in range(2):
            parameters["W"+str(t+1)]=parameters["W"+str(t+1)]-alpha*grad["dW"+str(t+1)]
            parameters["b"+str(t+1)]=parameters["b"+str(t+1)]-alpha*grad["db"+str(t+1)]
        lossmat.append(calculate_cost(a2,y_new))
    
    #plot loss over iterations
    iter=range(no_of_iterations)
    plt.plot(iter,lossmat)
    plt.xlabel("no of iterations")
    plt.ylabel("loss")
    plt.show()
    
    #calculate accuracy for the training set
    pred=a2.T
    predicted=np.reshape(np.argmax(pred,axis=1)+1,(len(pred),1))
    count=0
    for i in range(len(pred)):
        if(predicted[i]==y[i]):
            count+=1
    print("loss:",lossmat[no_of_iterations-1],"  accuracy:",count/len(pred))
    return parameters




def main():
    
    #input data file
    mat_contents = sio.loadmat('datafile.mat')
    
    # data preprocessing
    X=mat_contents['X']
    y=mat_contents['y']
    layers_dims=[len(X[0]),25,10]
    parameters=parameter_initializtion(layers_dims)
    y_new=np.zeros((len(y),layers_dims[2]))
    
    #convert y set into 10 node output marix
    for i in range(len(y)):
        y_new[i][y[i]-1]=1

    
    
    parameters=gradient_descent(X,y_new,y,0.1,parameters,10000)

    
    
    
if __name__ == '__main__':
    main()
