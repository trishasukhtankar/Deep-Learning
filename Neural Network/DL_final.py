#!/usr/bin/env python
# coding: utf-8

# In[364]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[341]:


data = pd.read_csv("D:\Assignments\Sem 2\Deep Learning\circles500.csv")
data = pd.DataFrame(data)


# In[342]:


#split data into features and labels
y = data["Class"]

X = data.drop(['Class'], axis = 1)


# In[343]:


#split data into test train set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[344]:


#initializing parameters
size_x = X_train.shape[1] #size of input layer
no_of_hidden = 4 #size of hidden layer
size_y =1 #size of output layer
m = X_train.shape[0] #size of the training set
#User-defined function to enable reusability of parameters
def init_param(size_x, no_of_hidden, size_y):

    #Randomizing weights and biases
    w1 = np.random.randn(size_x,no_of_hidden) * 0.01
    b1 = np.random.randn(no_of_hidden)
    w2 = np.random.randn(no_of_hidden,size_y) * 0.01
    b2 = np.random.randn(size_y)

    #creating a dictionary to store the weights
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    return parameters


# In[345]:



def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s

def sigmoid_d(Z):
    s = sigmoid(Z)*(1-sigmoid(Z))
    return s


# In[346]:


#feedforward function

def feedforward(X_train, parameters): #takes training data (features), weights and biases as input
   
    w1 = parameters['w1']
    w2 = parameters['w2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    z1 = np.dot(X_train,w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1,w2) + b2
    a2 = sigmoid(z2)
    
    return(z1,z2,a1,a2)


# In[347]:


def backprop(parameters,error,ff_output,X_train,alpha):
    delta=error.reshape(400,1)*sigmoid_d(ff_output[1])
    
    parameters['w2']= parameters['w2']+np.dot(ff_output[1].T,delta)*alpha
    
    delta_h=np.dot(delta,parameters['w2'].T)*sigmoid_d(ff_output[0])
    
    parameters['w1']= parameters['w1']+np.dot(X_train.T,delta_h)*alpha
    
    parameters['b2']=parameters['b2']+np.sum(delta)*alpha
    parameters['b1']=parameters['b1']+np.sum(delta_h)*alpha
    
    return(parameters)


# In[348]:


def update_init(init_param, parameters):
    init_param['w1'] = parameters['w1']
    init_param['w2'] = parameters['w2']
    init_param['b1'] = parameters['b1']
    init_param['b2'] = parameters['b2']
    
    return(init_param)


# In[362]:



def train(parameters,X_train,y_train):
    for i in range(10000):
        ff_output = feedforward(X_train, parameters)
        error=np.asarray(y_train-ff_output[3].reshape(400,))
        parameters_f=backprop(parameters,error,ff_output,X_train,0.0001)
        updated_param = update_init(parameters,parameters_f)
        parameters=updated_param
        if(i%1000 ==0):
            print(i,abs(np.mean(error)))
        


# In[363]:


parameters=init_param(size_x, no_of_hidden, size_y)
train(parameters,X_train,y_train)


# In[ ]:


sklearn.metrics.accuracy_score(y_test, y_train)


# In[ ]:




