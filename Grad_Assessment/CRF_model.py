#!/usr/bin/env python
# coding: utf-8

#Original ipynb file under Redundant Code folder

# In[7]:


import numpy as np
from tensorflow import math
import tensorflow as tf
import math
from scipy.linalg import expm
from scipy import optimize


# In[9]:


class CRF_model:
    
    def __init__(self, x_feats, y_feats, label_dict):
        
        #self.x: our np array of x features
        #self.y: our np array of y features, which is simply the label IDs of a given sentence + BOS/EOS tokens
        #self.ld: a dict containing labels as the keys and their respective IDs as the values
        self.x = x_feats
        self.y = y_feats
        self.ld = label_dict
    
    def forward_vec():
        
        return
    
    def backward_vec():
        
        return
    
    
    
    
    #lamb is short for "lambda" (the parameter λ as described in CRFS: an introduction.) Lamb TBD using an optimizer 
    #function, such as Adam. Since most optimizers determine the minimum instead of maximum,
    #simply optimize the negative likelihood instead of positive likelihood if using a minimum optimizer.
    def log_likelihood(self, lamb):
        
        #label IDs for 'BOS' and 'EOS' tokens
        BOS_ID = self.ld['BOS']
        EOS_ID = self.ld['EOS']
        
        #computes Mi(y′, y|x) as described in section 6 of CRF introductions by Hannah. Used in log likelihood to 
        #compute Z(x)
        #NOTE: can't use tf.math.reduce_logsumexp, since order is log -> exp -> sum instead of log -> sum -> exp
        def compute_M(lamb_sum):
            
            return expm(lamb_sum)
            
        likelihood = 0
        
        #for each observation sequence and its respective label sequence within our x y datasets:
        for x_feat, y_feat in zip(self.x, self.y):
               
            lamb_sum = 0
            Z = None
            
            #for each word within our observation sequence:
            for word in x_feat:

                #lamb features: lambda (λ) * our X transition + state functions (Fj (y(k), x(k))) for a single sentence
                #in our x features.
                #lamb sum: the sum of lamb features over all of our x features. Used in computing Z

                lamb_features = np.dot(word, lamb)
                lamb_sum += lamb_features
                if Z is None:
                    Z = compute_M(lamb_features)
                else:
                    Z = np.matmul(Z, lamb_features)
                
            
            try:    
                likelihood += (lamb_sum - math.log(Z[BOS_ID][EOS_ID]))
            except ValueError:
                print("ERROR: tried to log 0 or a negative no. when calculating Z(x)! Breaking function")
                print("Debug data:")
                print("Z[BOS_ID][EOS_ID]:")
                print(Z[BOS_ID][EOS_ID])
                print("Z: ")
                print(Z)
                break
            
        #Can be returned negative for optimizing purposes
        return -likelihood

    
    def train(self):
        
        lamb_2 = np.random.randn(self.x[0].shape[3])
        l = lambda lamb: self.log_likelihood(lamb)
        #adadelta = tf.keras.optimizers.Adadelta(learning_rate = 0.001, rho = 0.95, epsilon = 1e-07, name = 'Adadelta')
        
        lamb_final = optimize.fmin_l_bfgs_b(l, lamb_2)
        
        #adadelta_run = adadelta.minimize(l, var_list = [lamb_final])
        print(lamb_final)
        
        return lamb_final


# In[ ]:





# In[ ]:




