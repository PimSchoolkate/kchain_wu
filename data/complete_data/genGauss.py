#!/usr/bin/env python

import numpy as np

#n = 10
#X1 = np.random.randn(n,2) + np.array([10,10])
#X2 = np.random.randn(n,2) 
#X = np.vstack((X1,X2))
#Y = np.vstack(( np.zeros((n,1)), np.ones((n,1)) ))
#
#np.savetxt('gauss2.csv', X, delimiter=',', fmt='%f') 
#np.savetxt('gauss2_label.csv', Y, delimiter=',', fmt='%d')  
#
##np.savetxt('gauss2_test.csv', X, delimiter=',', fmt='%f') 
##np.savetxt('gauss2_test_label.csv', Y, delimiter=',', fmt='%d')  





#	Adversarial dataset

n = 40
X1 = np.random.rand(n,2)
X2 = X1 + 0.01*np.random.randn(n,2)

X = np.vstack((X1,X2))
Y = np.vstack(( np.zeros((n,1)), np.ones((n,1)) ))

np.savetxt('gauss4.csv', X, delimiter=',', fmt='%f') 
np.savetxt('gauss4_label.csv', Y, delimiter=',', fmt='%d')  

