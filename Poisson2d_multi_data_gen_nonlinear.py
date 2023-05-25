#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:24:57 2022

@author: kishan
"""

import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

train_size = 2000 #100
test_size = 1000 #50
val_size =1000 #50

train_size_b = 2000
test_size_b = 1000
val_size_b = 1000

x_range = [0,1]
y_range = [0,1]

x_grid = 101
y_grid = 101

a = [10,20,30,40,50,60,70,80,90,100] # [10,20,30,40,50] # [10,12,14,16,18,20]  #[10,20,30,40,50,60,70,80,90,100]



# Generate data
# Train

train_data = list()
test_data = list()
val_data = list()
train_label = list()
test_label = list()
val_label = list()

data_random_main = list()
for i in range(len(a)):
    data_random_main.append(np.random.random_sample([ train_size + test_size + val_size,2]) )
    train_data.append(np.zeros([ train_size,2]))
    test_data.append(np.zeros([ test_size,2]))
    val_data.append(np.zeros([ val_size,2]))
    train_label.append(np.zeros([ train_size]))
    test_label.append(np.zeros([ test_size]))
    val_label.append(np.zeros([ val_size]))


for l in range(len(a)):
    k = 0
    for i in range(train_size):
        #print(train_data[l][i][0])
        #print(data_random_main[l][k][0])
        #print(train_label[l][i])
        print(l)
        train_data[l][i][0] = data_random_main[l][k][0] #+ 1)/2
        train_data[l][i][1] = data_random_main[l][k][1] #+ 1)/2
        train_label[l][i] = -a[l]*(train_data[l][i][0]**2 - train_data[l][i][0] + train_data[l][i][1]**2 - train_data[l][i][1])     + 0.01*0.5*a[l]*(train_data[l][i][0]*(train_data[l][i][0]-1)*train_data[l][i][1]*(train_data[l][i][1]-1))**3
        k = k + 1
    print(train_data)
    
    for i in range(test_size):
        test_data[l][i][0] = data_random_main[l][k][0] #+ 1)/2
        test_data[l][i][1] = data_random_main[l][k][1] #+ 1)/2
        test_label[l][i] = -a[l]*(test_data[l][i][0]**2 - test_data[l][i][0] + test_data[l][i][1]**2 - test_data[l][i][1]) + 0.01*0.5*a[l]*(test_data[l][i][0]*(test_data[l][i][0]-1)*test_data[l][i][1]*(test_data[l][i][1]-1))**3  
        k = k + 1
    print(test_label)

    for i in range(val_size):
        val_data[l][i][0] = data_random_main[l][k][0] #+ 1)/2
        val_data[l][i][1] = data_random_main[l][k][1] #+ 1)/2
        val_label[l][i] = -a[l]*(val_data[l][i][0]**2 - val_data[l][i][0] + val_data[l][i][1]**2 - val_data[l][i][1])  + 0.01*0.5*a[l]*(val_data[l][i][0]*(val_data[l][i][0]-1)*val_data[l][i][1]*(val_data[l][i][1]-1))**3 
        k = k + 1
    print(val_label)


## Boundary data
bound_rand_data = list()
bound_rand_data.append(np.random.random_sample([len(a), (train_size_b + test_size_b + val_size_b) ,1]))
bound_rand_data.append(np.random.random_sample([len(a), (train_size_b + test_size_b + val_size_b) ,1]))
bound_rand_data.append(np.random.random_sample([len(a), (train_size_b + test_size_b + val_size_b) ,1]))
bound_rand_data.append(np.random.random_sample([len(a), (train_size_b + test_size_b + val_size_b) ,1]))


train_data_b = list()
test_data_b = list()
val_data_b = list()
train_label_b = list()
test_label_b = list()
val_label_b = list()
for i in range(4):
    train_data_b.append(np.zeros([len(a), train_size_b,2]))
    test_data_b.append(np.zeros([len(a), test_size_b,2]))
    val_data_b.append(np.zeros([len(a), val_size_b,2]))
    train_label_b.append(np.zeros([len(a), train_size_b]))
    test_label_b.append(np.zeros([len(a), test_size_b]))
    val_label_b.append(np.zeros([len(a), val_size_b]))




for l in range(len(a)):
    k = 0
    for i in range(train_size_b):
        train_data_b[0][l][i][0] = bound_rand_data[0][l][k][0]
        train_label_b[0][l][i] = 0
        k = k + 1
    print(train_data_b)
    #k = 0
    for i in range(test_size_b):
        test_data_b[0][l][i][0] = bound_rand_data[0][l][k][0]
        test_label_b[0][l][i] = 0
        k = k + 1
    print(test_data_b)
    #k = 0
    for i in range(val_size_b):
        val_data_b[0][l][i][0] = bound_rand_data[0][l][k][0]
        val_label_b[0][l][i] = 0
        k = k + 1
    print(val_data_b)
    
    k = 0
    for i in range(train_size_b):
        train_data_b[1][l][i][0] = bound_rand_data[1][l][k][0]
        train_data_b[1][l][i][1] = 1
        train_label_b[1][l][i] = 0
        k = k + 1
    print(train_data_b)
    #k = 0
    for i in range(test_size_b):
        test_data_b[1][l][i][0] = bound_rand_data[1][l][k][0]
        test_data_b[1][l][i][1] = 1
        test_label_b[1][l][i] = 0
        k = k + 1
    print(test_data_b)
    #k = 0
    for i in range(val_size_b):
        val_data_b[1][l][i][0] = bound_rand_data[1][l][k][0]
        val_data_b[1][l][i][1] = 1
        val_label_b[1][l][i] = 0
        k = k + 1
    print(val_data_b)

    k = 0
    for i in range(train_size_b):
        train_data_b[2][l][i][1] = bound_rand_data[2][l][k][0]
        train_label_b[2][l][i] = 0
        k = k + 1
    print(train_data_b)
    #k = 0
    for i in range(test_size_b):
        test_data_b[2][l][i][1] = bound_rand_data[2][l][k][0]
        test_label_b[2][l][i] = 0
        k = k + 1
    print(test_data_b)
    #k = 0
    for i in range(val_size_b):
        val_data_b[2][l][i][1] = bound_rand_data[2][l][k][0]
        val_label_b[2][l][i] = 0
        k = k + 1
    print(val_data_b)
    
    k = 0
    for i in range(train_size_b):
        train_data_b[3][l][i][1] = bound_rand_data[3][l][k][0]
        train_data_b[3][l][i][0] = 1
        train_label_b[3][l][i] = 0
        k = k + 1
    print(train_data_b)
    #k = 0
    for i in range(test_size_b):
        test_data_b[3][l][i][1] = bound_rand_data[3][l][k][0]
        test_data_b[3][l][i][0] = 1
        test_label_b[3][l][i] = 0
        k = k + 1
    print(test_data_b)
    #k = 0
    for i in range(val_size_b):
        val_data_b[3][l][i][1] = bound_rand_data[3][l][k][0]
        val_data_b[3][l][i][0] = 1
        val_label_b[3][l][i] = 0
        k = k + 1
    print(val_data_b)
        
 


print(len(train_label))
    
data = {'a': a, 'train_data': train_data, 'train_label':  train_label,'test_data': test_data, 'test_label':  test_label,'val_data': val_data, 'val_label':  val_label, 'train_data_b': train_data_b, 'train_label_b':  train_label_b,'test_data_b': test_data_b, 'test_label_b':  test_label_b,'val_data_b': val_data_b,'val_label_b': val_label_b}

filename = './poisson2D_multi_nonlinear_[10-100].mat'
# filename = 'sims.mat'
sio.savemat(filename, {'data': data})

#train_data =
#train_label = 
