#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:57:31 2023

@author: kishan
"""

import numpy as np
import math
import torch
import scipy.io as sio



class Data_Sampler:

    def __init__(self, fname,batch_size, num_b, device):
        self.data = sio.loadmat(fname)
        self.batch_size = batch_size
        self.device = device
        self.num_b =num_b  
        # data for equation parameter e.g. a for Poisson
        self.eq_param = list()
        
        # data for interior
        self.train_data_i = list()
        self.test_data_i = list()
        self.val_data_i = list()
        self.train_label_i = list()
        self.test_label_i = list()
        self.val_label_i = list()
        
        # data for boundary
        self.train_data_b = list()
        self.test_data_b = list()
        self.val_data_b = list()
        self.train_label_b = list()
        self.test_label_b = list()
        self.val_label_b = list()

        # data for contraol
        self.train_data_control = list()
        self.test_data_control = list()
        self.val_data_control = list()        

        # data for exact data
        self.train_data_exact = list()
        self.train_label_exact = list()
         
        
    def load_eq_param(self):        
        self.eq_param = list(self.data['data']['a'][0][0][0])
        
    def load_interior(self,train=1,test=1,val=1):
        for i in range(len(self.eq_param)):
            if train==1:
                self.train_data_i.append(torch.FloatTensor(torch.from_numpy(self.data['data']['train_data'][0][0][i]).float()).to(self.device))
                self.train_label_i.append(torch.FloatTensor(torch.from_numpy(self.data['data']['train_label'][0][0][i]).float()).to(self.device))
            if test==1:    
                self.test_data_i.append(torch.FloatTensor(torch.from_numpy(self.data['data']['test_data'][0][0][i]).float()).to(self.device))
                self.test_label_i.append(torch.FloatTensor(torch.from_numpy(self.data['data']['test_label'][0][0][i]).float()).to(self.device))
            if val==1:    
                self.val_data_i.append(torch.FloatTensor(torch.from_numpy(self.data['data']['val_data'][0][0][i]).float()).to(self.device))
                self.val_label_i.append(torch.FloatTensor(torch.from_numpy(self.data['data']['val_label'][0][0][i]).float()).to(self.device))

    def load_boundary(self,train=1,test=1,val=1):
      
        for i in range(len(self.eq_param)):
            for j in range(self.num_b):
                if train==1:
                    self.train_data_b.append(torch.FloatTensor(torch.from_numpy(self.data['data']['train_data_b'][0][0][j][i]).float()).to(self.device))
                    self.train_label_b.append(torch.FloatTensor(torch.from_numpy(self.data['data']['train_label_b'][0][0][j][i]).float()).to(self.device))
                if test==1:  
                    self.test_data_b.append(torch.FloatTensor(torch.from_numpy(self.data['data']['test_data_b'][0][0][j][i]).float()).to(self.device))
                    self.test_label_b.append(torch.FloatTensor(torch.from_numpy(self.data['data']['test_label_b'][0][0][j][i]).float()).to(self.device))
                if val==1:    
                    self.val_data_b.append(torch.FloatTensor(torch.from_numpy(self.data['data']['val_data_b'][0][0][j][i]).float()).to(self.device))
                    self.val_label_b.append(torch.FloatTensor(torch.from_numpy(self.data['data']['val_label_b'][0][0][j][i]).float()).to(self.device))
                    
    def load_control(self,train=1,test=1,val=1):
        for i in range(len(self.eq_param)):
            if train==1:
                self.train_data_control.append(torch.FloatTensor(torch.from_numpy(self.data['data']['train_data_control'][0][0][i]).float()).to(self.device))
            if test==1:  
                self.test_data_control.append(torch.FloatTensor(torch.from_numpy(self.data['data']['test_data_control'][0][0][i]).float()).to(self.device))
            if val==1:    
                self.val_data_control.append(torch.FloatTensor(torch.from_numpy(self.data['data']['val_data_control'][0][0][i]).float()).to(self.device))

    def load_exact(self):
        #d = torch.from_numpy(self.data['data']['train_exact_data'][0][0]).float().to(self.device)
        for i in range(4): #len(d)):
            #print(self.data['data']['train_exact_data'][0][0][0][i])
            #print(torch.FloatTensor(torch.from_numpy(self.data['data']['train_exact_data'][0][0][i]).float()))
            self.train_data_exact.append(torch.FloatTensor(torch.from_numpy(self.data['data']['train_exact_data'][0][0][0][i]).float()).to(self.device))
            self.train_label_exact.append(torch.FloatTensor(torch.from_numpy(self.data['data']['train_exact_label'][0][0][0][i]).float()).to(self.device))

    def rnd_sample(self,data_trpe = 'train'):
        x = list()
        y = list()
        rnd_s_list = list()
        for i in range(len(self.eq_param)):
            if data_trpe=='train':
                sx = len(self.train_data_i[i])
                rnd_s = torch.randperm(sx)
                rnd_s_list.append(rnd_s[:self.batch_size])
                x.append(self.train_data_i[i][rnd_s[:self.batch_size]])
                y.append(self.train_label_i[i][rnd_s[:self.batch_size]])
            if data_trpe=='test':
                sx = len(self.test_data_i[i])
                rnd_s = torch.randperm(sx)
                rnd_s_list.append(rnd_s[:self.batch_size])
                x.append(self.test_data_i[i][rnd_s[:self.batch_size]])
                y.append(self.test_label_i[i][rnd_s[:self.batch_size]])    
            if data_trpe=='val':
                sx = len(self.val_data_i[i])
                rnd_s = torch.randperm(sx)
                rnd_s_list.append(rnd_s[:self.batch_size])
                x.append(self.val_data_i[i][rnd_s[:self.batch_size]])
                y.append(self.val_label_i[i][rnd_s[:self.batch_size]])
        return x, y, rnd_s_list
    
    def full_indexes(self,data_trpe = 'train'):
        x = list()
        y = list()
        rnd_s_list = list()
        for i in range(len(self.eq_param)):
            if data_trpe=='train':
                sx = len(self.train_data_i[i])
                rnd_s = np.arange(sx)  #torch.randperm(sx)
                rnd_s_list.append(rnd_s)
                x.append(self.train_data_i[i][rnd_s])
                y.append(self.train_label_i[i][rnd_s])
            if data_trpe=='test':
                sx = len(self.test_data_i[i])
                rnd_s = np.arange(sx) #torch.randperm(sx)
                rnd_s_list.append(rnd_s)
                x.append(self.test_data_i[i][rnd_s])
                y.append(self.test_label_i[i][rnd_s])    
            if data_trpe=='val':
                sx = len(self.val_data_i[i])
                rnd_s = np.arange(sx)  #torch.randperm(sx)
                rnd_s_list.append(rnd_s)
                x.append(self.val_data_i[i][rnd_s])
                y.append(self.val_label_i[i][rnd_s])
        return x, y, rnd_s_list
    
    def rnd_sample_b(self,data_trpe = 'train'):
        x = list()
        y = list()
        rnd_s_list = list()
        for i in range(len(self.eq_param)*self.num_b):
            if data_trpe=='train':
                sx = len(self.train_data_b[i])
                rnd_s = torch.randperm(sx)
                rnd_s_list.append(rnd_s[:self.batch_size])
                x.append(self.train_data_b[i][rnd_s[:self.batch_size]])
                y.append(self.train_label_b[i][rnd_s[:self.batch_size]])
            if data_trpe=='test':
                sx = len(self.test_data_b[i])
                rnd_s = torch.randperm(sx)
                rnd_s_list.append(rnd_s[:self.batch_size])
                x.append(self.test_data_b[i][rnd_s[:self.batch_size]])
                y.append(self.test_label_b[i][rnd_s[:self.batch_size]])    
            if data_trpe=='val':
                sx = len(self.val_data_b[i])
                rnd_s = torch.randperm(sx)
                rnd_s_list.append(rnd_s[:self.batch_size])
                x.append(self.val_data_b[i][rnd_s[:self.batch_size]])
                y.append(self.val_label_b[i][rnd_s[:self.batch_size]])
        return x, y, rnd_s_list
    
    def rnd_full_indexes_b(self,data_trpe = 'train'):
        x = list()
        y = list()
        rnd_s_list = list()
        for i in range(len(self.eq_param)*self.num_b):
            if data_trpe=='train':
                sx = len(self.train_data_b[i])
                rnd_s = np.arange(sx) # torch.randperm(sx)
                rnd_s_list.append(rnd_s)                
                x.append(self.train_data_b[i][rnd_s])
                y.append(self.train_label_b[i][rnd_s])
            if data_trpe=='test':
                sx = len(self.test_data_b[i])
                rnd_s = np.arange(sx)  #torch.randperm(sx)
                rnd_s_list.append(rnd_s)
                x.append(self.test_data_b[i][rnd_s])
                y.append(self.test_label_b[i][rnd_s])    
            if data_trpe=='val':
                sx = len(self.val_data_b[i])
                rnd_s = np.arange(sx)  #torch.randperm(sx)
                rnd_s_list.append(rnd_s)
                x.append(self.val_data_b[i][rnd_s])
                y.append(self.val_label_b[i][rnd_s])
        return x, y, rnd_s_list
           ##if data_trpe=='test':    
                
            #if data_trpe=='val':    
                
    #sx =     
    #torch.randperm(4)

        
if __name__ == '__main__':
    fname = 'poisson2D_multi_[10,20,30,40,50].mat'
    d = Data_Sampler(fname,batch_size=43, num_b=4, device="cpu")
    d.load_eq_param()
    d.load_interior()
    d.load_boundary()
    x,y = d.rnd_sample('test')
    print(x[0])
    print(x[1])
    x,y = d.rnd_sample_b('test')
    print(x[0])
    print(x[1])
    print(x[2])
    print(x[4])