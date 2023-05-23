#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 22:01:25 2023

@author: kishan
"""

import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from activations_ext import *


class nn_mlp_last_na(nn.Module):
    
    def __init__(self, net_dims,  dropout = 0.0, act_fun='Tanh',act_out_fun='Sigmoid'):
        super(nn_mlp_last_na, self).__init__()
        self.dropout = dropout
        self.act_fun = act_fun
        self.net_dims = net_dims
        self.drop = nn.Dropout(p=dropout)
        
        if act_fun=='ReLU':
            self.act = nn.ReLU()
        elif act_fun=='Sigmoid':
            self.act = nn.Sigmoid()
        elif act_fun=='Tanh':
            self.act = nn.Tanh()
        elif act_fun=='SiLU':
            self.act = nn.SiLU()
        elif act_fun=='Softplus':
            self.act = nn.Softplus()
        elif act_fun=='':
            self.act = nn.Identity()        
        #else:
        
        
        if act_out_fun=='ReLU':
            self.act_out = nn.ReLU()
        elif act_out_fun=='Sigmoid':
            self.act_out = nn.Sigmoid()
        elif act_out_fun=='Tanh':
            self.act_out = nn.Tanh()
        elif act_out_fun=='SiLU':
            self.act_out = nn.SiLU()
        elif act_out_fun=='Softplus':
            self.act_out = nn.Softplus()
        elif act_out_fun=='':
            self.act_out = nn.Identity()  
            
        self.layers_X = nn.ModuleList()
        
        self.num_layers_X =  len(net_dims)-1 #.size(1)
        
       
        for i in range(self.num_layers_X):
            self.layers_X.append(nn.Linear(self.net_dims[i],self.net_dims[i+1]) )

    def forward(self,  input):
        out = input
        for i in range(self.num_layers_X-1):
            #print(i)
            out =self.drop(out)    
            out = self.layers_X[i](out) 
            #print(torch.norm(self.layers_X[i].weight))
            out = self.act(out)
        out = self.layers_X[-1](out) 
            
        return out     

class nn_mlp(nn.Module):
    
    def __init__(self, net_dims,  dropout = 0.0, act_fun='Tanh',act_out_fun='Sigmoid'):
        super(nn_mlp, self).__init__()
        self.dropout = dropout
        self.act_fun = act_fun
        self.net_dims = net_dims
        self.drop = nn.Dropout(p=dropout)
        if act_fun=='ReLU':
            self.act = nn.ReLU()
        elif act_fun=='Sigmoid':
            self.act = nn.Sigmoid()
        elif act_fun=='Tanh':
            self.act = nn.Tanh()
        elif act_fun=='SiLU':
            self.act = nn.SiLU()
        elif act_fun=='Softplus':
            self.act = nn.Softplus()
        #elif act_fun=='exp':
        #    self.act = nn.Exp()
        elif act_fun=='':
            self.act = nn.Identity()        
        
        if act_out_fun=='ReLU':
            self.act_out = nn.ReLU()
        elif act_out_fun=='Sigmoid':
            self.act_out = nn.Sigmoid()
        elif act_out_fun=='Tanh':
            self.act_out = nn.Tanh()
        elif act_out_fun=='SiLU':
            self.act_out = nn.SiLU()
        elif act_out_fun=='Softplus':
            self.act = nn.Softplus()
       
        elif act_out_fun=='':
            self.act_out = nn.Identity()  
            
        self.layers_X = nn.ModuleList()
        
        self.num_layers_X =  len(net_dims)-1 #.size(1)
        
       
        for i in range(self.num_layers_X):
            self.layers_X.append(nn.Linear(self.net_dims[i],self.net_dims[i+1]) )
       
    def forward(self,  input):
        out = input
        for i in range(self.num_layers_X):
            #print(i)
            out =self.drop(out)    
            out = self.layers_X[i](out) 
            #print(torch.norm(self.layers_X[i].weight))
            out = self.act(out) #self.act(out)
            
        return out 

## -- mlp with custom activation -------------------------------------------------------------------------------------------------------------------    

class nn_mlp_last_na_custom_act(nn.Module):
    
    def __init__(self, net_dims,  act_fun,   dropout = 0.0,):
        super(nn_mlp_last_na_custom_act, self).__init__()
        self.dropout = dropout
        self.act_fun = act_fun
        self.drop = nn.Dropout(p=dropout)
        self.net_dims = net_dims        
                    
        self.layers_X = nn.ModuleList()
        
        self.num_layers_X =  len(net_dims)-1 #.size(1)
        
       
        for i in range(self.num_layers_X):
            self.layers_X.append(nn.Linear(self.net_dims[i],self.net_dims[i+1]) )

    def forward(self,  input):
        out = input
        for i in range(self.num_layers_X-1):
            #print(i)
            out =self.drop(out)    
            out = self.layers_X[i](out) 
            #print(torch.norm(self.layers_X[i].weight))
            out = self.act_fun(out)
        out = self.layers_X[-1](out) 
            
        return out  
    
class nn_mlp_custom_act(nn.Module):
    
    def __init__(self, net_dims,  act_fun, dropout = 0.0 ):
        super(nn_mlp_custom_act, self).__init__()
        self.dropout = dropout
        self.act_fun = act_fun
        self.net_dims = net_dims
        self.drop = nn.Dropout(p=dropout)
            
        self.layers_X = nn.ModuleList()
        
        self.num_layers_X =  len(net_dims)-1 #.size(1)
      
        for i in range(self.num_layers_X):
            self.layers_X.append(nn.Linear(self.net_dims[i],self.net_dims[i+1]) )
       
    def forward(self,  input):
        out = input
        for i in range(self.num_layers_X):
            #print(i)
            out =self.drop(out)    
            out = self.layers_X[i](out) 
            #print(torch.norm(self.layers_X[i].weight))
            out = self.act_fun(out) #self.act(out)
            
        return out     


## MOD-NET--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    
class Modnet(nn.Module):
    
    def __init__(self, mlp_x,  r_func , device='cpu', dropout = 0.0): #num_layers, 
        super(Modnet, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.device = device
        
        
    def forward(self,input, eq_param, quad_x ):

        
        size_quad = quad_x.size()
        size_input = input.size()
        out =  torch.zeros([size_input[0],1]).to(self.device)
        x = quad_x.repeat(size_input[0],1)
        y = self.r_func(x, eq_param)
        #print(x.size())
        z = input.repeat(1,size_quad[0]).reshape(size_quad[0]*size_input[0], size_input[1])
        #print(z.size())
        concat = torch.cat((x,z),1)
        lhs = self.mlp_x(concat)
        #concat  =  input.repeat()
        #print(lhs.size())
        rhs = lhs*y

        #print(rhs)
        for i in range(size_input[0]):
            out[i] = torch.sum(rhs[i*size_quad[0]:(i+1)*size_quad[0]])
        #print(out.size())
        return out

# Brute-force implementation
class Modnet2(nn.Module):
    
    def __init__(self, mlp_x,  r_func , device='cpu', dropout = 0.0): #num_layers, 
        super(Modnet2, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.device = device
       
        
    def forward(self,input, eq_param, quad_x ):
       
        size_quad = quad_x.size()
        size_input = input.size()
        #print(size_quad[0])
        #print(size_input)
        out = torch.zeros([size_input[0],1]).to(self.device)
        y = self.r_func(quad_x, eq_param)
        #print(out.size())
        for i in range(size_input[0]):
            for j in range(size_quad[0]):
                #print(i,j)
                out[i,0] = out[i,0] + self.mlp_x(torch.cat((input[i,:],quad_x[j,:]),0))*y[j]

        return out
    


'''    
if __name__ == '__main__':
    mlp = nn_mlp([2,10,2])
    m = Modnet(mlp)

    problem = DiffusionReaction(case=1)
    print(problem)
'''