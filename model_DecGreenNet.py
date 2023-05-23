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

#MLP with no activation on the last layer and customized activation functions
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


## DecGreenNet --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    
class DecGreenNet_product(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  dropout = 0.0): #num_layers, 
        super(DecGreenNet_product, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
       
        
    def forward(self,input, eq_param, quad_x ):

        y = self.r_func(quad_x, eq_param)
        rhs = self.mlp_quad(quad_x)

        rhs = rhs*y

        rhs = torch.sum(rhs,0)
      
        rx = rhs.size()
        lhs = self.mlp_x(input)

        out = lhs@rhs.T        

        return out

    
class DecGreenNet_nonlinear(nn.Module):
    
    def __init__(self, mlp_x,  r_func, mlp_quad, mlp_out, dropout = 0.0): #num_layers, 
        super(DecGreenNet_nonlinear, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_out = mlp_out
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
       
        
    def forward(self,input, eq_param, quad_x ):

        y = self.r_func(quad_x,eq_param)
        #print(y.size())
        rhs = self.mlp_quad(quad_x)
        #print(rhs.size())
        rhs = rhs*y
        
        lhs = self.mlp_x(input)
        
        out = lhs@rhs.T        

        out = self.mlp_out(out)
        
        return out