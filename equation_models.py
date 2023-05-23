#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:25:30 2023

@author: kishan
"""

import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Poi2d(nn.Module):
    
    def __init__(self, a=1,device=0):
        super(Poi2d, self).__init__()
        self.a = a
        self.device = device


    def forward(self,input):
        sx = input.size()
        
        out = torch.zeros([sx[0],1]).to(self.device)
        for i in range(sx[0]):
            out[i,0] =  -self.a*( input[i,0]**2 - input[i,0] + input[i,1]**2 - input[i,1]) 
           
        return out  
        
class Poi2d_with_a_nonlinear(nn.Module):
    
    def __init__(self,device=0):
        super(Poi2d_with_a, self).__init__()
        self.device = device


    def forward(self,input,a):
        sx = input.size()
        
        out = torch.zeros([sx[0],1]).to(self.device)
        for i in range(sx[0]):
            out[i,0] =  -a*( input[i,0]**2 - input[i,0] + input[i,1]**2 - input[i,1]) + 0.01*0.5*a[l]*(input[i,0]*(input[i,0]-1)*input[i,1]*(input[i,1]-1))**3
           
        return out  
    
class Poi2d_with_a(nn.Module):
    
    def __init__(self,device=0):
        super(Poi2d_with_a, self).__init__()
        self.device = device


    def forward(self,input,a):
        sx = input.size()
        
        out = torch.zeros([sx[0],1]).to(self.device)
        for i in range(sx[0]):
            out[i,0] =  -a*( input[i,0]**2 - input[i,0] + input[i,1]**2 - input[i,1]) 
           
        return out  
    
class Helm_with_a(nn.Module):
    
    def __init__(self,p=1, device=0):
        super(Helm_with_a, self).__init__()
        self.device = device
        self.p = p
        self.pi = torch.acos(torch.zeros(1)).item()*2
    def forward(self,input,a):
        sx = input.size()
        #print(self.pi)
        out = torch.zeros([sx[0],1]).to(self.device)
        for i in range(sx[0]):
            out[i,0] =  -(self.pi**2)*torch.sin(self.pi*input[i,0])*torch.sin(self.p*self.pi*input[i,1])  -((self.p*self.pi)**2)*math.sin(self.pi*input[i,0])*torch.sin(self.p*self.pi*input[i,1]) + (a**2)*torch.sin(self.pi*input[i,0])*torch.sin(self.p*self.pi*input[i,1])
           
        return out  
    
class Helm_with_a_b(nn.Module):
    
    def __init__(self, device=0):
        super(Helm_with_a_b, self).__init__()
        self.device = device

        self.pi = torch.acos(torch.zeros(1)).item()*2
    
    def forward(self,input,a,b):
        sx = input.size()
        #print(self.pi)
        out = torch.zeros([sx[0],1]).to(self.device)
        for i in range(sx[0]):
            out[i,0] =  -((a*self.pi)**2)*torch.sin(a*self.pi*input[i,0])*torch.sin(b*self.pi*input[i,1])  -((b*self.pi)**2)*math.sin(a*self.pi*input[i,0])*torch.sin(b*self.pi*input[i,1]) + torch.sin(a*self.pi*input[i,0])*torch.sin(b*self.pi*input[i,1])
           
        return out  
    
##-- Uncertainty control--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Uncert_control_MODNET(nn.Module):
   
    def __init__(self, device=0):
        super(Uncert_control_MODNET, self).__init__()
        self.device = device
        
        self.pi = torch.acos(torch.zeros(1)).item()*2
    
    def forward(self,input,a):
        sx = input.size()
        #print(self.pi)
        out = torch.zeros([sx[0],1]).to(self.device)
        for i in range(sx[0]):
            x = input[i,0]
            y = input[i,1]
            if 1 <= y < 1.5:
                out[i,0]  = (y**2 - 3*y + 2)*(x**2 - x + (2*x -1 + (x*(x-1)*(2 + math.sin(4*self.pi*x*y)))/y )*(x + math.cos(self.pi*y)) )  
            elif 1.5 <= y < 2:
                out[i,0]  = (y**2 - 3*y + 2)*(x**2 - x + (2*x -1 + (x*(x-1)*(2 + math.sin(4*self.pi*x*y)))/y )*(x + math.sin(2*self.pi*y)) )
            
        return out  
    
##-- RF from GR-net -----------------------------------------------------------------------------------------------------------------------------------------------------

class RF_eq(nn.Module):
   
    def __init__(self, device=0):
        super(RF_eq, self).__init__()
        self.device = device
        
    
    def forward(self,input,a=1):
        sx = input.size()
        #print(self.pi)
        out = torch.zeros([sx[0],1]).to(self.device)
        for i in range(sx[0]):
            x = input[i,0]
            y = input[i,1]
            u = math.exp(-(x**2 + 2*(y**2) + 1) )
            du1 = -u*2*x 
            du2 = -u*4*y
            du11 = (u**2)*4*(x**2) - u*2  
            du22 = (u**2)*16*(y**2) - u*4 
            out[i] = -( (du11 + du22)*(1 + 2*y**2) + 4*y*(du2) ) + (1 + x**2)*u
            
        return out  