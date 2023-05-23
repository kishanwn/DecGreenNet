#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:00:46 2023

@author: kishan
"""

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F


class activation_ext(nn.Module):
    
    def __init__(self, act='sin'):
        super(activation_ext, self).__init__()
        self.act = act
        
        if act=='silu':
            self.act_fun = nn.SiLU()
        if act=='tanh':
            self.act_fun = nn.Tanh()    
       
        
    def forward(self, input):
        if self.act == 'sin':
            input = torch.sin(input)
        elif self.act == 'silu':
            input = self.act_fun(input)
        elif self.act == 'tanh':
            input = self.act_fun(input)
        elif self.act == 'cos':
            input = torch.cos(input)
        elif self.act == 'silu_sin':
            input = F.silu(input) + torch.sin(input)
        elif self.act == 'silu_sin':
            input = F.tanh(input) + torch.sin(input)
        elif self.act == 'exp':
            input = torch.exp(input)
        elif self.act == 'relu2':
            input = torch.clamp(input,min=0)**2
        elif self.act == 'relu3':
            input = torch.clamp(input,min=0)**3
        elif self.act == 'relu4':
            input = torch.clamp(input,min=0)**4
        elif self.act == 'relu5':
            input = torch.clamp(input,min=0)**5
        elif self.act == 'relu6':
            input = torch.clamp(input,min=0)**6
        elif self.act == 'relu7':
            input = torch.clamp(input,min=0)**7
        else:
            print("incorrect activation name")
            input = input
        
        return input