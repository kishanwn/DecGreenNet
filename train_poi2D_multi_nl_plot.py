#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:31:40 2022

@author: kishan
"""
import time
import random
import argparse
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from model_DecGreenNet import * 
from equation_models import * 
from activations_ext import * 
from data_sampler import * 
import matplotlib.pyplot as plt
import scipy.io as sio
import uuid
import torch.optim as optim
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')#1000
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=4, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='poisson2D_multi_[10-100]', help='dateset')
parser.add_argument('--num_quad', type=int, default=100, help='number of quadrature points')
parser.add_argument('--dev', type=int, default=2, help='device id')
parser.add_argument('--act_type_x', type=str, default='relu3', help='activation on input x')
parser.add_argument('--act_type_quad', type=str, default='relu3', help='activation on input quadrature points')
parser.add_argument('--act_type_last', type=str, default='relu3', help='activation on output layer')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--full', type=int, default=0)
parser.add_argument('--opt_method', type=str, default='Adam')

parser.add_argument('--r', type=int, default=50)

#LBFGS parameters
parser.add_argument('--history_size', type=int, default=100)
parser.add_argument('--max_iter', type=int, default=10)
parser.add_argument('--line_search_fn', type=str, default=None)# 'strong_wolfe'
args = parser.parse_args()

batch_size = 1024 # 512
hidden = [64] #[64,128,256,512]
lr = 0.001
wd = [0] #, 0.000001,0.001]
drop = [0.0] #, 0.1,0.5,0.7]
reg_para1 = [1]
reg_para2 = [1] #,0.01,0.0001] # 1e-2,1e-3,1e-4]
no_samples = 1
num_quad_x =10
epoch = 5000
x_dim = 2
#full_order = 5
#x_num = 50
r = args.r
num_quad=args.num_quad
net_config_X_arr = list()
#net_config_X_arr.append([x_dim, 8,8,  r])
#net_config_X_arr.append([x_dim, 8,8,8,  r])
#net_config_X_arr.append([x_dim, 16,16,  r])
#net_config_X_arr.append([x_dim, 16,16,16,  r])
#net_config_X_arr.append([x_dim, 32,32,  r])
#net_config_X_arr.append([x_dim, 32,32,32,  r])
#net_config_X_arr.append([x_dim, 64,64, r])
#net_config_X_arr.append([x_dim, 64,64,64,  r])
#net_config_X_arr.append([x_dim, 128,128,  r])
#net_config_X_arr.append([x_dim, 128,128,128,128,128,  r])
net_config_X_arr.append([x_dim, 128,128,128,128,128,  r])

net_config_outer_arr = list()
#net_config_outer_arr.append([2*x_dim*(x_num) + r, 32,  1])
#net_config_outer_arr.append([x_dim, 32,32,32,r])
net_config_outer_arr.append([x_dim, 64,64,64,64,64, r])

#hidden = 32

net_config_last_arr = list()
#net_config_outer_arr.append([2*x_dim*(x_num) + r, 32,  1])
net_config_last_arr.append([num_quad ,1])


fname = 'data_new/poisson2D_multi_[10-100].mat'

  

res_array = np.zeros([len(wd) ,len(drop) ,len(reg_para1)  ,len(reg_para2)  ,  no_samples])
cudaid =   "cuda:" + str(args.dev)   # "cpu" #
device = torch.device(cudaid)
checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
print(cudaid, checkpt_file)

#samples
sampler = Data_Sampler(fname,batch_size=batch_size, num_b=4, device=device)
sampler.load_eq_param()
sampler.load_interior()
sampler.load_boundary()   


# initialize varaibles
var_train_data = list() 
var_test_data = list() 
var_val_data = list() 

for i in range(len(sampler.eq_param)):
    var_train_data.append(Variable(sampler.train_data_i[i] , requires_grad = True).to(device))
    var_test_data.append(Variable(sampler.test_data_i[i] , requires_grad = True).to(device))
    var_val_data.append(Variable(sampler.val_data_i[i] , requires_grad = True).to(device))


q_x = np.random.rand(num_quad,2)
quad_x = torch.FloatTensor(q_x).to(device)
print(quad_x)

for net_config_X in net_config_X_arr:
    for net_config_outer in net_config_outer_arr:
        for net_config_last in net_config_last_arr:
            for wd_ in wd:
                for drop_ in drop:
                    for reg_p1 in reg_para1:
                        for reg_p2 in reg_para2:
                            #for reg_p3 in reg_para3:
            
                            random.seed(args.seed)
                            np.random.seed(args.seed)
                            torch.manual_seed(args.seed)
                            torch.cuda.manual_seed(args.seed)
                    
                            for j in range(no_samples):
                    
                                def train_step(model,sampler, optimizer):
                                    loss = nn.MSELoss()
                                    model.train()
                                    
                                    optimizer.zero_grad()
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x,y,rnd_s_list  = sampler.rnd_sample(data_trpe = 'train')
                                        else:
                                            x,y,rnd_s_list  = sampler.full_indexes(data_trpe = 'train')
                                    elif args.opt_method == 'LBFGS':
                                        x,y,rnd_s_list = sampler.full_indexes(data_trpe = 'train')
                                    
                                    loss_train = 0
                                    for i in range(len(sampler.eq_param)): #len(a)):
                                        #print(rnd_s_list[i])
                                        #print(var_train_data[i])
                                        d = model(var_train_data[i],sampler.eq_param[i],quad_x)
                                        
                                        d = torch.autograd.grad(d,var_train_data[i] , grad_outputs=torch.ones_like(d), create_graph=True )
                                        dx = d[0][:,0].reshape(-1,1)
                                        dy = d[0][:,1].reshape(-1,1)
                                        dxx = torch.autograd.grad(dx, var_train_data[i] , grad_outputs=torch.ones_like(dx), create_graph=True )[0][:,0].reshape(-1,1)
                                        dyy = torch.autograd.grad(dy, var_train_data[i] , grad_outputs=torch.ones_like(dy), create_graph=True )[0][:,1].reshape(-1,1)
                                        loss_train = loss_train + reg_p1*loss(-(dxx[rnd_s_list[i]] + dyy[rnd_s_list[i]]).squeeze(), y[i].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                    loss_train = loss_train/len(sampler.eq_param)
                                    
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_sample_b(data_trpe = 'train')
                                        else:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'train')
                                    elif args.opt_method == 'LBFGS':
                                        x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'train')
                                    #x_b,y_b,rnd_sample_b = sampler.rnd_sample_b(data_trpe = 'train')
                                    for j in range(4):
                                        temp = 0
                                        for i in range(len(sampler.eq_param)): #len(a)):
                                            #print(x_b[(i-1)*4+j])
                                            #print(x_b[(i-1)*4+j])
                                            d = model(x_b[(i-1)*4+j], sampler.eq_param[i], quad_x)
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(i-1)*4+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                        #print(temp)
                                        loss_train = loss_train + temp/len(sampler.eq_param)
                                    #'''
                                    #print("Train loss :" + str(loss_train.item()) )
                                    loss_train.backward()
                                    optimizer.step()
                                    return loss_train.item() #,acc_train.item()
                    
                                def test_step(model,sampler):
                                    model.eval()
                                   # with torch.no_grad():
                                    loss = nn.MSELoss()
                                    #model.eval()
                                    loss_test_ = list() 
                                    for k in range(len(sampler.eq_param)):
                                        if args.opt_method == 'Adam':
                                            if args.full == 0:
                                                x,y,rnd_s_list  = sampler.rnd_sample(data_trpe = 'test')
                                            else:
                                                x,y,rnd_s_list  = sampler.full_indexes(data_trpe = 'test')
                                        elif args.opt_method == 'LBFGS':
                                            x,y,rnd_s_list = sampler.full_indexes(data_trpe = 'test')
            
                                        loss_test = 0
                                        #for i in range(len(sampler.eq_param)): #len(a)):
                                        d = model(var_test_data[k],sampler.eq_param[k],quad_x)
                                        d = torch.autograd.grad(d,var_test_data[k] , grad_outputs=torch.ones_like(d), create_graph=True )
                                        dx = d[0][:,0].reshape(-1,1)
                                        dy = d[0][:,1].reshape(-1,1)
                                        dxx = torch.autograd.grad(dx, var_test_data[k] , grad_outputs=torch.ones_like(dx), create_graph=True )[0][:,0].reshape(-1,1)
                                        dyy = torch.autograd.grad(dy, var_test_data[k] , grad_outputs=torch.ones_like(dy), create_graph=True )[0][:,1].reshape(-1,1)
                                        loss_test = loss_test + reg_p1*loss(-(dxx[rnd_s_list[k]] + dyy[rnd_s_list[k]]).squeeze(), y[k].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                        #loss_test = loss_test
                                        
                                        
                                        if args.opt_method == 'Adam':
                                            if args.full == 0:
                                                x_b,y_b,rnd_sample_b  = sampler.rnd_sample_b(data_trpe = 'test')
                                            else:
                                                x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'test')
                                        elif args.opt_method == 'LBFGS':
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'test')
            
                                        temp = 0
                                        for j in range(4):
                                            d = model(x_b[(k-1)*4+j], sampler.eq_param[k], quad_x)
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(k-1)*4+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                            loss_test = loss_test + temp
                                        print("data set " + str(k) + " test error " + str(loss_test.item()) )
                                        loss_test_.append(loss_test.item())
                                    
                                    
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x,y,rnd_s_list  = sampler.rnd_sample(data_trpe = 'test')
                                        else:
                                            x,y,rnd_s_list  = sampler.full_indexes(data_trpe = 'test')
                                    elif args.opt_method == 'LBFGS':
                                        x,y,rnd_s_list = sampler.full_indexes(data_trpe = 'test')
        
                                    loss_test = 0
                                    for i in range(len(sampler.eq_param)): #len(a)):
                                        d = model(var_test_data[i],sampler.eq_param[i],quad_x)
                                        d = torch.autograd.grad(d,var_test_data[i] , grad_outputs=torch.ones_like(d), create_graph=True )
                                        dx = d[0][:,0].reshape(-1,1)
                                        dy = d[0][:,1].reshape(-1,1)
                                        dxx = torch.autograd.grad(dx, var_test_data[i] , grad_outputs=torch.ones_like(dx), create_graph=True )[0][:,0].reshape(-1,1)
                                        dyy = torch.autograd.grad(dy, var_test_data[i] , grad_outputs=torch.ones_like(dy), create_graph=True )[0][:,1].reshape(-1,1)
                                        loss_test = loss_test + reg_p1*loss(-(dxx[rnd_s_list[i]] + dyy[rnd_s_list[i]]).squeeze(), y[i].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                    loss_test = loss_test/len(sampler.eq_param)
                                    
                                    
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_sample_b(data_trpe = 'test')
                                        else:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'test')
                                    elif args.opt_method == 'LBFGS':
                                        x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'test')
        
                                    for j in range(4):
                                        temp = 0
                                        for i in range(len(sampler.eq_param)): #len(a)):
                                            d = model(x_b[(i-1)*4+j], sampler.eq_param[i], quad_x)
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(i-1)*4+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                        loss_test = loss_test + temp/len(sampler.eq_param)
            
                                    return loss_test.item(), loss_test_ #,acc_train.item()
               
                                def val_step(model, sampler ):
                                    model.eval()
                                    #with torch.no_grad():
                                    loss = nn.MSELoss()
                                    #model.eval()
                                    
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x,y,rnd_s_list  = sampler.rnd_sample(data_trpe = 'val')
                                        else:
                                            x,y,rnd_s_list  = sampler.full_indexes(data_trpe = 'val')
                                    elif args.opt_method == 'LBFGS':
                                        x,y,rnd_s_list = sampler.full_indexes(data_trpe = 'val')
                                        
                                    loss_val = 0
                                    for i in range(len(sampler.eq_param)): #len(a)):
                                        #print(rnd_s_list[i])
                                        #print(var_train_data[i])
                                        d = model(var_val_data[i],sampler.eq_param[i],quad_x)
                                        
                                        d = torch.autograd.grad(d,var_val_data[i] , grad_outputs=torch.ones_like(d), create_graph=True )
                                        dx = d[0][:,0].reshape(-1,1)
                                        dy = d[0][:,1].reshape(-1,1)
                                        dxx = torch.autograd.grad(dx, var_val_data[i] , grad_outputs=torch.ones_like(dx), create_graph=True )[0][:,0].reshape(-1,1)
                                        dyy = torch.autograd.grad(dy, var_val_data[i] , grad_outputs=torch.ones_like(dy), create_graph=True )[0][:,1].reshape(-1,1)
                                        loss_val = loss_val + reg_p1*loss(-(dxx[rnd_s_list[i]] + dyy[rnd_s_list[i]]).squeeze(), y[i].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                    loss_val = loss_val/len(sampler.eq_param)
                                    
                                    #'''
                                                                            
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_sample_b(data_trpe = 'val')
                                        else:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'val')
                                    elif args.opt_method == 'LBFGS':
                                        x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'val')
                                    
                                    for j in range(4):
                                        temp = 0
                                        for i in range(len(sampler.eq_param)): #len(a)):
                                            d = model(x_b[(i-1)*4+j], sampler.eq_param[i], quad_x)
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(i-1)*4+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                        #print(temp)
                                        loss_val = loss_val + temp/len(sampler.eq_param)
                    
                                    #print("Val loss :" + str(loss_val.item()) )
            
                                    return loss_val.item() #,acc_train.item()
                              
                    
                                def train(): #datastr,splitstr):
                                    
                                    mlp_x_act = activation_ext(act = args.act_type_x).to(device)
                                    mlp_x= nn_mlp_custom_act(net_config_X,mlp_x_act).to(device)
                                    
                                    mlp_quad_act = activation_ext(act = args.act_type_quad).to(device)
                                    mlp_quad= nn_mlp_custom_act(net_config_outer,mlp_quad_act).to(device)
                                    #mlp_quad= nn_mlp_custom_act(net_config_outer,mlp_quad_act)
                                    
                                    r_func = Poi2d_with_a(device=device).to(device)
                                    
                                    mlp_outer_act = activation_ext(act = args.act_type_last).to(device)
                                    mlp_outer = nn_mlp_last_na_custom_act(net_config_last ,mlp_outer_act).to(device)
                                    
                                    model = DecGreenNet_nonlinear(mlp_x,  r_func ,mlp_quad,mlp_outer).to(device)
                                    #model =Green_LR_mlp_gen_direct_low_rank(mlp_x,  r_func ,mlp_quad)
    
                                    if args.opt_method == 'Adam':
                                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd_)
                                    elif args.opt_method == 'LBFGS':
                                        optimizer = optim.LBFGS(model.parameters(), history_size=args.history_size, max_iter=args.max_iter) #lr=lr, history_size=args.history_size, max_iter=args.max_iter, line_search_fn=args.line_search_fn)lr=lr, history_size=args.history_size, max_iter=args.max_iter, line_search_fn=args.line_search_fn)
                    
                                    bad_counter = 0
                                    best = 999999999
                                    for epoch in range(args.epochs):
                                        #vx_rhs = torch.zeros(full_order)
                                        #vx_rhs[0] =  quadrature_para[0]
                                        #vx_rhs = vx_rhs.to(device)
                                        
                                        loss_tra = train_step(model,sampler, optimizer) # , vx_rhs)
                                        loss_val = val_step(model,sampler) #,  vx_rhs)
                                        #print("Train Loss: " + str(loss_tra))
                                        if np.mod(epoch,100) == 0:
                                            print("Train Loss: " + str(loss_tra) + " Validation Loss: " + str(loss_val))
                                        
                                        #'''
                                        if loss_val < best:
                                            best = loss_val
                                            torch.save(model.state_dict(), checkpt_file)
                                            bad_counter = 0
                                        else:
                                            bad_counter += 1
                    
                                        if bad_counter == args.patience:
                                            break
                                        #'''
                                    
                                    model.load_state_dict(torch.load(checkpt_file))
                                    loss_val = val_step(model,sampler) #, vx_rhs)
                                    ##loss_train = tr_step(model,a, train_data,train_label, train_data_b ,train_label_b)
                                    loss_test, loss_test_ = test_step(model,sampler)
                                    
                                    
                                    a_ = 15
                                    x = np.linspace(0,1,101)
                                    y = np.linspace(0,1,101)
                                    #a_ =  torch.FloatTensor(a_).to(device)
                                    print(x)
                                    
                                    quad_x1 = torch.FloatTensor(torch.from_numpy(x).float()).to(device)
                                    
                                    quad_y1 = torch.FloatTensor(torch.from_numpy(y).float()).to(device)
                                    u = np.zeros([101,101])
                                    u1 = np.zeros([101,101])
                                    x_ = np.zeros([1,2])
                                    for i in range(101):
                                        for j in range(101):
                                            x_[0,0] = x[i]
                                            x_[0,1] = y[j]
                                            xx_ = torch.FloatTensor(torch.from_numpy(x_).float()).to(device)
                                            u[i,j] =  0.5*a_*x[i]*(x[i]-1)*x[j]*(x[j]-1)
                                            u1[i,j] = model(xx_,a_,quad_x)
                                       
                                           
                                    plot_str = 'Poisson2D_gridsearch_10_100_new/DescGreenNet_nonlinear_quad_' + str(num_quad) + '_rank_' + str(r) +  '/plot/'
                                    ext_name = str(a_) + '_' + str(r) + '_' + str(num_quad)
                                    
                                    print(u)    
                                        
                                    fig = plt.figure()
                                    
                                    plt.imshow(u1,cmap=cm.coolwarm)
                                    plt.colorbar() #cax=cax)
                                    #plt.scatter(x, y, s=u, alpha=0.5)
                                    plt.xlabel('x')
                                    plt.ylabel('y')
                                    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) #np.arange(0, 100, step=10))
                                    plt.yticks([90, 80, 70, 60, 50, 40, 30, 20, 10,0], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])                             
                                    
                                   
                                    plt.savefig(plot_str + 'poi2d_multi_dec_prod_' + ext_name + '.eps' ,bbox_inches='tight')
                                    print(u1)
                                    
                                    u_ = u - u1
                                    print(u_)
                                    fig = plt.figure()
                                    #ax = fig.add_subplot(1, 1, 1, projection='2d')
                                    plt.imshow(u_,cmap=cm.coolwarm)
                                    plt.colorbar() #cax=cax)
                                    #plt.scatter(x, y, s=u, alpha=0.5)
                                    plt.xlabel('x')
                                    plt.ylabel('y')
                                    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) #np.arange(0, 100, step=10))
                                    plt.yticks([90, 80, 70, 60, 50, 40, 30, 20, 10,0], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
                                    plt.savefig(plot_str + 'poi2d_multi_dec_prod_diff_' + ext_name + '.eps',bbox_inches='tight')
                                    
                                    
                                    fig = plt.figure()
                                    #ax = fig.add_subplot(1, 1, 1, projection='2d')
                                    plt.imshow(u,cmap=cm.coolwarm)
                                    plt.colorbar() #cax=cax)
                                    #plt.scatter(x, y, s=u, alpha=0.5)
                                    plt.xlabel('x')
                                    plt.ylabel('y')
                                    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) #np.arange(0, 100, step=10))
                                    plt.yticks([90, 80, 70, 60, 50, 40, 30, 20, 10,0], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
                                    plt.savefig(plot_str + 'poi2d_' + ext_name + '.eps',bbox_inches='tight')
                                    
                                    
                                    return loss_val,loss_test, loss_test_ #acc_val
                                
            
    
                                loss_val,loss_test, loss_test_ = train() 
                                print("Test loss: " + str(loss_test_) )
                                
                                res_array[wd.index(wd_),  drop.index(drop_),  reg_para1.index(reg_p1), reg_para2.index(reg_p2),      j] = loss_val
                                
                                #filename = './Poisson2D_gridsearch_10_100_new/DescGreenNet_nonlinear_quad_' + str(num_quad) + '_rank_' + str(r) +  '/'+str(args.data) + '_' + str(args.seed)   + '_' + str(lr)     + '_' + str(wd)  + '_' + str(drop)   + '_' + str(reg_p1)   + '_' + str(reg_p2)   + '_' + str(net_config_X)   + '_'   + str(net_config_outer)  + '_'   + str(net_config_last)  + '_'   + str(r)  + '_' + str(args.act_type_x) + '_' + str(args.act_type_quad)  + '_' + str(args.act_type_last) + '_'   + str(num_quad)  + '_' + str(loss_val)  + '_' + str(loss_test)   + '.mat'
    
                                #sio.savemat(filename, {'res': res_array, 'q_x':q_x })
                                #print(res_array)
                                
