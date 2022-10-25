#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import pymatgen as mg
import pymatgen.analysis.diffraction as anadi
import pymatgen.analysis.diffraction.xrd as xrd
#import numpy as np
import glob
import matplotlib.pyplot as plt

#import torch
#import torch.nn as nn
#from torch.autograd import Variable

import math
import time


# In[2]:


import paddle_qcnn_5bits_5to4


# In[3]:


import paddle
from paddle import fluid
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import partial_trace, dagger, state_fidelity, NKron,pauli_str_to_matrix
from paddle import complex
from paddle.complex import matmul, trace, kron,elementwise_add


# In[4]:


import numpy as np
from sklearn import preprocessing


# datasets
# --
# 

# In[5]:


patt_xrd = xrd.XRDCalculator('CuKa')
person_path='/home/ipqi/.jupyter/20201020/'

train_path=person_path+'GcN-sur/train/'

test_path=person_path+'GcN-sur/test/'

global sample_num, rmat_num, series_num
sample_num=1 #output of G
rmat_num=32 #row nums of the matrix for the input of CNN

test_xrd=train_path+'00000'


# In[6]:


def tomgStructure(folder):
    POSfile=folder+'/POSCAR'      
    R_mgS=mg.Structure.from_file(POSfile)
    return R_mgS


# In[7]:


test_xrd2=tomgStructure(test_xrd)


# In[8]:


def get_xrdmat(mgStructure):
    global rmat_num
    xrd_data4 =patt_xrd.get_pattern(mgStructure)
    
    i_column = rmat_num
    xxx=[]
    yyy=[]
    mat4=[]
    xrd_i=len(xrd_data4)
    for i in range(xrd_i):
        if xrd_data4.y[i] > 1.5 and xrd_data4.y[i] < 70:
            xxx.append(xrd_data4.x[i])
            yyy.append(xrd_data4.y[i])
    mat4.append(np.asarray(xxx))
    mat4.append(np.asarray(yyy))
    mat4=np.asarray(mat4)
    
    xrd_x=[]
    xrd_y=[]
    xrd_mat4=[]
    xrow=len(mat4[0])
    
    if xrow < i_column:
        for i in mat4[0]:
            xrd_x.append(i)
        for j in mat4[1]:
            xrd_y.append(j)
        for i in range(0,i_column-xrow):
            xrd_x.append(0)
            xrd_y.append(0)
        xrd_x=np.asarray(xrd_x)
        xrd_y=np.asarray(xrd_y)
    if xrow > i_column:
        xrd_x=mat4[0][:i_column]
        xrd_y=mat4[1][:i_column]
    if xrow == i_column:
        xrd_x= mat4[0]
        xrd_y= mat4[1]
       
    #xrd_x=np.sin(np.dot(1/180*np.pi,xrd_x))
    #xrd_y=np.sqrt(np.dot(1/100,xrd_y))
    xrd_z=xrd_y/xrd_x
    #xrd_mat4.append(xrd_x)
    #xrd_mat4.append(xrd_y)
    xrd_mat4=np.array(xrd_z)
    
    return xrd_mat4


# In[9]:


test_xrd3=np.nan_to_num(get_xrdmat(test_xrd2))


# In[10]:


print(test_xrd3)


# In[11]:


test_xrd4=preprocessing.normalize(np.array([test_xrd3]), norm="l2", axis=1)


# In[12]:


test_xrd5=np.dot(test_xrd4.T,test_xrd4)


# In[13]:


test_xrd5=test_xrd5.astype('complex128')


# In[14]:


ITR=50
LR=0.1


# In[15]:


with fluid.dygraph.guard():
    net1= paddle_qcnn_5bits_5to4.NET_2([21])
    '''
    x=fluid.dygraph.to_variable(test_xrd5)
    print(type(x))
    print(x.shape)
    e1,e1_dagger=net1()
    print(type(e1))
    print(e1.shape)
    xxx=matmul(matmul(e1,x),e1_dagger)
    '''
    
  
    opt = fluid.optimizer.Adam(learning_rate=LR, 
                          parameter_list=net1.parameters())

    for itr in range(1, ITR *10 + 1):
        
        out_net1, loss_net1 = net1(test_xrd5)
        #print(type(out_net1))
        #print(type(test_xrd5))
        loss_net1.backward()
        opt.minimize(loss_net1)
        net1.clear_gradients()
        
        fid = state_fidelity(test_xrd5, out_net1.numpy())

        if itr % 10 == 0:
            print('iter:', itr, 'loss:', '%.4f' % loss_net1, 'fid:', '%.4f' % fid)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




