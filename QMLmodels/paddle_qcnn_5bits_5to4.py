#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qcnn_5bits_5to4


# In[2]:


import numpy as np


# In[3]:


#theta=np.ones(60)


# In[4]:


#import paddle
#from paddle import fluid
#from paddle_quantum.circuit import UAnsatz
#from paddle_quantum.utils import partial_trace, dagger, state_fidelity, NKron,pauli_str_to_matrix
#from paddle import complex
#from paddle.complex import matmul, trace, kron,elementwise_add

import paddle
from paddle import fluid
from paddle_quantum.ansatz import Circuit
from paddle_quantum.qinfo import state_fidelity, partial_trace, pauli_str_to_matrix, NKron
from paddle_quantum.linalg import dagger, haar_unitary
from paddle import complex
from paddle import matmul, trace, kron
import torch

torch.set_default_dtype(torch.float32)
paddle.set_default_dtype("float32")
# In[ ]:
rho_decode_C=paddle.to_tensor([1.0,0.0])
rho_decode_C=paddle.diag(rho_decode_C)
rho_decode_C=rho_decode_C+0j

N_B = 1
N_A = 4


# In[5]:


LR = 0.2       # 设置学习速率
ITR = 100      # 设置迭代次数
SEED = 14      # 固定初始化参数用的随机数种子
class NET_1(fluid.dygraph.Layer):
    """
    Construct the model net
    """
    def __init__(self, shape, param_attr=fluid.initializer.Uniform(
        low=0.0, high=2 * np.pi, seed = SEED), dtype='float64'):
        super(NET_1, self).__init__()
        
        self.theta = self.create_parameter(shape=shape, 
                     attr=param_attr, dtype=dtype, is_bias=False)        
    
    # 定义损失函数和前向传播机制
    def forward(self,x):
        # 生成初始的编码器 E 和解码器 D\n",
        rho_in= fluid.dygraph.to_variable(x) 
        #obz=fluid.dygraph.to_variable(ObZ())
        E = qcnn_5bits_5to4.Conv_5bits_Net(self.theta)
        E_dagger = dagger(E)        
        
        rho_conv = matmul(matmul(E, rho_in), E_dagger)          
        out=np.trace(matmul(obz,rho_conv).real.numpy())        

        return out


# In[13]:


LR = 0.2       # 设置学习速率
ITR = 100      # 设置迭代次数
SEED = 14      # 固定初始化参数用的随机数种子

class load_NET_2(paddle.nn.Layer):
    """
    Construct the model net
    """
    def __init__(self, theta):
        super(load_NET_2, self).__init__()
        
        self.rho_decode_C = rho_decode_C
        
        #self.theta = fluid.dygraph.to_variable(theta)   
        self.theta = theta
    
    # 定义损失函数和前向传播机制
    def forward(self,x):
        x = x.astype('float32')
        # 生成初始的编码器 E 和解码器 D\n",
        rho_in= paddle.Tensor(x)+0j
        #obz=fluid.dygraph.to_variable(ObZ())
        E = qcnn_5bits_5to4.Conv_5bits_Net(self.theta)
        E_dagger = dagger(E)
        D = E_dagger
        D_dagger = E
        
        rho_conv = matmul(matmul(E, rho_in), E_dagger)
        
        rho_encode = partial_trace(rho_conv, 2 ** N_B, 2 ** N_A, 1)
        rho_trash = partial_trace(rho_conv, 2 ** N_B, 2 ** N_A, 2)
        
        rho_CA = kron(self.rho_decode_C, rho_encode)
        rho_out = matmul(matmul(D, rho_CA), D_dagger)
        
        zero_Hamiltonian1=self.rho_decode_C
        
        #out=np.trace(matmul(obz,rho_conv).real.numpy()) 
        #loss = 1.0 - (trace(matmul(zero_Hamiltonian1, rho_trash))).real

        #return rho_out,loss,rho_encode
        return rho_out, rho_encode


# In[14]:

def measure_Z(rho_encode):
    c=rho_encode
    mat1=qcnn_5bits_5to4.readout_Net1()
    mat2=qcnn_5bits_5to4.readout_Net2()
    mat3=qcnn_5bits_5to4.readout_Net3()
    mat4=qcnn_5bits_5to4.readout_Net4()
    measure1=float(np.trace(matmul(mat1,c).numpy()).real)
    measure2=float(np.trace(matmul(mat2,c).numpy()).real)
    measure3=float(np.trace(matmul(mat3,c).numpy()).real)
    measure4=float(np.trace(matmul(mat4,c).numpy()).real)
    input_net3=[]
    input_net3.append(measure1)
    input_net3.append(measure2)
    input_net3.append(measure3)
    input_net3.append(measure4)
    input_net3=np.asarray(input_net3)
    return input_net3


