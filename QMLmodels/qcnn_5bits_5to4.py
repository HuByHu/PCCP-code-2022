#!/usr/bin/env python
# coding: utf-8

# In[1]:


import paddle
from paddle import fluid
from paddle_quantum.ansatz import Circuit
from paddle_quantum.qinfo import state_fidelity, partial_trace, pauli_str_to_matrix, NKron
from paddle_quantum.linalg import dagger, haar_unitary
from paddle import complex
from paddle_quantum.linalg import NKron as kron
from paddle import matmul, trace
import numpy as np
import torch

# In[ ]:
torch.set_default_dtype(torch.float32)
paddle.set_default_dtype("float32")

def Rx_U(theta,i):
    cir=Circuit(1)
    cir.rx(0,theta[i])
    return cir.unitary_matrix()
def Ry_U(theta,i):
    cir=Circuit(1)
    cir.ry(0,theta[i])
    return cir.unitary_matrix()
def Rz_U(theta,i):
    cir=Circuit(1)
    cir.rz(0,theta[i])
    return cir.unitary_matrix()
def I_U():
    ii=paddle.to_tensor([[1.0,0.0],[0.0,1.0]])
    return ii


# In[ ]:


def multi_kron(A,B,C,D,E):
    xx=kron(kron(kron(kron(A,B),C),D),E)
    #xx=kron(A,B,C,D,E)
    return xx

# In[2]:

#Conv_partA
def layer_1(theta):
    cir=Circuit(5)
    cir.rx(0,theta[0])
    cir.ry(0,theta[1])
    cir.rz(0,theta[2])
    cir.rx(1,theta[3])
    cir.ry(1,theta[4])
    cir.rz(1,theta[5])
    cir.rx(2,theta[0])
    cir.ry(2,theta[1])
    cir.rz(2,theta[2])
    cir.rx(3,theta[3])
    cir.ry(3,theta[4])
    cir.rz(3,theta[5])
    cir.rx(4,theta[0])
    cir.ry(4,theta[1])
    cir.rz(4,theta[2])
    #cir.rx(theta[3],5)
    #cir.ry(theta[4],5)
    #cir.rz(theta[5],5)

    return cir.unitary_matrix()

#conv_partC
def layer_3(theta):
    cir=Circuit(5)
    cir.rx(0,theta[9])
    cir.ry(0,theta[10])
    cir.rz(0,theta[11])
    cir.rx(1,theta[12])
    cir.ry(1,theta[13])
    cir.rz(1,theta[14])
    cir.rx(2,theta[9])
    cir.ry(2,theta[10])
    cir.rz(2,theta[11])
    cir.rx(3,theta[12])
    cir.ry(3,theta[13])
    cir.rz(3,theta[14])
    cir.rx(4,theta[9])
    cir.ry(4,theta[10])
    cir.rz(4,theta[11])
    #cir.rx(theta[12],5)
    #cir.ry(theta[13],5)
    #cir.rz(theta[14],5)
    return cir.unitary_matrix()


# In[3]:


#pool_partA & pool_partC
def pool_pre_layer(theta, control_bit,target_bit):
    cir=Circuit(5)
    cir.rx(target_bit,theta[15])
    cir.ry(target_bit,theta[16])
    cir.rz(target_bit,theta[17])
    cir.rx(control_bit,theta[18])
    cir.ry(control_bit,theta[19])
    cir.rz(control_bit,theta[20])
    return cir.unitary_matrix()
def pool_append_layer(theta,target_bit):
    cir=Circuit(5)
    cir.rx(target_bit,-theta[15])
    cir.ry(target_bit,-theta[16])
    cir.rz(target_bit,-theta[17])
    return cir.unitary_matrix()

bits_number=5

def multi_cnot(bits_num,control_bit,target_bit):
    n0=bits_num
    c_bit=str(control_bit)
    t_bit=str(target_bit)
    str1='i'+c_bit
    str2='z'+c_bit
    str3='x'+t_bit
    str4='z'+c_bit+',x'+t_bit
    out_cnot=pauli_str_to_matrix([[0.5,str1],[0.5, str2],[0.5,str3],[-0.5,str4]],n=n0)
    #out_cnot=fluid.dygraph.to_variable(out_cnot.astype('complex128'))
    return out_cnot


# In[6]:


def Conv_5bits_Net(theta):
    #get U of qcnn circuit
    #theta=fluid.dygraph.to_variable(theta)
    O1=layer_1(theta)
    
    Ax=Rx_U(theta,6)
    By=Ry_U(theta,7)
    Cz=Rz_U(theta,8)
    #D=fluid.dygraph.to_variable(I_U()+0j)
    D=I_U()+0j
    
    #conv_partB
    # conv_1
    convXX1=multi_kron(Ax,Ax,D,D,D)
    convYY1=multi_kron(By,By,D,D,D)
    convZZ1=multi_kron(Cz,Cz,D,D,D)
    convXX2=multi_kron(D,D,Ax,Ax,D)
    convYY2=multi_kron(D,D,By,By,D)
    convZZ2=multi_kron(D,D,Cz,Cz,D)
    convXX3=multi_kron(Ax,D,D,D,Ax)
    convYY3=multi_kron(By,D,D,D,By)
    convZZ3=multi_kron(Cz,D,D,D,Cz)
    convXX4=multi_kron(D,Ax,Ax,D,D)
    convYY4=multi_kron(D,By,By,D,D)
    convZZ4=multi_kron(D,Cz,Cz,D,D)
    convXX5=multi_kron(D,D,D,Ax,Ax)
    convYY5=multi_kron(D,D,D,By,By)
    convZZ5=multi_kron(D,D,D,Cz,Cz)

    O1_2=layer_3(theta)
    
    #conv_2
    '''
    convXX1_2=multi_kron(D,Ax,Ax,D,D,D)
    convYY1_2=multi_kron(D,By,By,D,D,D)
    convZZ1_2=multi_kron(D,Cz,Cz,D,D,D)
    convXX2_2=multi_kron(D,D,D,Ax,Ax,D)
    convYY2_2=multi_kron(D,D,D,By,By,D)
    convZZ2_2=multi_kron(D,D,D,Cz,Cz,D)
    convXX3_2=multi_kron(Ax,D,D,D,D,Ax)
    convYY3_2=multi_kron(By,D,D,D,D,By)
    convZZ3_2=multi_kron(Cz,D,D,D,D,Cz) 
    '''
    
    O2=convXX1
    O3=convYY1
    O4=convZZ1
    
    O5=convXX2
    O6=convYY2
    O7=convZZ2
    
    O8=convXX3
    O9=convYY3
    O10=convZZ3
    
    O11=convXX4
    O12=convYY4
    O13=convZZ4
    
    O14=convXX5
    O15=convYY5
    O16=convZZ5
    
    #O17=convXX3_2
    #O18=convYY3_2
    #O19=convZZ3_2
    
    out_conv1 = matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(matmul(O1,O2),O3),O4),O5),O6),O7),O8),O9),O10),O1_2),O1),O11),O12),O13),O14),O15),O16),O1_2)

    #pool1
    O20=pool_pre_layer(theta,0,2)
    #print(type(O20))
    #print(O20.shape)
    O21=multi_cnot(bits_number,0,2)
    #print(type(O21))
    #print(O21.shape)
    O22=pool_append_layer(theta,2)
    #print(type(O22))
    #print(O22.shape)
    out_pool1=matmul(matmul(O20,O21),O22)
    
    #pool2
    O23=pool_pre_layer(theta,0,3)
    O24=multi_cnot(bits_number,0,3)
    O25=pool_append_layer(theta,3)
    out_pool2=matmul(matmul(O23,O24),O25)
    
    #pool3
    O26=pool_pre_layer(theta,0,4)
    O27=multi_cnot(bits_number,0,4)
    O28=pool_append_layer(theta,4)
    out_pool3=matmul(matmul(O26,O27),O28)
    
    #pool4
    O29=pool_pre_layer(theta,0,1)
    O30=multi_cnot(bits_number,0,1)
    O31=pool_append_layer(theta,1)
    out_pool4=matmul(matmul(O29,O30),O31)

    '''
    #pool5
    O32=pool_pre_layer(theta,1,3)
    O33=multi_cnot(bits_number,1,3)
    O34=pool_append_layer(theta,3)
    out_pool5=matmul(matmul(O32,O33),O34)

    #pool6
    O35=pool_pre_layer(theta,1,4)
    O36=multi_cnot(bits_number,1,4)
    O37=pool_append_layer(theta,4)
    out_pool6=matmul(matmul(O35,O36),O37)
    '''

    out_conv_net = matmul(matmul(matmul(matmul(out_conv1,out_pool1),out_pool2),out_pool3),out_pool4)#,out_pool5),out_pool6)#,out_conv2),out_pool5)
    
    '''
    #pool4
    O29=pool_layer_7(theta)
    O30=multi_cnot_4()
    O31=pool_layer_8(theta)
    out_pool4=matmul(matmul(O29,O30),O31)
    
    #conv3
    O32=layer_after_pool_1(theta)
    O32_2=layer_after_pool_2(theta)
    
    Ax2=Rx_U(theta,27)
    By2=Ry_U(theta,28)
    Cz2=Rz_U(theta,29)
    convXX3_3=multi_kron(D,D,D,D,Ax2,Ax2)
    convYY3_3=multi_kron(D,D,D,D,By2,By2)
    convZZ3_3=multi_kron(D,D,D,D,Cz2,Cz2) 
    O33=convXX3_3
    O34=convYY3_3
    O35=convZZ3_3
    
    out_conv2=matmul(matmul(matmul(matmul(O32,O33),O34),O35),O32_2)
    
    #pool5
    O36=pool_layer_9(theta)
    O37=multi_cnot_5()
    O38=pool_layer_10(theta)
    out_pool5=matmul(matmul(O36,O37),O38)
    '''    
    return out_conv_net 


# In[ ]:

def readout_Net1():
    out=pauli_str_to_matrix([[1,'z0']],n=4)
    #out=fluid.dygraph.to_variable(out.astype('complex128'))
    return out

def readout_Net2():
    out=pauli_str_to_matrix([[1,'z1']],n=4)
    #out=fluid.dygraph.to_variable(out.astype('complex128'))
    return out

def readout_Net3():
    out=pauli_str_to_matrix([[1,'z2']],n=4)
    #out=fluid.dygraph.to_variable(out.astype('complex128'))
    return out

def readout_Net4():
    out=pauli_str_to_matrix([[1,'z3']],n=4)
    #out=fluid.dygraph.to_variable(out.astype('complex128'))
    return out


