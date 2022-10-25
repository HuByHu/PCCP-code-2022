#!/usr/bin/env python
# coding: utf-8

# In[5]:

import os
import numpy as np
import pymatgen as mg
import pymatgen.analysis.diffraction as anadi
import pymatgen.analysis.diffraction.xrd as xrd


# #Give_the_filepath_of_datasets
# =======

# In[ ]:


#person_path='/home/ipqi/.jupyter/20201020/'

#train_path=person_path+'GcN-sur/train/'

#test_path=person_path+'GcN-sur/test/'


# In[6]:


patt_xrd = xrd.XRDCalculator('CuKa')

global sample_num, rmat_num, series_num
sample_num=1 #output of G
rmat_num=32 #row nums of the matrix for the input of CNN

#test_xrd=train_path+'00000'

def tomgStructure(folder):
    POSfile=folder+'/CONTCAR'      
    R_mgS=mg.core.Structure.from_file(POSfile)
    return R_mgS

#test_xrd2=tomgStructure(test_xrd)

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

#test_xrd3=np.nan_to_num(get_xrdmat(test_xrd2))
#print(test_xrd3)

#test_xrd4=preprocessing.normalize(np.array([test_xrd3]), norm="l2", axis=1)

#test_xrd5=np.dot(test_xrd4.T,test_xrd4)

#test_xrd5=test_xrd5.astype('complex128')

#for GcN-sur
extend_num=1000
move_num=-122.69044

# In[ ]:
def get_energy(folder):
    energy_string=os.popen('grep TOTEN '+folder+'/OUTCAR | tail -1').read().split(' ')[-2]
    energy=round(np.float64(float(energy_string)),5)
    return energy

def get_cpx222_surface_energy(bluk,surf):
    global N
    sur_e= (bluk-surf)/(2*N)
    return sur_e

def linear_transform(energy):
    global extend_num, move_num
    energy_transform=(energy-move_num)*extend_num
    return energy_transform
def inverse_transform(energy_transform):
    global extend_num, move_num
    energy=energy_transform/extend_num+move_num
    return energy
def get_energy_per_atom(energy):
    energy_per_atom=energy/atoms_num
    return energy_per_atom





