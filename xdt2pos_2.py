#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:02:21 2019
version 2 readlines behave very funny, cannot understand...
@author: jiedeng
"""

import numpy as np
import os
from shutil import copyfile


def xdt2pos(begin = 0,end=1000):
    begin_lines = (total_atom+1)*(begin-1)
    
    if begin_lines != 0:
        for i in range(begin_lines):
            XDATCAR.readline()
        
    for i in range(begin, end):
        try:
            os.mkdir(str(i))
        except:
            print("Folder {0} already exists,skip".format(i))
        coord=XDATCAR.readline()
        atomic_position = []
        for j in range(total_atom):       
            line_tmp = XDATCAR.readline()
            atomic_position.append(line_tmp)
        # move first lien into the end
        atomic_position[0], atomic_position[-1] = atomic_position[-1], atomic_position[0]
        
        tobewriten=[]
        tobewriten.append(title+'\n')
        tobewriten.append('{0}\n'.format(scaling_factor))
        for j in range(3):
            tobewriten.append('    {0:14.9f}{1:14.9f}{2:14.9f}\n' .format(lattice[0][j],lattice[1][j],lattice[2][j]))
        for j in range(len(element_list)):
            tobewriten.append('    %4s%s'%(element_list[j],' '))
        tobewriten.append('\n')
        for j in range(len(element_amount)):
            tobewriten.append('    %4d%s'%(element_amount[j],' '))   
        tobewriten.append('\n')
        tobewriten.append('Direct\n')
        for j in range(len(total_elements)):
            tobewriten.append(atomic_position[j])
        
        #begin to write POSCAR content to tmp file
        fw = open('tmp','w')
        fw.writelines(tobewriten)
        fw.close()
        os.rename('tmp',str(i)+'/POSCAR')
        copyfile('INCAR',str(i))
        copyfile('KPOINTS',str(i))
        copyfile('POTCAR',str(i))
        copyfile('bs',str(i))

XDATCAR=open("XDATCAR",'r')

#title=XDATCAR.readline().strip()
title=XDATCAR.readline().rstrip('\r\n').rstrip('\n')
scaling_factor=float(XDATCAR.readline())
lattice = np.zeros((3,3))

for i in range(3):
    lattice[i]=np.array([float(j) for j in XDATCAR.readline().split()])
element_list=[j for j in XDATCAR.readline().split()]

#### add Hf in the end
element_list.append('Hf')

element_amount=[int(j) for j in XDATCAR.readline().split()]
#### add Hf in the end
element_amount[0] = element_amount[0] -1
element_amount.append(1)

total_elements=[]
for i in range(len(element_amount)):
    total_elements.extend([element_list[i]]*element_amount[i])
total_atom=sum(element_amount)

xdt2pos(7,10)

