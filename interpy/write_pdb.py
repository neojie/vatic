#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:51:19 2019

@author: jiedeng
"""
from copy import deepcopy
import os
from vatic import XDATCAR_toolkit as xdt


def write_pdb(begin = 0, end = None,file='POSCAR', timestep=1, 
              center=True, center0 = None, masses=None):
    """
    write PDB file based on XDATCAR at given frame range
    
    Parameters
    ----------
    begin : int
            start frame
    end :   int or False
            end frame,  
            if None means one frame, 
            if False means end (to do, do not know how to get max frame #)
            only way is to do len(u.trajectory) where u is MDAnalysis universe
    file : str
            input file
    timestep : float

    Returns
    -------
    None    
    
    """
    XDATCAR_inst = xdt.XDATCAR(file=file)
    XDATCAR_iter = iter(XDATCAR_inst)
    XDATCAR_inst.format_trans = True
    
    if end is None:
        name = 'XDATCAR'+'_'+str(begin)
    elif end is False:
        name = 'XDATCAR'+'_'+str(begin)+'_'+'end'
    else:
        name = 'XDATCAR'+'_'+str(begin)+'_'+str(end)
        
    target = name+".pdb"
    
    if masses is not None:
        masses_sum = sum(masses)
        
    sep = '+++++++'
    print(sep*10)
    if os.path.exists(target):
        print('{0} already exists. Skip re-making this file'.format(target))
        print(sep*10)
#        os.remove(target)
    else:
        print("No {0} exists. Now begin to make this pdb file".format(target))
        print(sep*10)
        XDATCAR_inst.timestep = timestep   #timestep 1fs    
        if end is None:
            XDATCAR_inst('t>= %r and t <= %r' %(int(begin),int(begin)))
        else:
            XDATCAR_inst('t>= %r and t <= %r' %(int(begin),int(end))) # frame 10~300  corresponding to 20~600fs
        _interval=1
        count=0 
        current_pdb=1  
        for i in range(XDATCAR_inst.uprange+1): 
            if (i>=XDATCAR_inst.lowrange):
                print("---------",i,"---------")
                cartesian_position=XDATCAR_iter.next()
                if count % _interval == 0:
                    if i == XDATCAR_inst.lowrange:
                        real_atomic_cartesian=deepcopy(cartesian_position)
                        if center:
                            real_atomic_cartesian = centering(real_atomic_cartesian,center0,
                                                              masses,masses_sum,XDATCAR_inst.lattice)                            
                        XDATCAR_inst.cartesian_position = real_atomic_cartesian
                        prev_atomic_cartesian=deepcopy(cartesian_position)
                    else:
                        if center:
                            real_atomic_cartesian=deepcopy(cartesian_position)
                            real_atomic_cartesian = centering(real_atomic_cartesian,center0,
                                                              masses,masses_sum,XDATCAR_inst.lattice) 
                        else:
#                            print("else")
                            prev_atomic_cartesian,diffs = XDATCAR_inst.unswrapPBC(prev_atomic_cartesian)
                            real_atomic_cartesian+=diffs                       
                        XDATCAR_inst.cartesian_position = real_atomic_cartesian 
                    # print('position is,',XDATCAR_inst.cartesian_position)
                    # print('pos diff is,', XDATCAR_inst.cartesian_position- cartesian_position)
                    XDATCAR_inst.writepdb(current_pdb,target)
                    current_pdb+=1
                count+=1
            else:
                XDATCAR_iter.skiplines_()
    
        print('Finish reading XDATCAR.')        
        print('Selected time-range:{0}~{1}fs'.format((XDATCAR_inst.lowrange)*timestep,\
                            (XDATCAR_inst.uprange)*timestep))
        XDATCAR_inst.XDATCAR.close()
        print('Timestep for new PDB trajectory is :{0}fs'.format(timestep*_interval))  

# import numpy as np  
def centering(real_atomic_cartesian,center0,masses,masses_sum,lattice):
    center_new = masses.dot(real_atomic_cartesian)/masses_sum
#    print('center_new',center_new)
#    print('cneter0',center0)
    real_atomic_cartesian_1 = real_atomic_cartesian - center_new + center0
    for i in range(3):
        idx = (real_atomic_cartesian_1[:,i] < 0)
        real_atomic_cartesian_1[idx,i] = real_atomic_cartesian_1[idx,i] + lattice[i][i]
        idx = (real_atomic_cartesian_1[:,i] > lattice[i][i])
        real_atomic_cartesian_1[idx,i] = real_atomic_cartesian_1[idx,i] - lattice[i][i]
    return real_atomic_cartesian_1