#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 09:55:22 2018

@author: jiedeng
Module requirement: 

File requirment: XDATCAR

"""

from JDlib import trajxdt as traj

# Initalize
xdat = traj.trajxdt(stepsize=25)

xdat.Tra2D(['O'])
xdat.Tra3D(['O'])
xdat.Tra2D(atom_index=16)
#car = xdat.car

#df = xt.xdatcar('XDATCAR')
#df.readxdat()


#print(np.shape(df))
#vel = df.velocity
#dos = df.phononDos()
#plt.figure()
#plt.plot(vel[:,1,1])
#plt.show()
#
#plt.figure()
#plt.plot(df.VAF2)
#plt.xlim([0,500])
#plt.show()
#
#plt.figure()
#plt.plot(df.Temp)
#plt.show()
