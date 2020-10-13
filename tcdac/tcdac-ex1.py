#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 00:16:05 2019
vs. /Users/jiedeng/Box Sync/xpaper5/T_correct_Badro/benchmark_tcdac.m
@author: jiedeng
"""

import numpy as np
from tcdac_v2 import tcdac, planck
import matplotlib.pyplot as plt

### D is um
#sg = 0.8; pvs = 0.84; sm = 0.56; mm = 11.12; ms = 0.68; spv = 0.52; gs = 1.12;

sg =  0.02; pvs =  0.02; sm = 0.02; mm = 1; ms = 0.02; spv = 0.02; gs = 0.02;
#sg =  1; pvs =  1; sm = 1; mm = 1; ms = 1; spv = 1; gs = 1;

D  = [sg,pvs,sm,mm,ms,spv,gs]
Ti = 4000
T0 = 300
### lamda is m
lamda = np.linspace(518, 683, 100)*1e-9
inp   = tcdac(D, Ti,lamda,T0=T0)
inp.l7_grad()
inp.plotT()
### ab is in lamda order
ab1 = np.linspace(2970, 2970, 100)*1e2
ab  = np.zeros((len(lamda),7))

#ab[:,0] = ab1
#ab[:,1] = ab1
#ab[:,2] = ab1
#ab[:,3] = np.ones(len(lamda))*1e8
#ab[:,4] = np.ones(len(lamda))*1e8
#ab[:,5] = np.ones(len(lamda))*1e8
#ab[:,6] = np.ones(len(lamda))*1e8
ab[:,0] = ab1
ab[:,1] = ab1
ab[:,2] = ab1
ab[:,3] = np.ones(len(lamda))*1e8
ab[:,4] = ab1
ab[:,5] = ab1
ab[:,6] = ab1

#ab[:,0] = np.ones(len(lamda))*1e8
#ab[:,1] = np.ones(len(lamda))*1e8
#ab[:,2] = np.ones(len(lamda))*1e8
#ab[:,3] = np.ones(len(lamda))*1e8
#ab[:,4] = ab1
#ab[:,5] = ab1
#ab[:,6] = ab1

inp.l7_ab(ab)
inp.plotab()
inp.cal_I()
plt.figure()
plt.title('positive side, should be hot as 4000 K')
plt.plot(inp.lamda*1e9,inp.I,label='cal')
plt.plot(inp.lamda*1e9,planck(lamda, Ti))
plt.legend()
inp.fit_planck()

inp.cal_I(side='n')

plt.figure()
plt.title('negative side, should be dim as 300 K')
plt.plot(inp.lamda*1e9,inp.I,label='cal')
plt.plot(inp.lamda*1e9,planck(lamda, Ti))
plt.legend()
inp.fit_planck()