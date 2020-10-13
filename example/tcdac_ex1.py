#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:59:55 2019

@author: jiedeng
"""

sg = 0.8; pvs = 0.84; sm = 0.56; mm = 11.12; ms = 0.68; spv = 0.52; gs = 1.12;
D = [sg,pvs,sm,mm,ms,spv,gs]
Ti = 4000
T0 = 300

lamda = np.linspace(518, 683, 100)*1e-9