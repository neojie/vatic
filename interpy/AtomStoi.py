#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:51:29 2019
ref: https://kitchingroup.cheme.cmu.edu/blog/2014/09/23/Generating-an-atomic-stoichiometric-matrix/
=> deprecated 
=> one can easily use the periodictable to all the jobs
example

import periodictable as pdt
pdt.formula('Mg32Si32O96').atoms
@author: jiedeng
"""

import re

m = re.findall('([A-Z][a-z]?)(\d?)' , 'ArC2H6Cu56Pd47Co')
print(m)

# save for future use
cf = re.compile('([A-Z][a-z]?)(\d?)')

species = ['H2O', 'CO2', 'H2', 'CO2']

all_elements = []

for s in species:
    for el, count in re.findall(cf, s):
        all_elements += [el]

atoms = set(all_elements)

# we put a placeholder in the first row
counts = [[""] + species]
for e in atoms:
    # store the element in the first column
    count = [e]
    for s in species:    
        d = dict(re.findall(cf, s))
        n = d.get(e, 0)
        if n == '': n = 1
        count += [int(n)]
    counts += [count]

# this directly returns the array to org-mode

import Periodic
