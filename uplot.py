#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:31:24 2019

@author: jiedeng
"""
from uncertainties import unumpy
import matplotlib.pyplot as plt

def uplot(x,y,*args,**kwargs):
    """
    plot output of uncertainties package
    x -> numpy array
    y -> unumpy array
    """
    nominal_curve = plt.plot(x, unumpy.nominal_values(y), *args, **kwargs)
    plt.fill_between(x, 
                        unumpy.nominal_values(y)-unumpy.std_devs(y), 
                        unumpy.nominal_values(y)+unumpy.std_devs(y),
                        facecolor=nominal_curve[0].get_color(),
                        edgecolor='face',
                        alpha=0.3,
                        linewidth=0)
    
    return nominal_curve