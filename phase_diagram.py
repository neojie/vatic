#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:20:46 2018

@author: jiedeng
"""

from scipy.interpolate import interp1d
import numpy as np
import scipy.constants as con

def linfit(x,k,y0):
    """
    The polyfit does not give std 
    """
    y = x*k+y0
    return y

def lf12(P,A,B):
    """
    equation 9 of Liebske and Frost 2012 EPSL
    A kJ/mol
    B kJ*GPa/mol
    
    """
    return A+B/P

def lf12_aug(P,A,B,C):
    """
    equation 9 of Liebske and Frost 2012 EPSL
    A kJ/mol
    B kJ*GPa/mol
    
    """
    return A+B*P+C*(P**(.5))