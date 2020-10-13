#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:33:23 2019

@author: jiedeng
"""

import numpy as np

#########FITTING#########
def gds_single_sided(z,w,rhol,rhov,z0=0):
    """
    Gibbs dividing surface
    ---------
    input:
        rhol: liquid
        rhov: vapor
        w: width of the dividing surface
    output:
        rho:    
    """
    rho = .5*(rhol + rhov) + .5*(rhol-rhov)*np.tanh((z-z0)/w)
    return rho

def gds_double_sided(z,w,rhol,rhov,z0=0,z1=5):
    """
    Gibbs dividing surface
    ---------
    input:
        rhol: liquid
        rhov: vapor
        w: width of the dividing surface
    output:
        rho:    
    """
#    rho = .5*(rhol + rhov) + .5*(rhol-rhov)*(np.tanh((z-z0)/w) - np.tanh((z-z1)/w))
    rho = rhov + .5*(rhol-rhov)*(np.tanh((z-z0)/w) - np.tanh((z-z1)/w))

    return rho

def gds_double_sided_skewed(z,w,rhol,rhov,z0=0,z1=5,k=1):
    """
    Gibbs dividing surface
    ---------
    input:
        rhol: liquid
        rhov: vapor
        w: width of the dividing surface
        k : float, skewness factor
    output:
        rho:    
    """
#    rho = .5*(rhol + rhov) + .5*(rhol-rhov)*(np.tanh((z-z0)/w) - np.tanh((z-z1)/w))
    rho = rhov + .5*(rhol-rhov)*(np.tanh(k*(z-z0)/w) - np.tanh(k*(z-z1)/w))
    return rho

def fit_gds_single_sided(zi,rho_zi,plot=True,verbose=True):
    """
    LS fitting to GDS
    """
    from lmfit import Parameters, Model, report_fit
    zi_filter  = zi[rho_zi>0]
    rho_filter = rho_zi[rho_zi>0]
    params = Parameters()
    params.add('z0', value=45,vary=True)
    params.add('w' , value=1.5,vary=True)    
    params.add('rhov', value=0.2,min = 0.1,max=0.5,vary=False)
    params.add('rhol', value=2.3,min = 0,max=2.5,vary=True) 
    model = Model(gds_single_sided)
    half = int(len(rho_filter)/2)
    result = model.fit(rho_filter[:half],params,z = zi_filter[:half])
    if verbose:
        report_fit(result)
        result.params.pretty_print
    if plot:
        result.plot_fit()
    return result

def fit_gds_double_sided(zi,rho_zi,weights=None,
                         z0min = 0, z1min = 0,
                         z0max = 400, z1max = 400,
                         rhovmin = 0, rholmin = 0,
                         rhovmax = 10, rholmax = 10,
                         wmin = 0, wmax = 100,
                         plot=True, verbose=True):
    """
    LS fitting to GDS
    """
    from lmfit import Parameters, Model, report_fit
#    zi_filter  = zi[rho_zi>0]
#    rho_filter = rho_zi[rho_zi>0]
    zi_filter  = zi
    rho_filter = rho_zi
    params = Parameters()
    params.add('z0', value=30,   min = z0min,    max = z0max,   vary=True)
    params.add('z1', value=45,   min = z1min,    max = z1max,   vary=True)
    params.add('w' , value=5,    min = wmin,     max =  wmax,   vary=True)    
    params.add('rhov', value=0.2, min = rhovmin, max = rhovmax, vary=True)
    params.add('rhol', value=2.4, min = rholmin, max = rholmax, vary=True) 
    model = Model(gds_double_sided)
    result = model.fit(rho_filter,params,z = zi_filter,weights=weights)
    if verbose:
        report_fit(result)
        result.params.pretty_print
    if plot:
        result.plot_fit()
    return result

def fit_gds_double_sided_skewed(zi,rho_zi,zmax = 400,plot=True,verbose=True):
    """
    LS fitting to GDS
    """
    from lmfit import Parameters, Model, report_fit
#    zi_filter  = zi[rho_zi>0]
#    rho_filter = rho_zi[rho_zi>0]
    zi_filter  = zi
    rho_filter = rho_zi
    params = Parameters()
    params.add('z0', value=30,min = 0, max = zmax,vary=True)
    params.add('z1', value=45,min = 0, max = zmax, vary=True)
    params.add('k', value=1,min = 0, max = 1, vary=True)
    params.add('w' , value=5,vary=True)    
    params.add('rhov', value=0.2,min = 0, vary=True)
    params.add('rhol', value=2.4,min = 0,max=10,vary=True) 
    model = Model(gds_double_sided)
    result = model.fit(rho_filter,params,z = zi_filter)
    if verbose:
        report_fit(result)
        result.params.pretty_print
    if plot:
        result.plot_fit()
    return result
