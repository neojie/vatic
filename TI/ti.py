#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:47:05 2019

@author: jiedeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post-analysis of thermodynamic integration method


@author: jiedeng
"""

import numpy as np
import matplotlib.pyplot as plt
import uncertainties as uct
from vatic import blockAverage as ba


### mean square fluctuation
kb = 8.617e-5

def msf(du):
    """
    cal mean square fluctuation
    """
    return ((du - du.mean())**2).mean()

def err(du):
    """
    build ufloat for du
    uct.ufloat(du.mean(),ba.blockAverage(du,isplot=False)[0])
    
    Notes
    -----
    time average = simply average
    and also should == ensemelbe at the ergodicity 
    However, because the phase space is very large, our limited simulatin 
    does not sample all of them, we are supposed to use the time average 
    instead of the ensemble average here
    """
    return uct.ufloat(du.mean(),ba.blockAverage(du,isplot=False)[0])

def cubic(du_free,du_dope,T,plot=False,label='test'):
    """
    ax^3 + bx^2 + cx^ + d = f(x);  x is lambda
    
    Notes
    ----
    we have f(0) = d, f'(0) = c, f(1) = a+b+c+d; f'(1) = 3a+2b+c
    
    """
    f0 = du_free.mean(); f0p = -msf(du_free)/kb/T
    f1 = du_dope.mean(); f1p = -msf(du_dope)/kb/T
    c  = f0p; d = f0
    a  = f1p + f0p - 2*f1 + 2*f0
    b  = 3*f1 - 3*f0 - f1p - 2*f0p
    df_linear = (err(du_free)+err(du_dope))/2
    df_cubic  = a/4 + b/3 + c/2 + d
    
    if plot:
        l  = np.linspace(0,1)
        fl = a*(l**3) + b*(l**2) + c*l + d
        plt.plot(l,fl,label=label)
        plt.plot([0,1],[f0,f1],'k--o')
        plt.xlabel(r'$\lambda$',fontsize=14)
        plt.ylabel('<U(dope) - U(free)>',fontsize=14)
    return df_linear.nominal_value, df_cubic, df_linear.std_dev, f0p,f1p



def analyze(r1,r1_traj,r1_dope,r1_dope_traj,r3,r3_traj,r3_dope,r3_dope_traj,T,dPV=0,plot=True):
    """
    analyze the TI results
    
    Params
    ------
    x & x_traj : arrays 1d, internal energy
                 x_traj and x must have the same dimension, 
                 but r1 and r3 do not have 
    T : float, temperature
    dPV : PV of l - PV of s, default is 0
    plot : plot the <dU> vs. lambda, default is true
    
    Returns
    ------
    out : panadas DataFrame

    """
    import pandas as pd
    du_s_free = r1_traj - r1
    du_s_dope = r1_dope - r1_dope_traj
    
    du_l_free = r3_traj - r3
    du_l_dope = r3_dope - r3_dope_traj

    columns = ['r1','r1-traj','r1-dope','r1-dope-traj',
               'r3','r3-traj','r3-dope','r3-dope-traj',
               'dr1','dr1-dope','dr3','dr3-dope',
               'dF_s','dF_s_cub','dF_l','dF_l_cub','s0p','s1p','l0p','l1p',
               'ddF','ddF_cub','dPV','dG', 'dG_cub','uct_dG','D','D_cub','uct_D',
               'log10_D','log10_D_cub','log10_uct_D']
    
    dr1      = du_s_free.mean()
    dr1_dope = du_s_dope.mean()
    dr3      = du_l_free.mean()
    dr3_dope = du_l_dope.mean()
    
    df_s_linear, df_s_cubic, uct_s, s0p,s1p = cubic(du_s_free,du_s_dope,T,plot=plot,label='solid')
    
    df_l_linear, df_l_cubic, uct_l, l0p,l1p = cubic(du_l_free,du_l_dope,T,plot=plot,label='liquid')
    plt.legend()
    
    ddf = uct.ufloat(df_s_linear,uct_s) - uct.ufloat(df_l_linear,uct_l) # s - l because Ds/l
    ddf_cub = df_s_cubic - df_l_cubic
    
    dG      =  ddf + dPV
    dG_cub  =  ddf_cub - dPV
    D     = uct.umath.exp(dG/T/kb)
    D_cub = uct.umath.exp(dG_cub/T/kb) 
    
    data = [r1.mean(),r1_traj.mean(),r1_dope.mean(),r1_dope_traj.mean(),
            r3.mean(),r3_traj.mean(),r3_dope.mean(),r3_dope_traj.mean(),
            dr1, dr1_dope, dr3, dr3_dope,
            df_s_linear,df_s_cubic,df_l_linear,df_l_cubic,s0p,s1p,l0p,l1p,
            ddf.nominal_value,ddf_cub,dPV,dG.nominal_value, dG_cub,ddf.std_dev, D.nominal_value, D_cub,D.std_dev,
            uct.umath.log10(D).nominal_value,uct.umath.log10(D_cub), uct.umath.log10(D).std_dev]
    out = pd.DataFrame([data],columns=columns)   
    return out


def cubic_err(du_free,du_dope,T,plot=False,label='test'):
    """
    ax^3 + bx^2 + cx^ + d = f(x);  x is lambda
    
    Notes
    ----
    we have f(0) = d, f'(0) = c, f(1) = a+b+c+d; f'(1) = 3a+2b+c
    
    """
    f0 = err(du_free); f0p = -msf(du_free)/kb/T
    f1 = err(du_dope); f1p = -msf(du_dope)/kb/T
    c  = f0p; d = f0
    a  = f1p + f0p - 2*f1 + 2*f0
    b  = 3*f1 - 3*f0 - f1p - 2*f0p
    df_linear = (err(du_free)+err(du_dope))/2
    df_cubic  = a/4 + b/3 + c/2 + d
    
    # if plot:
    #     l  = np.linspace(0,1)
    #     fl = a*(l**3) + b*(l**2) + c*l + d
    #     plt.plot(l,fl,label=label)
    #     plt.plot([0,1],[f0,f1],'k--o')
    #     plt.xlabel(r'$\lambda$',fontsize=14)
    #     plt.ylabel('<U(dope) - U(free)>',fontsize=14)
    return df_linear.nominal_value, df_cubic.nominal_value, df_linear.std_dev,df_cubic.std_dev, f0p,f1p

def analyze_err(r1,r1_traj,r1_dope,r1_dope_traj,r3,r3_traj,r3_dope,r3_dope_traj,T,dPV=0,plot=True):
    """
    analyze the TI results
    consider uncertainties throughout
    
    Params
    ------
    x & x_traj : arrays 1d, internal energy
                 x_traj and x must have the same dimension, 
                 but r1 and r3 do not have 
    T : float, temperature
    dPV : PV of l - PV of s, default is 0
    plot : plot the <dU> vs. lambda, default is true
    
    Returns
    ------
    out : panadas DataFrame

    """
    import pandas as pd
    du_s_free = r1_traj - r1
    du_s_dope = r1_dope - r1_dope_traj
    
    du_l_free = r3_traj - r3
    du_l_dope = r3_dope - r3_dope_traj

    columns = [
               'dr1','dr1-dope','dr3','dr3-dope',
               'dF_s','dF_s_cub','dF_l','dF_l_cub', 
               'ddF','ddF_cub','dPV','dG', 'dG_cub','uct_dG','uct_dG_c',
               'log10_D','log10_D_cub','log10_uct_D']
    
    dr1      = du_s_free.mean()
    dr1_dope = du_s_dope.mean()
    dr3      = du_l_free.mean()
    dr3_dope = du_l_dope.mean()
    
    df_s_linear, df_s_cubic, uct_s, uct_cs, s0p,s1p = cubic_err(du_s_free,du_s_dope,T,plot=plot,label='solid')
    
    df_l_linear, df_l_cubic, uct_l, uct_cl, l0p,l1p = cubic_err(du_l_free,du_l_dope,T,plot=plot,label='liquid')
    plt.legend()
    
    ddf = uct.ufloat(df_s_linear,uct_s) - uct.ufloat(df_l_linear,uct_l) # s - l because Ds/l
    ddf_cub = uct.ufloat(df_s_cubic,uct_cs) - uct.ufloat(df_l_cubic,uct_cl)
    
    dG      =  ddf + dPV
    dG_cub  =  ddf_cub - dPV
    D     = uct.umath.exp(dG/T/kb)
    D_cub = uct.umath.exp(dG_cub/T/kb) 
    
    data = [dr1, dr1_dope, dr3, dr3_dope,
            df_s_linear,df_s_cubic,df_l_linear,df_l_cubic,
            ddf.nominal_value,ddf_cub,dPV,dG.nominal_value, dG_cub,dG.std_dev, dG_cub.std_dev,
            uct.umath.log10(D).nominal_value,uct.umath.log10(D_cub), uct.umath.log10(D).std_dev,uct.umath.log10(D_cub).std_dev]
    out = pd.DataFrame([data],columns=columns)   
    return out

