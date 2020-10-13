#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:58:16 2019

@author: jiedeng
"""
import os
import pickle

def get_variable_name(var,global_vars):
    """
    Notes
    -----
    may not work for var with different shape
    """
#    tmp = globals()
    tmp = global_vars
    for key in tmp:        
        if (not (key[0] is '_')) and (type(tmp[key]) == type(var)):
            if tmp[key] is var:
                return key
            
def psave(*argv,global_vars = globals()):  
    """
    save arguments in the folder presults
    
    Params
    -----
    *argv : arguments
    **Description :  future
    
    Example
    -----
    psave(rdfs_homo,first_peaks_homo,first_trenches_homo)

    """
    if not('presult' in os.listdir()):
        print("No Pickel result folder exist, rebuild")
        os.mkdir('presult')
    print("global_vars is",global_vars)
    for arg in argv:
        arg_name = get_variable_name(arg,global_vars)
        print(arg_name)
        with open('presult/'+arg_name, 'wb') as fp:
            pickle.dump(arg, fp, protocol=pickle.HIGHEST_PROTOCOL)

def pread(*argv,**Description):  
    """
    read arguments in the folder presult
    
    Params
    ------
    *argv : future
    
    Returns
    -----
    out : list of variables, the name of each is printed out during run-time
    """
    out = []
    if not ('presult' in os.listdir()):
        print("No Pickel result folder exist!")
    else:   
        print("Arguements are: ")
        argvs = os.listdir('presult')
        for arg in argvs:
            if arg[0] != '.':
                print(arg)
                with open('presult/' + arg, 'rb') as fp:
                    tmp = pickle.load(fp)
                out.append(tmp)    
    return out


def pread_single(file):  
    """
    read arguments in the folder presult
    
    Params
    ------
    file : file
    
    Returns
    -----
    tmp : variable
    """

    with open(file, 'rb') as fp:
        tmp = pickle.load(fp)
    return tmp