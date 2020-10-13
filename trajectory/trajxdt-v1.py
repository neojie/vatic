#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:43:41 2018
This is based on msdpre-JD.py
@author: jiedeng
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import OrderedDict


class trajxdt:
    def __init__(self,stepsize=10,File=None):
        if File is None:
            self.xdatcar = 'XDATCAR'
        else:
            self.xdatcar = File
        # step gap to buil trajectory
        self.stepsize = stepsize
        self.system = None
        # time step of MD
        self.potim = None
        # mass per type
        self.scale =None

        self.TypeName = None
        self.ChemSymb = None
        self.Ntype = None
        self.Nions = None
        self.Nelem = None
        self.Niter = None
        self.NiterChosed = None
        self.car   = None
        self.readxdat()
        
    def readxdat(self):
        """ Read VASP XDATCAR """
        xdatcar = open(self.xdatcar, 'r')
        xdt     = xdatcar.readlines()
        self.system = xdt[1]
        self.scale  = float(xdt[1].rstrip('\n'))
        
        a1_map = list(map(float,xdt[2].rstrip('\n').split()))
        a2_map = list(map(float,xdt[3].rstrip('\n').split()))
        a3_map = list(map(float,xdt[4].rstrip('\n').split()))
        self.a1 = self.scale*np.array(a1_map)
        self.a2 = self.scale*np.array(a2_map)
        self.a3 = self.scale*np.array(a3_map)
        
        self.ChemSymb = xdt[5].rstrip('\n').split()
        # 6- element number
        ele_str     = xdt[6].rstrip('\n').split()
        self.eleNum = list(map(int,ele_str))
        self.eleSum = np.sum(self.eleNum)
        self.Niter  = int(xdt[-self.eleSum-1].rstrip().split('=')[-1])
    
        """ Restore data as pandas """
        ele_index = []
        for i in np.arange(np.size(self.ChemSymb)):
            ele_index = [self.ChemSymb[i]]*self.eleNum[i]+ele_index
        ele_num_index  = list(np.arange(self.eleSum)+1)
        
        iterChosed  = np.arange(1,self.Niter ,self.stepsize)
        self.NiterChosed = np.size(iterChosed)
        
        count = 0
        for niter in iterChosed:
            iter_num_index = [niter]*self.eleSum
            arrays = [iter_num_index,ele_index,ele_num_index]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=['iter','ele', 'ele_num'])
            
            row_start = 9+niter*(self.eleSum+1)
            row_end   = 9+self.eleSum-1+niter*(self.eleSum+1)
            ## I did not show frational coordinate
            xyz_fra   = np.loadtxt(xdt[(row_start-1):row_end])   
            xyz_car   = xyz_fra*[self.a1[0],self.a2[1],self.a3[2]]
            #ele1=xyz_fra[:ele_num[0]]
            df_xyz_car = pd.DataFrame(xyz_car,index =index,columns=['x','y','z'])
            #df_xyz_fra = pd.DataFrame(xyz_fra,index =index,columns=['x','y','z'])
            if count == 0:
                car = df_xyz_car
            #    fra = df_xyz_fra
            else:
                car = car.append(df_xyz_car)
            #    fra = fra.append(df_xyz_fra)
            count =count+1
            #print("count is",count)
        self.car = car
        xdatcar.close()
    
    def viewTra(self):
        colorlist = ['r','b','k','g','y','m']
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        for i in np.arange(np.size(self.eleNum)):
            if i == 0:
                atom_list = np.arange(self.eleNum[i])+1
            else:     
                atom_list = np.arange(np.sum(self.eleNum[:i]),np.sum(self.eleNum[:i+1]),1)+1
#            print(atom_list)
        
            for j in atom_list:
                atom = self.car.loc(axis=0)[:, :, j]
                
                ### PBC if the migrates exceeds half of the mininal dimension -> cross PB
                pos_diff  = np.diff(atom,axis=0)
                statement = np.abs(pos_diff) >= min([self.a1[0]*self.scale, self.a2[1]*self.scale, self.a3[2]*self.scale])/2
                true_row,true_col = np.where(statement)
                true_row  = sorted(set(true_row))
                ### divide atom into several groups based on true_row, true col
                seg_num     = np.size(true_row)+1
                iloc_list   = list(np.arange(0,self.NiterChosed))    
#                print("seg # is",seg_num)
#                print("False row is",true_row)
#                print("statement",statement)
                # No segments
                if seg_num == 1:
                    ax.plot(atom['x'], atom['y'], atom['z'],color=colorlist[i],label=self.ChemSymb[i])
                # Segments =1, only one breakpoint
                elif seg_num == 2:
                    iloc_list_all = [None]*2
                    iloc_list_all[0] = iloc_list[:(true_row[0])+1]
                    iloc_list_all[1] = iloc_list[(true_row[0])+1:]
                 # Segments =1, only one breakpoint
                else:
                    iloc_list_all = [None]*seg_num
                    iloc_list_all[0] = iloc_list[:(true_row[0]+1)]
                    iloc_list_all[-1] = iloc_list[true_row[-1]+1:]
                    for seg in np.arange(seg_num-1):
                        if seg+1 < np.size(true_row) and true_row[seg+1] <self.NiterChosed:
                            iloc_list_all[seg+1] = iloc_list[(true_row[seg]+1):(true_row[seg+1]+1)]
#                print("iloc_list_all",iloc_list_all)      
                for c, value in enumerate(iloc_list_all):
#                    print(c)
#                    print(value)                   
                    atom_iloc = atom.iloc[value]
#                    print("atom_iloc",atom_iloc)
                    ax.plot(atom_iloc['x'], atom_iloc['y'], atom_iloc['z'],color=colorlist[i],label=self.ChemSymb[i])
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = OrderedDict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())
        plt.show()

# test code of the above class
if __name__ == '__main__':
    inp = trajxdt(stepsize=1000)
    # plt.plot((np.abs(fft(inp.VAF[inp.Niter-2:]))**2))
    # print inp.VAF.shape
    # plt.plot(inp.Time, inp.VAF, 'ko-', lw=1.0, ms=2,
    #         markeredgecolor='r', markerfacecolor='red')
    # 
    # plt.xlabel('Time [fs]')
    # plt.ylabel('Velocity Autocorrelation Function')

    # x, y = inp.phononDos('cm-1')
    # plt.plot(x, y, 'ko-')
    # plt.xlim(0, 5000)
    # # plt.ylim(-0.5, 1.0)
