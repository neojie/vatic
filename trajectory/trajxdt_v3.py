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

### TODO
### make it a class
### PBC- done
### same atom same clor -done
### select element mode - done
### select atom mode - done 
### use dictionary to simplify 'Fe' match - done
### projection -done
### projection on x, y, z -done
### set as standard class  -done
### show basic info of imported XDATCAR
### set figure handle as self property
### line transparency 

class trajxdt:
    def __init__(self,stepsize=15,File=None):
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

        self.eleName = None
        self.Niter = None
        self.NiterChosed = None
        self.car   = None
        

        ## build color dictionary
        colorlist = ['r','b','m','y']
        elelist   = ['Fe','Si','O','H']
        self.colordic  = dict(zip(elelist,colorlist))
        ## define element list dictionary
        ## for Fe O, 32, 64, eledic['Fe']= 1..32, 
        self.eledic = None        
        self.readxdat()
#        self.Tra3D()
#        self.sliceTra()
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
        
        self.eleName = xdt[5].rstrip('\n').split()
        # 6- element number
        ele_str     = xdt[6].rstrip('\n').split()
        self.eleNum = list(map(int,ele_str))
        self.eleSum = np.sum(self.eleNum)
        self.Niter  = int(xdt[-self.eleSum-1].rstrip().split('=')[-1])
        atom_list_order = [None]*np.size(self.eleName)
        for i in np.arange(np.size(self.eleName)):
            if i == 0:
                atom_list_order[i] = np.arange(self.eleNum[i])+1
            else:     
                atom_list_order[i] = np.arange(np.sum(self.eleNum[:i]),np.sum(self.eleNum[:i+1]),1)+1
        self.eledic = dict(zip(self.eleName,atom_list_order))
            
        """ Restore data as pandas """
        ele_index = []
        for i in np.arange(np.size(self.eleName)):
            ele_index = ele_index+[self.eleName[i]]*self.eleNum[i]
#            print("creating ele_index, loop is", i)
#            print(ele_index)
            
        ele_num_index  = list(np.arange(self.eleSum)+1)
        
        iterChosed  = np.arange(1,self.Niter ,self.stepsize)
        self.NiterChosed = np.size(iterChosed)
#        print("iterChosed is",iterChosed)
        
        count = 0
        for niter in iterChosed:
            iter_num_index = [niter]*self.eleSum
#            print(ele_index)
            arrays = [iter_num_index,ele_index,ele_num_index]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=['iter','ele', 'ele_num'])
#            print("index chosen is",index)
#            print("niter chosed is",niter)
            row_start = 9+(niter-1)*(self.eleSum+1)
            row_end   = 9+self.eleSum-1+(niter-1)*(self.eleSum+1)
            ## I did not show frational coordinate
            xyz_fra   = np.loadtxt(xdt[(row_start-1):(row_end)])  
#            print("starting row is", row_start)
#            print("end row is", row_end)
#            print("xyz_fra is", xyz_fra)
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
#            print(car)
            #print("count is",count)
        self.car = car
        xdatcar.close()

    def sliceTra(self,atom):
        """ 
        For a given atom, judge if one needs slice the trajectory
        Consider PB condition, 
        otherwise a lot of long straight line appears 
        """                
        ### PBC if the migrates exceeds half of the mininal dimension -> cross PB
        pos_diff  = np.diff(atom,axis=0)
        statement = np.abs(pos_diff) >= min([self.a1[0]*self.scale, self.a2[1]*self.scale, self.a3[2]*self.scale])/2
        ### true row is the indix of row for which the atom moves too much in given step
        true_row,true_col = np.where(statement)
        true_row  = sorted(set(true_row))
        ### divide atom into several groups based on true_row, true col
        seg_num     = np.size(true_row)+1
        iloc_list   = list(np.arange(0,self.NiterChosed)) 
#        print("atom is")
#        print(atom)
#        print("seg # is",seg_num)
#        print("False row is",true_row)
#        print("statement",statement)
        # No segments
        if seg_num == 1:
#            ax.plot(atom['x'], atom['y'], atom['z'],color=colorlist[i],label=ele)
            iloc_list_all = iloc_list
            ## This is critical, see error 1 
            iloc_list_all = [iloc_list_all]
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
        return iloc_list_all
    
    def plotTra3D(self,atom,iloc_list_all,ax,ele):
        for c, value in enumerate(iloc_list_all):               
            atom_iloc = atom.iloc[value]
#            print("atom_iloc is",atom_iloc)
#            print("ele", ele)
#            print("value of the iloc_list is",value)
#            print("atom_iloc is",atom_iloc)
#            print("x is",atom_iloc['x'].values)
#            print("y is",atom_iloc['y'].values)
#            print("z is",atom_iloc['z'].values)
            ax.plot(atom_iloc['x'], atom_iloc['y'], atom_iloc['z'],color=self.colordic[ele],label=ele)
#            ax.plot(atom_iloc['x'], atom_iloc['y'], atom_iloc['z'])
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

    def plotTra2D(self,atom,iloc_list_all,ax1,ax2,ax3,ele):
        for c, value in enumerate(iloc_list_all):               
            atom_iloc = atom.iloc[value]            
            ax1.plot(atom_iloc['x'], atom_iloc['y'],color=self.colordic[ele],label=ele)
            ax2.plot(atom_iloc['x'], atom_iloc['z'],color=self.colordic[ele],label=ele)
            ax3.plot(atom_iloc['y'], atom_iloc['z'],color=self.colordic[ele],label=ele)
            self.legend(ax1)
            self.legend(ax2)
            self.legend(ax3)
#            ax.plot(atom_iloc['x'], atom_iloc['y'], atom_iloc['z'])

    def legend(self,ax_new):
        handles, labels = ax_new.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax_new.legend(by_label.values(), by_label.keys())  
        
    def Tra3D(self,elements='All',atom_index='All',elev=None,azim=None): 
        """
        elements, must be string list ['Fe'], not 'Fe'
        atom_index, 1-same number
        elev,azim=ax.view_init function 
        'elev' stores the elevation angle in the z plane.
        'azim' stores the azimuth angle in the x,y plane.
        """

        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if elements is 'All' and (atom_index=='All'):     
            for ele in self.eleName:
                atom_list = self.eledic[ele]
                for j in atom_list:
                    atom           = self.car.loc(axis=0)[:, :, j]
                    atom_iloc_list = self.sliceTra(atom)
                    self.plotTra3D(atom,atom_iloc_list,ax,ele)
                    
        elif (elements !='All') and (atom_index=='All'):
            for ele in elements:
                atom_list = self.eledic[ele]
                
                for j in atom_list:
                    atom           = self.car.loc(axis=0)[:, :, j]
                    atom_iloc_list = self.sliceTra(atom)
                    self.plotTra3D(atom,atom_iloc_list,ax,ele)
            
        elif (elements is 'All') and (atom_index!= 'All'):
            atom = self.car.loc(axis=0)[:, :, atom_index]
#            print(atom)
            ele  = atom.index.values[0][1]
#            print(ele)
            atom_iloc_list = self.sliceTra(atom)
            self.plotTra3D(atom,atom_iloc_list,ax,ele)
        else:
            print("input wrong")
        plt.xlim([0,self.a1[0]*self.scale])
        plt.ylim([0,self.a2[1]*self.scale])
        plt.xlabel('a1')
        plt.ylabel('a2')
#        plt.zlabel('a3')
        #plt.zlim([0,self.a3[2]*self.scale])
        ax.view_init(elev, azim)
        plt.show()       
 
    def Tra2D(self,elements='All',atom_index='All'): 
        """
        2D projection
        elements, must be string list ['Fe'], not 'Fe'
        atom_index, 1-same number
        elev,azim=ax.view_init function 
        'elev' stores the elevation angle in the z plane.
        'azim' stores the azimuth angle in the x,y plane.
        """

        mpl.rcParams['legend.fontsize'] = 10
        plt.figure(figsize=[14,4])
        ## ax1 - xy
        ax1 = plt.subplot(1,3,1)
        ax1.set_xlabel('a1')
        ax1.set_ylabel('a2')
        ax1.set_xlim([0, self.a1[0]*self.scale])
        ax1.set_ylim([0, self.a2[1]*self.scale])
        ## ax2 - xz        
        ax2 = plt.subplot(1,3,2)
        ax2.set_xlabel('a1')
        ax2.set_ylabel('a3')
        ax2.set_xlim([0, self.a1[0]*self.scale])
        ax2.set_ylim([0, self.a3[2]*self.scale])
        ## ax2 - yz
        ax3 = plt.subplot(1,3,3)
        ax3.set_xlabel('a2')
        ax3.set_ylabel('a3')
        ax3.set_xlim([0, self.a2[1]*self.scale])
        ax3.set_ylim([0, self.a3[2]*self.scale])
        plt.subplot
        if elements is 'All' and (atom_index=='All'):     
            for ele in self.eleName:
                atom_list = self.eledic[ele]
                for j in atom_list:
                    atom           = self.car.loc(axis=0)[:, :, j]
                    atom_iloc_list = self.sliceTra(atom)
                    self.plotTra2D(atom,atom_iloc_list,ax1,ax2,ax3,ele)
                    
        elif (elements !='All') and (atom_index=='All'):
            for ele in elements:
                atom_list = self.eledic[ele]
                
                for j in atom_list:
                    atom           = self.car.loc(axis=0)[:, :, j]
                    atom_iloc_list = self.sliceTra(atom)
                    self.plotTra2D(atom,atom_iloc_list,ax1,ax2,ax3,ele)
            
        elif (elements is 'All') and (atom_index!= 'All'):
            atom = self.car.loc(axis=0)[:, :, atom_index]
#            print(atom)
            ele  = atom.index.values[0][1]
            #print(atom)
            #print("ele is",ele)
#            print(ele)
            atom_iloc_list = self.sliceTra(atom)
            self.plotTra2D(atom,atom_iloc_list,ax1,ax2,ax3,ele)
        else:
            print("input wrong")
        plt.show()       


# test code of the above class
#if __name__ == '__main__':
#    inp = trajxdt(stepsize=1000)
#    #inp.Tra3D()
#    inp.Tra2D(elements=['Fe'])
    
    ####
   

    
### Errors-1
## car.iloc[[2]] vs. car.iloc[2], although result value is the same, output is totally different
