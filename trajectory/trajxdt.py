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
### show basic info of imported XDATCAR -done
### set figure handle as self property  --done
### line transparency  --half done
### show trajectory at given timeframe

### Errors-1
## car.iloc[[2]] vs. car.iloc[2], although result value is the same, output is totally different

class trajxdt:
    def __init__(self,stepsize=15,File=None,Nitermin=1,Nitermax=None,colordic=None):
        if File is None:
            self.xdatcar = 'XDATCAR'
        else:
            self.xdatcar = File
        # step gap to buil trajectory
        self.stepsize = stepsize
        self.system = None
        # mass per type
        self.scale =None

        self.eleName = None
        self.Niter = Nitermax
        self.Nitermin = Nitermin
        self.NiterChosed = None
        self.car   = None
        self.fig   = None
        
        

        ## build color dictionary
        colorlist = ['r','b','m','y']
        elelist   = ['Fe','Si','O','H']
        if colordic is None:
            self.colordic  = dict(zip(elelist,colorlist))
        else:
            self.colordic  = colordic
            
        ## define element list dictionary
        ## for Fe O, 32, 64, eledic['Fe']= 1..32, 
        self.eledic = None  
        # put the function here so that it will run atomatically
        self.readxdat()
#        self.tra3D()
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
        
        if self.Niter is None:
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
            
        ele_num_index  = list(np.arange(self.eleSum)+1)
        
        iterChosed  = np.arange(self.Nitermin,self.Niter ,self.stepsize)
        self.NiterChosed = np.size(iterChosed)
        
        #### print info ####
        print(self.eleName)
        print(self.eleNum)
        print('',self.a1,'\n',self.a2,'\n',self.a3)
        print("Number of total iteration is:", self.Niter)
        print("Number of iteration chosen is:",self.NiterChosed)        
        #### print info ####        
        
#        print("iterChosed is",iterChosed)
        
        count = 0
        for niter in iterChosed:
            iter_num_index = [niter]*self.eleSum
#            print(ele_index)
            arrays = [iter_num_index,ele_index,ele_num_index]
            tuples = list(zip(*arrays))
            index  = pd.MultiIndex.from_tuples(tuples, names=['iter','ele', 'ele_num'])
#            print("index chosen is",index)
#            print("niter chosed is",niter)
            row_start  = 9+(niter-1)*(self.eleSum+1)
            row_end    = 9+self.eleSum-1+(niter-1)*(self.eleSum+1)
            ## I did not show frational coordinate
            xyz_fra    = np.loadtxt(xdt[(row_start-1):(row_end)])  
#            print("starting row is", row_start)
#            print("end row is", row_end)
#            print("xyz_fra is", xyz_fra)
            xyz_car    = xyz_fra*[self.a1[0],self.a2[1],self.a3[2]]
            #ele1=xyz_fra[:ele_num[0]]
            df_xyz_car = pd.DataFrame(xyz_car,index =index,columns=['x','y','z'])
            #df_xyz_fra = pd.DataFrame(xyz_fra,index =index,columns=['x','y','z'])
            if count == 0:
                car = df_xyz_car
            #    fra = df_xyz_fra
            else:
                car = car.append(df_xyz_car)
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
        true_row    = sorted(set(true_row))
        ### divide atom into several groups based on true_row, true col
        seg_num     = np.size(true_row)+1
        iloc_list   = list(np.arange(0,self.NiterChosed)) 
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
    
    def plottra3D(self,atom,iloc_list_all,ax,ele,alpha=1):
        for c, value in enumerate(iloc_list_all):               
            atom_iloc = atom.iloc[value]
            ax.plot(atom_iloc['x'], atom_iloc['y'], atom_iloc['z'],color=self.colordic[ele],label=ele,alpha=alpha)
#            ax.plot(atom_iloc['x'], atom_iloc['y'], atom_iloc['z'])
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

    def plottra2D(self,atom,iloc_list_all,ax1,ax2,ax3,ele,alpha=1):
        for c, value in enumerate(iloc_list_all):               
            atom_iloc = atom.iloc[value]            
            ax1.plot(atom_iloc['x'], atom_iloc['y'],color=self.colordic[ele],label=ele,alpha=alpha)
            ax2.plot(atom_iloc['x'], atom_iloc['z'],color=self.colordic[ele],label=ele,alpha=alpha)
            ax3.plot(atom_iloc['y'], atom_iloc['z'],color=self.colordic[ele],label=ele,alpha=alpha)

            self.legend(ax1)
            self.legend(ax2)
            self.legend(ax3)
#            ax.plot(atom_iloc['x'], atom_iloc['y'], atom_iloc['z'])

    def legend(self,ax_new):
        handles, labels = ax_new.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax_new.legend(by_label.values(), by_label.keys())  
        
    def tra3D(self,elements='All',atom_index='All',elev=None,azim=None,alpha=0.6,showfig=True): 
        """
        elements, must be string list ['Fe'], not 'Fe'
        atom_index, 1-same number
        elev,azim=ax.view_init function 
        'elev' stores the elevation angle in the z plane.
        'azim' stores the azimuth angle in the x,y plane.
        """

        mpl.rcParams['legend.fontsize'] = 10
        self.fig = plt.figure()
        ax = self.fig.gca(projection='3d')
        if elements is 'All' and (atom_index=='All'):     
            for ele in self.eleName:
                atom_list = self.eledic[ele]
                for j in atom_list:
                    atom           = self.car.loc(axis=0)[:, :, j]
                    atom_iloc_list = self.sliceTra(atom)
                    self.plottra3D(atom,atom_iloc_list,ax,ele,alpha)
                    
        elif (elements !='All') and (atom_index=='All'):
            for ele in elements:
                atom_list = self.eledic[ele]
                
                for j in atom_list:
                    atom           = self.car.loc(axis=0)[:, :, j]
                    atom_iloc_list = self.sliceTra(atom)
                    self.plottra3D(atom,atom_iloc_list,ax,ele,alpha)
            
        elif (elements is 'All') and (atom_index!= 'All'):
            atom = self.car.loc(axis=0)[:, :, atom_index]
#            print(atom)
            ele  = atom.index.values[0][1]
#            print(ele)
            atom_iloc_list = self.sliceTra(atom)
            self.plottra3D(atom,atom_iloc_list,ax,ele,alpha)
        else:
            print("input wrong")
        plt.xlim([0,self.a1[0]*self.scale])
        plt.ylim([0,self.a2[1]*self.scale])
        plt.xlabel('a1')
        plt.ylabel('a2')
#        plt.zlabel('a3')
        #plt.zlim([0,self.a3[2]*self.scale])
        ax.view_init(elev, azim)
        # ax, alpha setting why not working here
#        ax.set_alpha(alpha)
        if showfig:
            plt.show()       
 
    def tra2D(self,elements='All',atom_index='All',alpha=0.6,showfig=True,showlegend=True): 
        """
        2D projection
        elements, must be string list ['Fe'], not 'Fe'
        atom_index, 1-same number
        """

        mpl.rcParams['legend.fontsize'] = 9
        xy_label_fontsize = 8
        xy_tick_fontsize  = 7
        self.fig=plt.figure(figsize=[9.2,2.75])
        ## ax1 - xy
        ax1 = plt.subplot(1,3,1)
        ax1.set_xlabel(r'$\mathit{a}\mathrm{1}$' +' '+r'$(\AA)$',fontsize=xy_label_fontsize)
        ax1.set_ylabel(r'$\mathit{a}\mathrm{2}$' +' '+r'$(\AA)$',fontsize=xy_label_fontsize)
        ax1.tick_params(axis='both',labelsize=xy_tick_fontsize)
        # once axis('equal') is added, the result became very wierd
#        ax1.axis('equal')
#        ax1.set_alpha(alpha)
        ## ax2 - xz        
        ax2 = plt.subplot(1,3,2)
        ax2.set_xlabel(r'$\mathit{a}\mathrm{1}$'+' '+r'$(\AA)$',fontsize=xy_label_fontsize)
        ax2.set_ylabel(r'$\mathit{a}\mathrm{3}$'+' '+r'$(\AA)$',fontsize=xy_label_fontsize)
        ax2.tick_params(axis='both',labelsize=xy_tick_fontsize)
#        plt.axis([0, self.a1[0]*self.scale,0, self.a3[2]*self.scale])
#        ax2.axis('equal')
#        ax2.set_alpha(alpha)
        ## ax2 - yz
        ax3 = plt.subplot(1,3,3)
        ax3.set_xlabel(r'$\mathit{a}\mathrm{2}$'+' '+r'$(\AA)$',fontsize=xy_label_fontsize)
        ax3.set_ylabel(r'$\mathit{a}\mathrm{3}$'+' '+r'$(\AA)$',fontsize=xy_label_fontsize)
        ax3.tick_params(axis='both',labelsize=xy_tick_fontsize)
#        plt.axis([0, self.a2[1]*self.scale,0, self.a3[2]*self.scale])
#        plt.axis('equal')
        
        if elements is 'All' and (atom_index=='All'):     
            for ele in self.eleName:
                atom_list = self.eledic[ele]
                for j in atom_list:
                    atom           = self.car.loc(axis=0)[:, :, j]
                    atom_iloc_list = self.sliceTra(atom)
                    self.plottra2D(atom,atom_iloc_list,ax1,ax2,ax3,ele,alpha)
                    
        elif (elements !='All') and (atom_index=='All'):
            for ele in elements:
                atom_list = self.eledic[ele]
                
                for j in atom_list:
                    atom           = self.car.loc(axis=0)[:, :, j]
                    atom_iloc_list = self.sliceTra(atom)
                    self.plottra2D(atom,atom_iloc_list,ax1,ax2,ax3,ele,alpha)
            
        elif (elements is 'All') and (atom_index!= 'All'):
            atom = self.car.loc(axis=0)[:, :, atom_index]
#            print(atom)
            ele  = atom.index.values[0][1]
            #print(atom)
            #print("ele is",ele)
#            print(ele)
            atom_iloc_list = self.sliceTra(atom)
            self.plottra2D(atom,atom_iloc_list,ax1,ax2,ax3,ele,alpha)
        else:
            print("input wrong")
            
        ax1.set_xlim([0, self.a1[0]*self.scale])
        ax1.set_ylim([0, self.a2[1]*self.scale])
        ax2.set_xlim([0, self.a1[0]*self.scale])
        ax2.set_ylim([0, self.a3[2]*self.scale])
        ax3.set_xlim([0, self.a2[1]*self.scale])
        ax3.set_ylim([0, self.a3[2]*self.scale])
        if showlegend is False:
            ax1.legend_.remove()
            ax2.legend_.remove()
            ax3.legend_.remove()
            
        if showfig:
            plt.show()  
    
    def dsplot(self,ele):
        """
        datashader plot
        ele: element, must be char 'Si' instead of list ['Si']      
        """
        import datashader as ds
        from datashader import transfer_functions as tf
        df = self.car.loc(axis=0)[:, :, self.eledic[ele]]
        cvs = ds.Canvas(plot_height=200, plot_width=200)
        agg_xy = cvs.points(df, 'x', 'y')
        img_xy = tf.shade(agg_xy, how='eq_hist')
        agg_yz = cvs.points(df, 'y', 'z')
        img_yz = tf.shade(agg_yz, how='eq_hist')
        agg_xz = cvs.points(df, 'x', 'z')
        img_xz = tf.shade(agg_xz, how='eq_hist')
        ds.utils.export_image(img_xy, ele+'xy')
        ds.utils.export_image(img_yz, ele+'yz')
        ds.utils.export_image(img_xz, ele+'xz')


        
    def his2D(self,elements='All',atom_index='All',alpha=0.6,showfig=True,ngrid=10):
    
        """
        2D count the atoms 
        refer to /Users/jiedeng/Box Sync/Learn/Python_learn/pandas_learn.py       
        """
#        dim = self.a1[0]*self.scale
#        
#        x=np.linspace(0,dim,ngrid)
        
    def his3D(self,elements='All',atom_index='All',alpha=0.6,showfig=True,ngrid=10):

        """
        sD count the atoms 
        refer to /Users/jiedeng/Box Sync/Learn/Python_learn/pandas_learn.py       
        """
        if elements is 'All' and (atom_index=='All'):     
            atom  = self.car
                    
        elif (elements !='All') and (atom_index=='All'):
            for ele in elements:
                atom_list = self.eledic[ele]
                atom      = self.car.loc(axis=0)[:, :, atom_list]
            
        elif (elements is 'All') and (atom_index!= 'All'):
            atom = self.car.loc(axis=0)[:, :, atom_index]
            ele  = atom.index.values[0][1]
        else:
            print("input wrong")
            
        #ngrid = ngrid-1    
        dim = self.a1[0]*self.scale        
        x   = np.linspace(0,dim,ngrid) 
        dis = x[1]-x[0]
        df  = atom/dis             
        # append one more col
        df['u'] = pd.Series(np.ones(df.shape[0]), index=df.index)
        dfs = df.astype(int).groupby(['x','y','z']).count()
        ngrid = ngrid-1
        base  = list(range(ngrid))
        x_index = []
        y_index = []
        for i in base:
            x_index = x_index+[i]*(ngrid**2)
        for i in base:
            y_index = y_index+[i]*ngrid
        y_index = y_index*ngrid
        z_index = base*(ngrid**2)  
        arrays = [x_index,y_index,z_index]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['x','y', 'z'])
        df_org = pd.DataFrame(np.zeros(ngrid**3),index =index,columns=dfs.columns)      
        temp = df_org+dfs
        self.atom_count = temp.fillna(0)
        
#    def count(self,atom,dis): 


        
# test code of the above class
#if __name__ == '__main__':
#    inp = trajxdt(stepsize=1000)
#    #inp.tra3D()
#    inp.tra2D(elements=['Fe'])
    
    ####
   

    
