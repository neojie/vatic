#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:34:56 2019
ref : 
    https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib

@author: jiedeng
"""
import numpy as np
from itertools import product, combinations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

def switch_y_z(inter,liq_cutoff,vap_cutoff,liq_in,vap_in,int_in):
    triangles   = inter.triangulated_surface[0][inter.triangulated_surface[1]]
    

    interface1  = np.zeros_like(triangles)
    interface2  = np.zeros_like(triangles)
        
    xlim, zlim, ylim = inter.universe.dimensions[0],inter.universe.dimensions[1],inter.universe.dimensions[2]
    
    for i in range(len(triangles)):
        ## swap y and z
        tmp = np.array([triangles[i][:,0],triangles[i][:,2],triangles[i][:,1]]).T
        if triangles[i][:,-1][0] < zlim:
            interface1[i] = tmp + np.array([0,liq_cutoff,0])
            interface2[i] = tmp + np.array([0,vap_cutoff,0])
        else:
            interface1[i] = tmp - np.array([0,liq_cutoff,0])
            interface2[i] = tmp - np.array([0,vap_cutoff,0])    
    return xlim,zlim,ylim,interface1,interface2
      
def plot_interfaces(inter,liq_cutoff,vap_cutoff,liq_in,vap_in,int_in,box_color='k'):
  
    xlim,zlim,ylim,interface1,interface2 = switch_y_z(inter,liq_cutoff,vap_cutoff,liq_in,vap_in,int_in)
    fig = plt.figure(figsize=(12, 12))
    
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.grid(False)

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
#    mesh1 = Poly3DCollection(triangles)
#    mesh1.set_edgecolor('none')
#    mesh1.set_alpha(0.3)
#    ax1.add_collection3d(mesh1)
    
    mesh2 = Poly3DCollection(interface1)
    mesh2.set_edgecolor('none')
    mesh2.set_alpha(0.3)
    ax1.add_collection3d(mesh2)
    
    mesh3 = Poly3DCollection(interface2)
    mesh3.set_edgecolor('none')
    mesh3.set_alpha(0.3);
#    mesh3.set_facecolor('b')
    ax1.add_collection3d(mesh3)
    
    pos = inter.universe.atoms.positions
    pos = np.array([pos[:,0],pos[:,2],pos[:,1]]).T
    pos_liq = pos[liq_in]
    xyz_liq = np.vstack([pos_liq[::, 0], pos_liq[::, 1], pos_liq[::, 2]])
    
    ax1.scatter(xyz_liq[0],xyz_liq[1],xyz_liq[2],color='r')
    
    pos_vap = pos[vap_in]
    xyz_vap = np.vstack([pos_vap[::, 0], pos_vap[::, 1], pos_vap[::, 2]])
    ax1.scatter(xyz_vap[0],xyz_vap[1],xyz_vap[2],color='c')
    
    pos_int = pos[int_in]
    xyz_int = np.vstack([pos_int[::, 0], pos_int[::, 1], pos_int[::, 2]])
    ax1.scatter(xyz_int[0],xyz_int[1],xyz_int[2],color='k')

    
    
    pts = np.array(list(product([0,xlim], [0,ylim], [0,zlim])))    
    for s, e in combinations(pts, 2):
        if np.sum(np.abs(s-e)) in (xlim,ylim,zlim):
            ax1.plot3D(*zip(s, e), 'k-',color=box_color,linewidth=1)     
    
    
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.set_zlabel("y")

    plt.xlim([0,xlim])
    plt.ylim([0,ylim])
#    plt.ylim([0,ylim])

    ax1.set_xlim([0,xlim])
    ax1.set_ylim([0,ylim])
    ax1.set_zlim([0,zlim])
    
    # ax1.set_aspect('equal') 
    set_axes_equal(ax1)
    ax1.view_init(0, 10)
    plt.tight_layout()
    plt.show()

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),])
#    print(limits)
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


#plot_interfaces(inter,liq_cutoff,vap_cutoff,liq_in,vap_in,int_in)