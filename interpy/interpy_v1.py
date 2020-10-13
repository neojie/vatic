#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:58:18 2019

@author: jiedeng
"""
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pytim
from vatic.interpy.GDS import fit_gds_double_sided
from vatic.interpy.write_pdb import write_pdb
import os
from vatic.interpy.PlotInter import plot_interfaces
def build_universe(begin, end = None, center = True):
    """
    build universe snapshot for MDAnalysis
    Args:
        begin = 1 means first frame
    """
    print(os.path.dirname(os.path.realpath('__file__')))
    file = os.path.dirname(os.path.realpath('__file__')) + '/XDATCAR'
    if center:
        print('******************* Mass center fixed *******************')
        write_pdb(0,1,file,center=False)  # the first frame does not need centering
        u_dummy = mda.Universe('XDATCAR_0_1.pdb')
        masses  = u_dummy.atoms.masses
        center0 = masses.dot(u_dummy.atoms.positions)/sum(masses)
        print("in build universe, center0 is,",center0)
        print("sum masses is,",sum(masses))
        print("masses.dot(u_dummy.atoms.positions) is,",masses.dot(u_dummy.atoms.positions))
        write_pdb(begin,end,file,center=True, center0 = center0, masses = masses)
    else:
        print('******************* Mass center NOT-fixed *******************')
        write_pdb(begin,end,file,center=False)
    if end is None:
        name = 'XDATCAR'+'_'+str(begin)
    elif end is False:
        name = 'XDATCAR'+'_'+str(begin)+'_'+'end'
    else:
        name = 'XDATCAR'+'_'+str(begin)+'_'+str(end)
    u = mda.Universe(name + '.pdb')
    return u

def cal_proximity(inter):
    """
    calculate proximity
    """
    pos   = inter.pos    
    ### calculate distance    
    verts, faces, normals = inter.triangulated_surface  
    proximity = np.zeros(len(pos))
    tag = np.zeros(len(pos))  
    for num, point in enumerate(pos):
        RS = (point - verts) # R to surface vector
        tmp = RS**2
        dis = np.sum(tmp,axis=1)**.5
        min_ind = np.argmin(np.abs(dis))      # min distance index for the vertex
        min_nor = normals[min_ind]            # normal vector corresponding to min distance
        min_dis = dis[min_ind]
        tmp2    = np.sum(min_nor*RS[min_ind])
        min_dis = -tmp2/np.abs(tmp2)*min_dis # same direction, out of the interface
        proximity[num] = min_dis              # based on test
        if normals[min_ind,2]>0:
            tag[num] = 1                #upper interface
        else:
            tag[num] = -1
    return proximity,tag


def plot_temporal_inter(i=1, mesh=1.5, alpha=2.5,level=None):
    """
    plot interface for ith step
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ## plot ith 
    u = build_universe(i)    
    inter     = pytim.WillardChandler(u, mesh=mesh, alpha=alpha,level=level)
    triangles = inter.triangulated_surface[0][inter.triangulated_surface[1]]
    ###
    pos   = inter.pos
    ### calculate distance
    verts, faces, normals = inter.triangulated_surface   
    # calculate proximity
    prox, tag = cal_proximity(inter)
    
    liq_in = prox>0
    vap_in = prox<0
        
    fig = plt.figure(figsize=(4, 5))
    ax1 = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh1 = Poly3DCollection(triangles)
    mesh1.set_edgecolor('none')
    mesh1.set_alpha(0.3)
    ax1.add_collection3d(mesh1)
    
    pos_liq = pos[liq_in]
    xyz_liq = np.vstack([pos_liq[::, 0], pos_liq[::, 1], pos_liq[::, 2]])
    
    ax1.scatter(xyz_liq[0],xyz_liq[1],xyz_liq[2],color='r')
    
    pos_vap = pos[vap_in]
    xyz_vap = np.vstack([pos_vap[::, 0], pos_vap[::, 1], pos_vap[::, 2]])
    
    ax1.scatter(xyz_vap[0],xyz_vap[1],xyz_vap[2],color='k')
    ax1.view_init(0, 135)
    
    ax1.set_xlabel("x-axis: a")
    ax1.set_ylabel("y-axis: b")
    ax1.set_zlabel("z-axis: c")
   
    plt.tight_layout()
    plt.show()
    return inter,prox,tag


def number_density_z_coarse(inter):
    """
    Calculate number density profile along z with coarse graining
    """
    dens_re = inter.volume # equal to number_density_field_reshape
    tmp     = np.sum(dens_re,axis=0)
    tmp2    = np.sum(tmp,axis=0) #un-normalized density along z
    volume  = inter.box[0]*inter.box[1]*inter.box[2] #A^3  
    volume0 = volume/inter.ngrid[-1]
    rho_av  = inter.universe.atoms.n_atoms/volume    
    norm_factor = inter.universe.atoms.n_atoms/volume0/tmp2.sum()
    return tmp2*norm_factor,rho_av


def mass_density_proximity(begin=1,end=False,mesh=1.5, alpha=2.5,plot=True,level=None):
    """
    calculate average (begin - end) mass density profile vs. p(roximity)

    """       
    write_pdb(0,1,file='XDATCAR',center=False)
    u  =  mda.Universe('XDATCAR_0_1.pdb')
    ux = build_universe(begin,end) 
    nsw = len(ux.trajectory)

    zp = np.linspace(-u.coord.dimensions.max()/4,u.coord.dimensions.max()/4,80)
    dz = zp[1]-zp[0]
    L  = u.coord.dimensions[0] # assume 0,1 are the shorter edges
    rho_zp = np.zeros_like(zp)
    
    for i in range(nsw):
        u.atoms.positions =  ux.trajectory[i].positions
        inter     = pytim.WillardChandler(u, mesh=mesh, alpha=alpha,level=level)
        prox, tag = cal_proximity(inter)
        for j in range(len(zp)-1):
            ind_j     = [ind_tmp for ind_tmp, item in enumerate(prox) if (zp[j]<item) and (item<zp[j+1])]
            rho_j     = sum(u.atoms.masses[ind_j])/dz/(L**2) #g/mol/A^3
            rho_zp[j] = (rho_zp[j]*i + rho_j/6.022e23*1e30*1e-6/2)/(i+1)
    if plot:
        plot_rho(zp,rho_zp,xlabel=r'Prximity $(\AA)$')  
    return rho_zp,zp

def mass_density_z_no_coarse(begin=1,end=False,plot=True):
    """
    Obsolete, use the coarsen version!!
    calculate average (begin - end) mass density profile vs. z without coarse graining 

    """
    write_pdb(0,1,file='XDATCAR1',center=False)
    u  =  mda.Universe('XDATCAR1.pdb')
    ux = build_universe(begin,end) 
    nsw = len(ux.trajectory)
    
    zz = np.linspace(0,u.coord.dimensions.max(),200)
    dz = zz[1]-zz[0]
    L  = u.coord.dimensions[0]
    rho_zz = np.zeros_like(zz)
    for i in range(nsw):
        pos = ux.trajectory[i].positions   
        for j in range(len(zz)-1):        
            ind_j = np.logical_and(zz[j]<pos[:,-1],  pos[:,-1]<zz[j+1])  
            rho_j = sum(u.atoms.masses[ind_j])/dz/(L**2) #g/mol/A^3
            rho_zz[j] = rho_j/6.022e23*1e30*1e-6
    if plot:
        plot_rho(zz,rho_zz)
    return rho_zz,zz
    
def temporal_number_density_z_coarse(i=1, mesh=1.5, alpha=2.5,plot=True,level=None):
    """
    plot coarse graing mass density profile file of ith step
    """
    u = build_universe(i)
    inter             = pytim.WillardChandler(u, mesh=mesh, alpha=alpha,level=level)
    rho_zi,rho_av     = number_density_z_coarse(inter)
    zi = np.linspace(0,inter.box[-1],inter.ngrid[-1])
    if plot:
        plot_rho(zi,rho_zi,ylabel = r'$1/\AA^{3}$')
    return rho_zi,zi,rho_av

def mass_density_z_coarse(inter):
    """
    Calculate mass density profile along z with coarse graining
    """
    NA      = 6.022e23
    dens_re = inter.mass_density_field_reshape
    tmp     = np.sum(dens_re,axis=0)
    tmp2    = np.sum(tmp,axis=0) #un-normalized density along z
    volume  = inter.box[0]*inter.box[1]*inter.box[2]*NA*1e-24
    volume0 = volume/inter.ngrid[-1]
    rho_av  = sum(inter.universe.atoms.masses)/volume    
    norm_factor = sum(inter.universe.atoms.masses)/volume0/tmp2.sum()
    return tmp2*norm_factor,rho_av

def temporal_mass_density_z_coarse(i=1, mesh=1.5, alpha=2.5,plot=True,level=None):
    """
    plot coarse graining mass density profile file of ith step
    """
    u = build_universe(i)
    inter             = pytim.WillardChandler(u, mesh=mesh, alpha=alpha,level=level)
    rho_zi,rho_av     = mass_density_z_coarse(inter)
    zi = np.linspace(0,inter.box[-1],inter.ngrid[-1])
    if plot:
        plot_rho(zi,rho_zi)
    return rho_zi,zi,rho_av

def plot_rho(z,rho,xlabel = r'$z (\AA)$', ylabel = r'$\rho (g/cm^{3})$' ):
    """
    plot density
    """
    fig,ax = plt.subplots(1,1)
    ax.plot(z,rho)
    ax.set_xlabel(xlabel,fontsize =13)
    ax.set_ylabel(ylabel,fontsize =13)
    fig.show()

#########build companion file#########
def build_fit_prox(begin=1,end=False, mesh=1.5, alpha=2.5,file='XDATCAR',level=None):
    """
    build two file
    calculate average (begin - end) mass density profile vs. p(roximity)

    """       
    import pandas as pd
#    file = os.path.dirname(os.path.realpath('__file__')) + '/XDATCAR1.pdb' 
    write_pdb(0,1,file=file,center=False)
    u  = mda.Universe('XDATCAR_0_1.pdb')
    ux = build_universe(begin,end) 
    nsw = len(ux.trajectory)
    
    for i in range(nsw):
        u.atoms.positions =  ux.trajectory[i].positions
        inter     = pytim.WillardChandler(u, mesh=mesh, alpha=alpha,level=level)
        prox, tag = cal_proximity(inter)
        rho_zi,_  = mass_density_z_coarse(inter)
        zi        = np.linspace(0,inter.box[-1],inter.ngrid[-1])
        result    = fit_gds_double_sided(zi,rho_zi,plot=False,verbose=False)
        if i == 0:
            pd_fit  = pd.DataFrame(data = result.best_values,index=[begin+i])
            pd_prox = pd.DataFrame(data = prox,index=len(prox)*[begin+i],columns=['prox'])
        else:
            tmp_fit  = pd.DataFrame(data = result.best_values,index=[begin+i])
            tmp_prox = pd.DataFrame(data = prox,index=len(prox)*[begin+i],columns=['prox'])
            pd_fit   = pd_fit.append(tmp_fit)
            pd_prox  = pd_prox.append(tmp_prox)
    pd_fit.to_csv('fit'+'_'+str(begin)+'_'+str(end),header=True,index=True, sep='\t', mode='a')
    pd_prox.to_csv('prox'+'_'+str(begin)+'_'+str(end),header=True,index=True, sep='\t', mode='a')

##################### time average mass density profile ###########################################
import copy
import scipy.spatial as ss

def _add_points(vert,i,mins,maxs,inter):
    """
    subroutine for mass_density_z_coarse_cont
    """
    tmp = vert[i]
    if vert[i] == mins[i]:
        tmp = 0
    elif vert[i] == maxs[i]:
        if vert[i] + mins[i] <= inter.box[i]:
            tmp = vert[i] + mins[i]
        else:
            tmp = inter.box[i]
    return tmp

def _extend_verts(verts, inter):
    """
    subroutine for mass_density_z_coarse_cont
    When mesh is large, verts may not cover the whole space
    """
    verts_modified = copy.deepcopy(verts)
    mins = [min(verts[:,0]),min(verts[:,1])]
    maxs = [max(verts[:,0]),max(verts[:,1])]
    for vert in verts:
        tmpx = _add_points(vert,0,mins,maxs,inter)
        tmpy = _add_points(vert,1,maxs,maxs,inter)
        tmpz = vert[2]
        new  = [tmpx,tmpy,tmpz]
        if not np.array_equal(vert,new):
            verts_modified = np.append(verts_modified,np.array([new]),axis=0)
    return verts_modified

def mass_density_z_coarse_cont(begin=1,end=False,liq_cutoff = 0, vap_cutoff = 0,
                               mesh=1.5, alpha=2.5,level=None,plot=False,plot3d=False):
    """
    calcualte the time average grain coarsened mass density along z
    level defined as the prox = 0 
    liq defined as prox < liq_cutoff
    vap defined as prox > vap_cutoff
    
    Params
    ------
    begin : int
    end : int
    liq_cutoff : float, cutoff for proximity, liquid is defined as > liq_cutoff
    vap_cutoff : float, cutoff for proximity, liquid is defined as > liq_cutoff
    mesh : float
    alpha : float, Gaussian width
    level : contour cutoff
    plot : plot the results
    
    Returns
    ------  
    rho_z : time average density profile
    rho : [rho_l, rho_i, rho_v]
    zi : correpsonding z axis
    """
    if liq_cutoff < 0:
        print('********* liquid proximity cutoff is INCORRECT, must >=0 *********')
    if vap_cutoff > 0:
        print('********* vapor  proximity cutoff is INCORRECT, must <=0 *********')
        
    write_pdb(0,1,file='XDATCAR',center=False)
    u   =  mda.Universe('XDATCAR_0_1.pdb')
    ux  = build_universe(begin,end) 
    nsw = len(ux.trajectory)

    rho_z = 0
    rho_l = []
    rho_v = []
    rho_i = []
    volume = u.dimensions[0]*u.dimensions[1]*u.dimensions[2]
    NA      = 6.022e23    
    dummy = ux
    for i in range(nsw):
        dummy.atoms.positions = ux.trajectory[i].positions
        u.atoms.positions     =  ux.trajectory[i].positions
        inter         = pytim.WillardChandler(u, mesh=mesh, alpha=alpha,level=level)
        rho_zi,rho_av = mass_density_z_coarse(inter)
        rho_z  = (rho_z*i + rho_zi)/(i+1)
        
        ### extract interface triangles
        verts, faces, normals = inter.triangulated_surface   
        # calculate proximity
        prox, tag = cal_proximity(inter)
        
        liq_in = prox > liq_cutoff
        vap_in = prox < vap_cutoff
        int_in = np.logical_not(np.logical_or(liq_in,vap_in))
    
        verts_modified = _extend_verts(verts, inter)
        hull_modified  = ss.ConvexHull(verts_modified)
        
        liq_vol = hull_modified.volume 
        int_vol = inter.surface_area*(liq_cutoff - vap_cutoff)
        vap_vol = volume - (liq_vol + int_vol)
        
        liq_den = sum(inter.universe.atoms.masses[liq_in])/(liq_vol*NA*1e-24)
        vap_den = sum(inter.universe.atoms.masses[vap_in])/(vap_vol*NA*1e-24)
        
        if plot3d and (i==nsw-1): # only plot the last frame if many present
            plot_interfaces(inter,liq_cutoff,vap_cutoff,liq_in,vap_in,int_in)
        
        if int_vol > 0:
            vap_liq = sum(inter.universe.atoms.masses[int_in])/(int_vol*NA*1e-24)
            rho_i.append(vap_liq)
        
        rho_l.append(liq_den)
        rho_v.append(vap_den)   
        
    zi = np.linspace(0,inter.box[-1],inter.ngrid[-1])             
    if plot:   # plot profiles along z        
        fig, ax = plt.subplots(1,2,figsize=(7,3),sharey=True)
        ax[0].plot(zi, rho_z);ax[0].set_xlim([zi[0],zi[-1]])
        ax[0].plot(zi, np.ones_like(zi)*np.array(rho_l).mean(),'k-')
        ax[0].plot(zi, np.ones_like(zi)*np.array(rho_v).mean(),'k--')

        ax[0].set_xlabel(r'$z (\AA)$'); ax[0].set_ylabel(r'$\rho (g/cm^{3})$')
        ax[1].plot(np.array(range(nsw)) + int(begin), rho_l, 'k-')
        ax[1].plot(np.array(range(nsw)) + int(begin), rho_v, 'k--')
        ax[1].set_xlabel('NSW'); ax[1].set_xlim([begin,end])
        ax[1].annotate('liq is prox >' + str(liq_cutoff) , xy=(0.3,0.6), xycoords='axes fraction')
        ax[1].annotate('vap is prox <' + str(vap_cutoff) , xy=(0.3,0.1), xycoords='axes fraction')
        if rho_i != []:
            ax[0].plot(zi,np.ones_like(zi)*np.array(rho_i).mean(),'k:')
            ax[1].plot(np.array(range(nsw))+begin,rho_i,'k:')
            ax[1].annotate('intreface is prox in between', xy=(0.1,0.3), xycoords='axes fraction')
    if rho_i != []:
        rho = np.array([rho_l,rho_i,rho_v])
    else:
        rho = np.array([rho_l,[0]*len(rho_l),rho_v])
    return rho_z,rho,zi


#build_fit_prox(1,5)
#########Structure Analysis#########
def structure_factor():
    """
    calculate structure factor  
    ref. equation (1) by Xiao and Stixrude 2018
    """
    
    
def cn():
    """
    calculate coordinaiton number 
    ref. equation (3) by Xiao and Stixrude 2018
    """    
#########Wegner Expansion#########
def wegner_expansion():
    """
    caption of figure 3
    """
    
#TODO
   #1)pressure tensor 
   
if __name__ == "__main__":
    ######## build fit and prox ######
    build_fit_prox(9000,9100)
#    pd.read_csv('')
#    rho_zp,zp = mass_density_proximity(9000,level=9)
    ## convert 1st frame of XDATCAR file to XDATCAR.pdb
#    write_pdb(0,1,name='XDATCAR1')
#    u  =  mda.Universe('XDATCAR1.pdb')
#    
#    ## convert all XDATCAR file to XDATCAR.pdb and delete 1st frame PBD file
#    #build_universe(0,name='XDATCAR')
#    
#    ## single frame nalaysis
#    i = 5000
#    rho_zi,zi,rho_av = temporal_mass_density_z_coarse(i)
##    rho_zi_num,zi,rho_av_num = temporal_number_density_z_coarse(i)
###    plt.figure()
###    plt.plot(zi,rho_zi,label='mass')
###    plt.plot(zi,rho_zi_num*rho_av/rho_av_num,label='number')
###    rho_zi - rho_zi_num*rho_av/rho_av_num
#    inter,prox,tag  = plot_temporal_inter(i) 
###    # fit_gds
##    result = fit_gds(zi,rho_zi,plot=True)
###    plt.figure()
###    plt.plot(inter.pos[:,-1],prox,'o')
##    # fit_gds2
#    result = fit_gds_double_sided(zi,rho_zi,plot=True)
#    mass_density_proximity(i)
##    plt.figure()
##    plt.plot(inter.pos[:,-1],pro,'o')
##    
##    ## radial distribution function for selected frames
#    ux = build_universe(500,900)
#    g1= ux.select_atoms('type O')
#    g2= ux.select_atoms('type Mg')
#    rdf = mda.analysis.rdf.InterRDF(g1,g2,nbins=75, range=(0.0, min(ux.dimensions[:3])/2.0))
###               
#    rdf.run()
##    
#    fig = plt.figure(figsize=(5,4))
#    ax = fig.add_subplot(111)
#    ax.plot(rdf.bins, rdf.rdf, 'k-',  label="rdf")    
#    ax.legend(loc="best")
#    ax.set_xlabel(r"Distance ($\AA$)")
#    ax.set_ylabel(r"RDF")
##
###write_pdb(800,801)
##import mda.analysis.rdf as rdf
###
###mda.analysis.rdf.InterRDF
##plt.plot(zi,gds2(zi,w=1.5,rhol=2,rhov=0.2,z0=40,z1=80))
##plt.grid(True)
##
##rho_zp,zp = mass_density_proximity(2000,level=9)
##
##inter     = pytim.WillardChandler(u, mesh=1.5, alpha=2.5,level=None)
#ux.atoms.atoms
#ux.atoms.positions
#ux.atoms.names
#cal_proximity(inter)
#inter.pos