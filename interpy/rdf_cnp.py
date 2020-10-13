#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:08:25 2019

@author: jiedeng
"""
#import MDAnalysis as mda
from MDAnalysis.analysis import rdf as RDF
import matplotlib.pyplot as plt
import numpy as np
#import vatic
#vatic.plot_tool.load_default_setting()

def cal_rdf(u,ele1,ele2,cutoff=None,nbins=75,plot=True):
    """
    calculate RDF
    
    Parameters
    ----------
    u : MDAnalysis universe
    ele1 : str, element 1
    ele2 : str, element 2
    cutoff : float, cutoff for RDF, 
             default, half of the box size
    Return
    ----------
    bins : array
    rdf  : array
    
    Examples
    ----------
    from vatic.interpy.write_pdb import write_pdb
    import MDAnalysis as mda
    from vatic.interpy.rdf_cnp import cal_cn, cal_rdf
    
    file = '/Users/jiedeng/GD/Computation/VESTA/VData/Bridgmanite/Pv+H/Pv_H_1to4_3k/homogeneous_run/XDATCAR'
    write_pdb(8000,9900,file=file)
    u = mda.Universe('XDATCAR_8000_9900.pdb')
    
    ele1 = 'H'
    ele2 = 'H'
    
    bins,rdf = cal_rdf(u,ele1,ele2)
    cn,cnp,bins = cal_cn(u,ele1,ele2,1.2)
    
    Notes
    ----------    
    tested against StrucAna for the example run
    For coding details, refer to 
    1) /Users/jiedeng/Google Drive/Learn/MDanalysis_learn/learn_rdf.py
    2) /Users/jiedeng/Google Drive/Learn/MDanalysis_learn/ckDTree_learn.py
    3) https://arxiv.org/pdf/1808.01826.pdf
    4) https://www.mdanalysis.org/docs/documentation_pages/analysis/rdf.html
    5) https://github.com/patvarilly/periodic_kdtree
    6) https://stackoverflow.com/questions/42397870/calculation-of-contact-coordination-number-with-periodic-boundary-conditions
    7) /Users/jiedeng/Google Drive/Computation/VESTA/VData/Bridgmanite/Pv+H/Pv_H_1to4_3k/homogeneous_run/rdf_cnp.py
    """
    e1 = u.select_atoms('type ' + ele1); 
    e2 = u.select_atoms('type ' + ele2); 
    if cutoff is None:
        cutoff = min(u.dimensions[:1])/2
    rdf_mda = RDF.InterRDF(e1,e2,nbins=75, range=(0.0, cutoff))
    rdf_mda.run()
    bins = rdf_mda.bins
    rdf  = rdf_mda.rdf
    if ele1 is ele2:
        bins[0] = bins[1] -(bins[2]-bins[1]) ## why the first element blocked?
        rdf[0]  = rdf[1]
    else:
        bins = rdf_mda.bins
        rdf  = rdf_mda.rdf
        
    if plot:
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        ax.plot(bins, rdf, 'k-',  label=ele1 + '-' + ele2)    
        ax.legend(loc="best")
        ax.set_xlabel(r"Distance ($\AA$)")
        ax.set_ylabel(r"RDF")
    return bins, rdf




def cal_cn(u,ele1,ele2,cutoff,max_coor = 10,print_result=True):
    """
    calculate coordination of pair ele1-ele2
    
    Parameters
    ----------
    u : MDAnalysis universe
    ele1 : str, element 1
    ele2 : str, element 2
    cutoff : float, cutoff, should be the 1st min of RDF
    max_coor : int, max coordination number
    
    Return
    ----------
    cn : array
    cnp  : array
    bins : array
    
    Examples
    ----------
    from vatic.interpy.write_pdb import write_pdb
    import MDAnalysis as mda
    from vatic.interpy.rdf_cnp import cal_cn, cal_rdf
    
    file = '/Users/jiedeng/GD/Computation/VESTA/VData/Bridgmanite/Pv+H/Pv_H_1to4_3k/homogeneous_run/XDATCAR'
    write_pdb(8000,9900,file=file)
    u = mda.Universe('XDATCAR_8000_9900.pdb')
    
    ele1 = 'H'
    ele2 = 'H'
    
    bins,rdf = cal_rdf(u,ele1,ele2)
    cn,cnp,bins = cal_cn(u,ele1,ele2,1.2)

    Notes
    ----------    
    tested against StrucAna for the example run    
    """
    from vatic.others.periodic_kdtree.periodic_kdtree import PeriodicCKDTree
    e1 = u.select_atoms('type ' + ele1); e1_indices = e1.indices; e1_len = len(e1)
    e2 = u.select_atoms('type ' + ele2); e2_indices = e2.indices
    box     = u.dimensions[:3]
    nframes = len(u.trajectory)
    bins    = np.array(list(range(max_coor)))
    cnp     = np.zeros(bins.shape)
    for i in range(nframes):
        count = i + 1
        pos1  = u.trajectory[i][e1_indices]
        pos2  = u.trajectory[i][e2_indices]
        e2_tree = PeriodicCKDTree(box, pos2)
        e1_list = e2_tree.query_ball_point(pos1,cutoff)
        e1_count = np.array([len(i) for i in e1_list])        #list of nearest neighbor for every point in pos1
        ## 0-1,1-2,0,9 total 9 bins, we need 10
        tmp_bins = np.concatenate((bins,[max_coor]))
        e1_perc  = np.histogram(e1_count,bins=tmp_bins)[0]/e1_len  #percentage
        cnp  = (cnp*i+e1_perc)/count
    cn = sum(cnp*bins)
    if print_result:
        row_format  = "{:<8.3}" * (len(cnp) + 1)
        head_format = "{:<8}" * (len(bins) + 1)
        dashed_lines = "------"
        print(dashed_lines*6 + 'cn = '+ "{0}".format(str(cn)) + dashed_lines*6)
        print(head_format.format("", *(bins)))
        print(row_format.format("", *(cnp*100)))
    return cn,cnp,bins


def bs_rdfs(eles,u,peak2trench = 1,plot=True,savedata=True,saveplot=True):
    """
    bash process rdf
    Params
    ------
    eles : list of string
    u    : universe
    peak2trench : peak to trench distance in Ang
    plot : invoke plot_rdfs function?
    
    Returns
    ------
    
    
    """
    Nele = len(eles)
    rdfs           = {}
    first_peaks    = {}
    first_trenches = {}
    
    # seeking for max and min points
    for i in range(Nele):
        for j in range(Nele):
            pair       = eles[i]+'-'+eles[j]
            bins,rdf   = cal_rdf(u,eles[i],eles[j],plot=False)
            dbins = bins[1] - bins[0]
            rdfs[pair] = [bins,rdf]
            peak2trench_end = np.argmax(rdf)+int(peak2trench/dbins+1)
            peak2trench_end = peak2trench_end if peak2trench_end<len(rdf) else len(rdf)
            peak2trench_ind = range(np.argmax(rdf), peak2trench_end)
            first_peak      = np.round(bins[np.argmax(rdf)], 2)
            first_trench    = np.round(bins[np.argmax(rdf) + np.argmin(rdf[peak2trench_ind])],2)
            first_peaks[pair]    = first_peak 
            first_trenches[pair] = first_trench 
    if plot:
        plot_rdfs(eles,rdfs,first_peaks,first_trenches,saveplot=saveplot)   
    if savedata:
        tmp = np.zeros((len(bins),len(rdfs.keys())*2))
        i = 0
        header = []
        for key in rdfs:
            header.append('bins')
            header.append(key)
            tmp[:,i]   =  rdfs[key][0]
            tmp[:,i+1] =  rdfs[key][1]
            i+=2
        np.savetxt('rdfs',tmp, header="    ".join(header))
        
    return rdfs,first_peaks,first_trenches


def plot_rdfs(eles,rdfs,first_peaks,first_trenches,saveplot=True):
    """
    subroutine of bs_rdfs
    bash plot rdfs with 1st peak and trench shown
    """
    Nele = int(len(rdfs)**.5)
    max_rdf = 0
    for i in range(Nele):
        for j in range(Nele):
            _, rdf = rdfs[eles[i]+'-'+eles[j]][0], rdfs[eles[i]+'-'+eles[j]][1] 
            tmp   = max(rdf)
            if max_rdf < tmp:
                max_rdf = tmp  
    fig,ax = plt.subplots(Nele,Nele,figsize=(10,10),sharex=True,sharey=True)
    for i in range(Nele):
        for j in range(Nele):
            pair = eles[i]+'-'+eles[j]
            bins, rdf = rdfs[pair][0], rdfs[pair][1] 
            ax[i,j].plot(bins,rdf,label = eles[i]+'-'+eles[j])
            ax[i,j].plot([first_peaks[pair],first_peaks[pair]],[0,max_rdf],'k--',label = str(first_peaks[pair]))
            ax[i,j].plot([first_trenches[pair],first_trenches[pair]],[0,max_rdf],'k:',label = str(first_trenches[pair]))
            if i == Nele-1:
                ax[i,j].set_xlabel(r"r ($\AA$)")
            if j == 0:
                ax[i,j].set_ylabel('g(r)')
    #        ax[i,j].annotate(np.round(bins[np.argmax(rdf)],2), xy=(0.1,0.6), xycoords='axes fraction')
            ax[i,j].legend()
    if saveplot:
        fig.savefig("rdfs.png",bbox_inches='tight')

        
from MDAnalysis.analysis.waterdynamics import MeanSquareDisplacement as MSD

def msd(ele, u, begin=0, timestep=1, msdstarting = 50, savedata = True, plot=True, saveplot=False):
    """
    calculate MSD of ele of u
    
    Parameters
    ----------
    u : MDAnalysis universe
    ele : string
    timestep : in fs
    begin : begin frame
    msdstarting : the starting step for diffusion analysis
    
    Return
    ----------
    msd : array
    
    Examples
    ----------


    Notes
    ----------    
    """
    length = len(u.trajectory)
    MSD_analysis = MSD(u,'name ' + ele, begin,length,length//2)
    MSD_analysis.run()
    _msd  = MSD_analysis.timeseries
    time = np.array(list((range(length//2))))*timestep
    
    ## diffusion ##
    p    = np.polyfit(time[msdstarting:],_msd[msdstarting:],1)
    D    = p[0]/6*1e-5  # A^2/fs => m^2/s
    
    if savedata:
        np.savetxt('msd_'+ele, np.vstack((time, _msd)).T, header="Time(fs)    MSD(A^2)")
    if plot:
        plot_msd(time,_msd,ele,p=p,saveplot=saveplot)

    return _msd, time, D

def bs_msd(eles,u,begin=0, timestep = 1, plot=True,savedata = True, saveplot=False):
  
    Nele = len(eles)
    msds = {}
    Ds   = {}
    
    # seeking for max and min points
    for i in range(Nele):
        print("MSD for {0}".format(eles[i]))
        ele       = eles[i]
        _msd, time, D = msd(ele, u, begin=begin, timestep=timestep, msdstarting = 50, savedata = False, plot=False, saveplot=False)
        msds[ele]   =  _msd
        Ds[ele]     = D
    msds['time']  = time

    if plot:
        plt.figure()
        for i in range(Nele):
            ele = eles[i]
            plt.loglog(time,msds[ele],label = ele +' '+ "{0:2.3} ".format(Ds[ele]) + r"$ (m^2/s)$")
        plt.xlabel("Time (fs)")
        plt.ylabel('MSD' + r"$ (\AA^2)$")  
        plt.legend()    
        if saveplot:
            plt.savefig('msds.png', bbox_inches='tight')
    if savedata:
        tmp = np.zeros((len(time),Nele+1))
        tmp[:,0] = time
        header = ['time']
        i = 1
        for key in msds:
            if key != 'time':
                header.append(key)
                tmp[:,i] =  msds[key]
                i += 1
        np.savetxt('msds',tmp,header="    ".join(header))
    return msds, Ds




def plot_msd(time,_msd,ele,p=None,saveplot=False):
    """
    plot MSD of ele of u
    
    Parameters
    ----------
    u : MDAnalysis universe
    ele : string
    timestep : in fs
    begin : begin frame
    
    Return
    ----------
    msd : array
    
    Examples
    ----------


    Notes
    ----------    
    """    
    plt.figure()
    plt.loglog(time,_msd)
    plt.loglog(time,np.poly1d(p)(time),'--')
    plt.xlabel("Time (fs)")
    plt.ylabel('MSD' + r"$ (\AA^2)$")
    if saveplot:
        plt.savefig('msd_'+ ele + '.png',bbox_inches='tight')