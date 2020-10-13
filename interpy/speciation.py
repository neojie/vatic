#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 09:18:40 2019
Find Nearest Neighbor
TODO
0 - Mutual neareast neighor -> done
1 - TO DO plot with color, 2D projection with color coded element
2 - make extract_species_vesta to show the relative number as well -> done
3 - color graph plot by elemetns
https://stackoverflow.com/questions/13517614/draw-different-color-for-nodes-in-networkx-based-on-their-node-value
@author: jiedeng
"""

import matplotlib.pyplot as plt
import numpy as np
from vatic.others.periodic_kdtree.periodic_kdtree import PeriodicCKDTree
#import vatic.interpy.interpy_v1 as ity
import re
import networkx as nx

def extract_vapor(u_i,prox_i,cutoff_prox,cutoffs_rdf,show_vapor=True):
    """
    calculate coordination of pair ele1-ele2
    
    Parameters
    ----------
    u : MDAnalysis universe
    i : ith frame
    cutoffs_rdf : dict, key is piar, value is float, cutoffs of pairs, should be the 1st min of RDF
    cutoff_prox : float, cutoff for proximity
    prox_file   : str, file name of the proximity, 
                  generated with ity.build_fit_prox(begin,end,'XDATCAR',level=9)
    show_vapor : boolean, plot vapor?
    
    Return
    ----------
    pairs : map of pairs of elements in vapor only, for N elements, N^2 pairs
    eles_num : map of elements, key - elements of all (not only for vapor), value - # of atoms
    subgroups : map of elements of universe for vapor
    u_g: universe, vapor universe
    
    Examples
    ----------
    from vatic.interpy.write_pdb import write_pdb
    import MDAnalysis as mda
    from vatic.interpy.rdf_cnp import cal_cn, cal_rdf

    cutoffs_rdf = {'MgMg':2,'SiSi':2,'OO':2,'MgO':2,'OMg':2,'SiO':2,'OSi':2,'MgSi':2,'SiMg':2}    
    
    Notes
    ----------    
    tested against StrucAna for the example run    
    """

#    prox      = pd.read_csv(prox_file,index_col=[0],sep='\t')    
#    prox_i    = prox.loc[i]
#    u_i       = ity.build_universe(i)
    box       = u_i.dimensions[:3]

#    u_g       = u_i.atoms[prox_i.values[:,0]<cutoff_prox]   # for prox botain from pandas
    
    u_g       = u_i.atoms[prox_i<cutoff_prox]   # vapor phase atom group
    eles      = []
    subgroups = {}  ## atom subgroups
    eles_num  = {}  ## similiar to subgroups, but the # of atoms only and also include the 0 ones
    pairs     = {}  ## rdf pairs
    
    ## build subgroups and eles list
    if u_g.n_atoms > 0:
        for i in u_g.atoms.names:
            if i not in eles:
                eles.append(i)
                subgroups[i] = u_g.select_atoms('type ' + i)      
                eles_num[i]  = subgroups[i].atoms.n_atoms
    for i in u_i.atoms.names:
        if i not in subgroups:
            eles_num[i] = 0            
    
    for i in eles:
        gi = subgroups[i]
        for j in eles:
            gj = subgroups[j]
            gj_tree = PeriodicCKDTree(box, gj.atoms.positions)
            gi_list = gj_tree.query_ball_point(gi.atoms.positions,cutoffs_rdf[i+j])
            pairs[i+j] = gi_list
#            print('ele1 is', gi.atoms.names)
#            print('ele2 is', gj.atoms.names)
#            print(gi_list)
    if show_vapor:
        show_species(subgroups)
    return pairs,eles_num,subgroups,u_g

def extract_species(u_g,eles_num,pairs,species_count = {},draw_species=False):
    """
    extract species
    
    Refs
    --------
    What I used
    1. NetworkX
        https://networkx.github.io/documentation/stable/reference/algorithms/coloring.html

    I did not use cluster algos, but they may be useful for immiscibility study
    some refs I searched are 
    0. https://en.wikipedia.org/wiki/Category:Cluster_analysis
       => cluster analysis!!
    1. https://molml.readthedocs.io/en/latest/molml.html#module-molml
    2. https://www.cmm.ki.si/~FAMNIT-knjiga/wwwANG/The_Adjancency_Matrix-4.htm   
       => ACM atom connectivity matrices
       https://www.cmm.ki.si/~FAMNIT-knjiga/wwwANG/  
       => group theory in chemistry
    3. https://wiki.fysik.dtu.dk/ase/ase/data.html 
       => ASE
    4. https://zh.wikipedia.org/wiki/%E9%9F%A6%E5%B0%94%E8%8E%B1%E8%A1%A8
       https://hoomd-blue.readthedocs.io/en/stable/nlist.html#lbvh-tree
       cell list vs. Verlet list, import for MD simualtions
    5. https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
       scipy implementation
    6. https://11011110.github.io/blog/2019/02/21/mutual-nearest-neighbors.html
    7. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
       => sklearn, k neighors
    """
    atoms_list = u_g.names
    atoms = []
    atoms_dummy = []
    count = 0
    for ele in atoms_list:
        if ele not in atoms_dummy:
            count = 0
        atoms.append(ele+str(count))
        atoms_dummy.append(ele)
        count += 1
                
    G = nx.Graph()
    G.add_nodes_from(atoms)
        
    eles = eles_num.keys()
            
    for i in eles:
        for j in eles:
            ele_ele = i+j
            if ele_ele in pairs:  # it is possible that ele_ele is not pairs as pairs are for vapor phase only
                for n,pair in enumerate(pairs[ele_ele]):
                    if i == j:    ## same element pair
                        try:
                            pair.remove(n)   ## KDTree does not work well when atom to atom itself, because it generates 0 distance
                        except:
                            pass
                        if len(pair) >= 1:
                            G.add_edges_from(generate_edges(i,j,n,pair))
                    else:        ## different element pair
                        if len(pair)>0:
                            G.add_edges_from(generate_edges(i,j,n,pair))    
    if draw_species:                  
        nx.draw(G,with_labels=True)                
    
    species = sorted(nx.connected_components(G), key=len, reverse=True)
        
    for specie in species:   
        specie_index_removed = [re.sub('[^a-zA-Z]+', '', i) for i in specie]
        specie_index_removed.sort() # aftr sorting the MgOH and HMgO should equal
        specie_normal = ""
        for i in specie_index_removed:
            specie_normal += i
        if specie_normal in species_count:
            species_count[specie_normal] += 1
        else:
            species_count[specie_normal] = 1
    return species_count
    
def generate_edges(i,j,n,pair):
    """
    subroutine of extract_species
    """
    edges = []
    host  = i + str(n)
    for tmp in pair:
        guest = j + str(tmp)
        edges.append((host,guest))
    return edges    

def show_species(subgroups):
    """
    plot gas species
    
    Parameters
    ------------
    subgroups : map, key is ele, value is universe object    
    """
    fig = plt.figure(figsize=(4, 5))
    ax1 = fig.add_subplot(111, projection='3d')
    for ele in subgroups:
        gi = subgroups[ele]
        xyz_gi = np.vstack([gi.atoms.positions[::, 0], gi.atoms.positions[::, 1], gi.atoms.positions[::, 2]])    
        ax1.scatter(xyz_gi[0],xyz_gi[1],xyz_gi[2],label=ele)
        ax1.view_init(0, 45)
    
    ax1.set_xlabel("x-axis: a")
    ax1.set_ylabel("y-axis: b")
    ax1.set_zlabel("z-axis: c")
       
    plt.tight_layout()
    plt.legend()
    plt.show()


def make_dataframe(species_sum,begin,end,plot = True,write = True):
    """
    change species_sum to pandas dataframe
    
    Parameters
    --------
    species_sum : list, generated by extract_species
    begin : int, begin frame
    end   : int, end frame
    plot  : boolean, if true plot all the species 
    write : boolean, if true write to a csv file
            to open it use
           test = pd.read_csv('species_' + str(begin) + '_' + str(end),sep=',',index_col=[0])
    
    Returns
    --------
    df : pandas dataframe
    
    """
    import pandas as pd
    import copy
    # to avoid original file be affected, pass by name
    species_count_copied = copy.deepcopy(species_sum)    
    df                   = pd.DataFrame()
    for n, tmp in enumerate(species_count_copied):
        for key,value in tmp.items():
            tmp[key] = [value]
        df_tmp = pd.DataFrame.from_dict(tmp)
        df     = pd.concat([df,df_tmp],axis=0,sort=False).fillna(0) # order [df,df_tmp] matters!
            
    df.set_index([list(range(begin,end+1))],inplace=True)
    
    if plot:    
        for key in df:
            plt.plot(df.index,df[key],label=key)    
        plt.legend()

    if write:
        df.astype('int',inplace=True).to_csv('species_' + str(begin) + '_' + str(end),sep=',')
    return df


def extract_species_vesta(u_g,eles_num,pairs,species_count = {},draw_species=False):
    """
    extract species
    
    Refs
    --------
    What I used
    1. NetworkX
        https://networkx.github.io/documentation/stable/reference/algorithms/coloring.html

    I did not use cluster algos, but they may be useful for immiscibility study
    some refs I searched are 
    0. https://en.wikipedia.org/wiki/Category:Cluster_analysis
       => cluster analysis!!
    1. https://molml.readthedocs.io/en/latest/molml.html#module-molml
    2. https://www.cmm.ki.si/~FAMNIT-knjiga/wwwANG/The_Adjancency_Matrix-4.htm   
       => ACM atom connectivity matrices
       https://www.cmm.ki.si/~FAMNIT-knjiga/wwwANG/  
       => group theory in chemistry
    3. https://wiki.fysik.dtu.dk/ase/ase/data.html 
       => ASE
    4. https://zh.wikipedia.org/wiki/%E9%9F%A6%E5%B0%94%E8%8E%B1%E8%A1%A8
       https://hoomd-blue.readthedocs.io/en/stable/nlist.html#lbvh-tree
       cell list vs. Verlet list, import for MD simualtions
    5. https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
       scipy implementation
    6. https://11011110.github.io/blog/2019/02/21/mutual-nearest-neighbors.html
    7. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
       => sklearn, k neighors
    """
    ### build subgroups to make the signature the same as extract_species ###
    subgroups = {}  ## atom subgroups
    
    ## build subgroups and eles list
    if u_g.n_atoms > 0:
        for i in eles_num:
            if i not in subgroups:
                subgroups[i] = u_g.select_atoms('type ' + i)    
                
    atoms_list   = u_g.names
    atoms_indice = u_g.indices
    atoms = []
    for i in range(len(atoms_list)):
         atoms.append(atoms_list[i] + str(atoms_indice[i]+1))
                
    G = nx.Graph()
    G.add_nodes_from(atoms)
        
    eles = eles_num.keys()
            
    for i in eles:
        for j in eles:
            ele_ele = i+j
            if ele_ele in pairs:  # it is possible that ele_ele is not pairs as pairs are for vapor phase only
                for n,pair in enumerate(pairs[ele_ele]):
                    if i == j:    ## same element pair
                        try:
                            pair.remove(n)   ## KDTree does not work well when atom to atom itself, because it generates 0 distance
                        except:
                            pass
                        if len(pair) >= 1:
                            G.add_edges_from(generate_edges_vesta(i,j,n,pair,subgroups))
                    else:        ## different element pair
                        if len(pair)>0:
                            G.add_edges_from(generate_edges_vesta(i,j,n,pair,subgroups))    
    if draw_species:                  
        nx.draw(G,with_labels=True)                
    
    species = sorted(nx.connected_components(G), key=len, reverse=True)
        
    for specie in species:   
        specie_index_removed = [re.sub('[^a-zA-Z]+', '', i) for i in specie]
        specie_index_removed.sort() # aftr sorting the MgOH and HMgO should equal
        specie_normal = ""
        for i in specie_index_removed:
            specie_normal += i
        if specie_normal in species_count:
            species_count[specie_normal] += 1
        else:
            species_count[specie_normal] = 1
    return species_count


def generate_edges_vesta(i,j,n,pair,subgroups):
    """
    subroutine of extract_species
    """
    edges = []
    host = i + str(subgroups[i].indices[n]+1)
    for tmp in pair:
        guest = j + str(subgroups[j].indices[tmp] + 1)
#        print(guest)
        edges.append((host,guest))
    return edges 
  

##### post-analysis on speciation ####
def ele_stat_all_species(eles,df):
    """
    get a dict, with # of elements for each species, which can be used for ele_stat_all_elements
    
    Param
    ------
    eles : dict, key is all eles, value ele number
    df   : dataframe, 
           pd.read_csv('species_' + str(begin) + '_' + str(end),sep=',',index_col=[0])
    returns
    -------
    ele_stats : dictionary, key is all eles, value ele # stats for each species
    
    example
    -------
    ele_stats         = ele_stat_all_species({'H','O','Si','Mg'},df)

    """
    import copy
    ele_stat           = copy.deepcopy(df.iloc[0])
    # intialize 
    ele_stat.values[:] = 0
    ele_stats = {}
    validity_check(eles)
    for ele in eles:
        ele_stats[ele] = copy.deepcopy(ele_stat)
        for indice in df.columns:
            ele_stats[ele][indice] += indice.count(ele)
    return ele_stats

def ele_stat_all_elements(ele_stats,df):
    """
    get a dict with elements counts and a combined dataframe ele_pd
    
    Params
    ------
    ele_stats : dict, output of ele_stat_all_species
    df        : dataframe, 
    
    Returns
    ------
    ele_counts : dictionary, key is all eles, value atom # stats for each element
    ele_pd : combined dataframe based on ele_counts
    
    example
    -------
    ele_counts,ele_pd = ele_stat_all_elements(ele_stats,df)        
    df_sum  = df.join(ele_pd)

    """
    import pandas as pd
    ele_counts = {}
    ele_pd     = pd.DataFrame()
    for ele in ele_stats:
        tmp =ele_stats[ele]*df
        ele_counts[ele+'_sum'] = pd.DataFrame(tmp.sum(axis=1),columns=[ele+'_sum'])
        ele_pd = ele_counts[ele+'_sum'].join(ele_pd)
    
    return ele_counts,ele_pd
        

def validity_check(eles):
    """
    check if "ele_stat_all_species" works
    
    Notes
    ------
    not working, e.g., Hg vs. H; S vs. Si 

    """
    out = True
    # check this method #
    for ele1 in eles:
        for ele2 in eles:
            if ele1 != ele2:
               if (ele1 in ele2) or (ele2 in ele1):
                   print("**********Warning**********")
                   print("This method does not work for {0} and {1}".format(ele1,ele2))
                   out =  False
    return out          



#### deprecated old composition specific ways ####
    
#def mgsio(pairs,eles_num):
#    """
#    analyze the mgfeo system
#    """
#    species     = {'SiO2':0,'SiO':0, 'MgO':0, 'O3':0, 'O2':0, 'Mg':0, 'Si':0, 'O':0}
#    species_fra = {'SiO2':0,'SiO':0, 'MgO':0, 'O3':0, 'O2':0, 'Mg':0, 'Si':0, 'O':0}
#    if 'SiO' in pairs:
#        for pair in pairs['SiO']:
#            if len(pair) == 1:
#                species['SiO'] += 1
#            elif len(pair) == 2:
#                species['SiO2'] += 1
#            elif len(pair)>2:
#                print("SiO3+ species occur ?!")
#    
#    if 'MgO' in pairs:        
#        for pair in pairs['MgO']:
#            if len(pair) == 1:
#                species['MgO'] += 1
#            elif len(pair) > 1:
#                print("MgO2+ species occur ?!")
#            
#    if 'OO' in pairs:
#        for n, pair in enumerate(pairs['OO']):
#            if len(pair) == 2:
#                species['O2'] += 1
#            elif len(pair) == 3:
#                species['O3'] += 1
#            elif len(pair)>3:
#                print("O4+ species occur ?!")
#            
#    species['Si'] = eles_num['Si'] - (species['SiO2'] + species['SiO'])
#    species['Mg'] = eles_num['Mg'] - (species['MgO'])
#    species['O']  = eles_num['O']  - (species['SiO2']*2 + species['SiO'] + species['MgO'] )
#    
#    species_num = sum(species.values())
#    for key in species_fra.keys():
#        species_fra[key] = species[key]/species_num
#    return species,species_fra
#
#def mgsioh(pairs,eles_num):
#    """
#    analyze the mgfeoh system
#    """
#    species     = {'SiO2':0,'SiO':0, 'MgO':0, 'O3':0, 'O2':0, 'Mg':0, 'Si':0, 'O':0,
#                   'H2O':0, 'H2': 0, 'OH': 0, 'H': 0 }
#    species_fra = {'SiO2':0,'SiO':0, 'MgO':0, 'O3':0, 'O2':0, 'Mg':0, 'Si':0, 'O':0,
#                   'H2O':0, 'H2': 0, 'OH': 0, 'H': 0 }
#    if 'SiO' in pairs:
#        for pair in pairs['SiO']:
#            if len(pair) == 1:
#                species['SiO'] += 1
#            elif len(pair) == 2:
#                species['SiO2'] += 1
#            elif len(pair)>2:
#                print("SiO3+ species occur ?!")
#    
#    if 'MgO' in pairs:        
#        for pair in pairs['MgO']:
#            if len(pair) == 1:
#                species['MgO'] += 1
#            elif len(pair) > 1:
#                print("MgO2+ species occur ?!")
#            
#    if 'OO' in pairs:
#        for n, pair in enumerate(pairs['OO']):
#            if len(pair) == 2:
#                species['O2'] += 1
#            elif len(pair) == 3:
#                species['O3'] += 1
#            elif len(pair)>3:
#                print("O4+ species occur ?!")
#        species['O2'] /= 2  ### double counting
#        species['O3'] /= 3  ### trpple counting
#
#    if 'OH' in pairs:
#            for n, pair in enumerate(pairs['OH']):
#                if len(pair) == 2:
#                    species['H2O'] += 1
#                elif len(pair) == 1:
#                    species['OH'] += 1
#                elif len(pair)>2:
#                    print("OH3+ species occur ?!")  
#
#    if 'HH' in pairs:
#            for n, pair in enumerate(pairs['HH']):
#                if len(pair) == 2:
#                    species['H2'] += 1                  
#                elif len(pair)>2:
#                    print("H3 species occur ?!")  
#            species['H2'] /= 2   ## double counting  
#                    
#    species['Si'] = eles_num['Si'] - (species['SiO2'] + species['SiO'])
#    species['Mg'] = eles_num['Mg'] - (species['MgO'])
#    species['O']  = eles_num['O']  - (species['SiO2']*2 + species['SiO'] + species['MgO'] + species['OH'] + species['H2O'] )
#    species['H']  = eles_num['H']  - (species['OH'] + species['H2O']*2 +  species['H2']*2)
#    
#    species_num = sum(species.values())
#    for key in species_fra.keys():
#        species_fra[key] = species[key]/species_num
#    return species,species_fra

