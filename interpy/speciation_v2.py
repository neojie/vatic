#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:11:37 2019
Find Nearest Neighbor

@author: jiedeng
"""
from scipy.stats import rankdata
from vatic.others.periodic_kdtree.periodic_kdtree import PeriodicCKDTree
import numpy as np
import networkx as nx
import re
import matplotlib.pyplot as plt


def get_gas(u_i,prox_i,cutoff_prox):
    """
    extract gas 
    
    Parameters
    ----------
    u_i : MDAnalysis universe of ith frame
    prox_i : proximity array
                  generated with ity.build_fit_prox(begin,end,'XDATCAR',level=9)
    cutoff_prox : cutoff for proximity
    show_vapor : boolean, plot vapor?
    
    Return
    ----------
    u_g: universe, vapor universe
    
    Examples
    ----------
    from vatic.interpy.write_pdb import write_pdb
    import MDAnalysis as mda
    from vatic.interpy.rdf_cnp import cal_cn, cal_rdf

    
    Notes
    ----------    
    tested against StrucAna for the example run    
    """
    u_g       = u_i.atoms[prox_i<cutoff_prox]   # vapor phase atom group
    return u_g

def get_liq(u_i, prox_i, cutoff_prox):
    """
    extract liquid

    Parameters
    ----------
    u_i : MDAnalysis universe of ith frame
    prox_i : proximity array
                  generated with ity.build_fit_prox(begin,end,'XDATCAR',level=9)
    cutoff_prox : cutoff for proximity
    show_vapor : boolean, plot vapor?

    Return
    ----------
    u_g: universe, vapor universe

    Examples
    ----------
    from vatic.interpy.write_pdb import write_pdb
    import MDAnalysis as mda
    from vatic.interpy.rdf_cnp import cal_cn, cal_rdf


    Notes
    ----------
    tested against StrucAna for the example run
    """
    u_l       = u_i.atoms[prox_i > cutoff_prox]   # vapor phase atom group
    return u_l
    
def analyze_gas(u_g,cutoffs_rdf,show_gas=False):
    """
    analyze gas to find pairs, # of atoms, and subgroups 
    
    Parameters
    ----------
    u_g : MDAnalysis universe for vapor
    cutoffs_rdf : dict, key is piar, value is float, cutoffs of pairs, should be the 1st min of RDF
    show_gas : boolean, plot vapor?
    
    Return
    ----------
    pairs : map of pairs of elements in vapor only, for N elements, N^2 pairs
    eles_num : map of elements, key - elements of all (not only for vapor), value - # of atoms
    subgroups : map of elements of universe for vapor
    
    Examples
    ----------
    begin = 5000; end = 10000
    u  =  mda.Universe('XDATCAR_0_1.pdb')
    ux = ity.build_universe(begin,end) 
    nsw = len(ux.trajectory)
    
    #cutoffs_rdf = {'MgMg':2,'SiSi':2, 'OO':2,  'MgO':2, 'OMg':2,'SiO':2,'OSi':2,
    #               'MgSi':2,'SiMg':2, 'MgH':2, 'HMg':2, 'SiH':2,'HSi':2, 
    #               'OH'  :1.5, 'HO' :1.5, 'HH':1}   
    
    cutoffs_rdf = first_peaks
    cutoff_prox = -1.5
    
    species_sum_abs =[]
    
    for i in range(nsw):
        u.atoms.positions =  ux.trajectory[i].positions
        inter     = pytim.WillardChandler(u, mesh=1.5, alpha=2.5,level=None)
        prox, tag = ity.cal_proximity(inter)
        u_g = get_gas(u,prox,cutoff_prox = cutoff_prox)
    
        pairs, eles_num, subgroups                      = analyze_gas(u_g,cutoffs_rdf = cutoffs_rdf,show_gas=False)
        atoms_abs_all, atoms_abs_gas, atoms_vesta       = get_atom_index_gas(u,u_g)
        species_abs = count_species_abs(u_g, pairs, index = atoms_abs_all, draw_species=False)
        species_sum_abs.append(species_abs) ## len of species_sum_abs is the same as nsw 
    
    Notes
    ---------- 
    XXX
    """
    box       = u_g.dimensions[:3]
    
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
                
    for i in u_g.atoms.names:
        if i not in subgroups:
            eles_num[i] = 0            
    
    for i in eles:
        gi = subgroups[i]
        for j in eles:
            gj = subgroups[j]
            gj_tree = PeriodicCKDTree(box, gj.atoms.positions)
            gi_list = gj_tree.query_ball_point(gi.atoms.positions,cutoffs_rdf[i+'-'+j])
            pairs[i+'-'+j] = gi_list
    if show_gas:
        show_species(subgroups)
    return pairs,eles_num,subgroups

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


########## Rountine below summerize what lies below##########
def count_species(u_g,pairs,index = None,draw_species=False):
    """
    may be deprecated?  
    
    Params
    -----
    u_g :  gas uninverse
    pairs : pairs
    index : if None, atoms represented by numbers
    draw_species : draw species
    
    Return
    -----
    
    Example
    -----
    
    Note
    -----
    Equal to
    g_eles_new     = get_count_ind(u_g)
    pairs_reshaped = reshape_pairs(pairs,g_eles_new)
    nn             = build_nn(u_g,pairs_reshaped,g_eles_new)
    
    G = build_graph_from_nn(nn,index=atoms_vesta)
    
    species_count = speciation_from_G(G,draw_species=True)
    """    
    g_eles_new     = get_count_ind(u_g)
    pairs_reshaped = reshape_pairs(pairs,g_eles_new)
    nn             = build_nn(u_g,pairs_reshaped,g_eles_new)
#    print(nn)
    G = build_graph_from_nn(nn,index=index)
#    nx.draw(G)
    species_count = speciation_from_G(G,draw_species=draw_species)

    return species_count

def count_species_abs(u_g,pairs,index = None,draw_species=False):
    """
    may be deprecated? 
    
    Params
    -----
    u_g :  gas uninverse
    pairs : pairs
    index : if None, atoms represented by numbers
    draw_species : draw species
    
    Return
    -----
    
    Example
    -----
    
    Note
    -----
    Equal to
    g_eles_new     = get_count_ind(u_g)
    pairs_reshaped = reshape_pairs(pairs,g_eles_new)
    nn             = build_nn(u_g,pairs_reshaped,g_eles_new)
    
    G = build_graph_from_nn(nn,index=atoms_vesta)
    
    species_count = speciation_from_G(G,draw_species=True)
    """    
    g_eles_new     = get_count_ind(u_g)
    pairs_reshaped = reshape_pairs(pairs,g_eles_new)
    nn             = build_nn(u_g,pairs_reshaped,g_eles_new)
#    print(nn)
    G = build_graph_from_nn(nn,index=index)
#    nx.draw(G)
    species_count = speciation_from_G_abs(G,draw_species=draw_species)

    return species_count

########## Rountine below summerize what lies below##########
   
def get_atom_index_gas(u_i,u_g):   
    """
    index gas in a frame only, find atom starting index
    
    Parameters
    ------------
    u_i : universe, atoms must be consecutive
    u_g : gas universe of u_i,atoms may be inconsecutive
    
    return
    ------
    u_eles : dict, key is ele in ith frame, value are list, 
             list[0] is the count of atmos
             list[1] is the starting index, starting from 1, not 0
    {'H': [18, 136], 'Mg': [27, 1], 'O': [81, 55], 'Si': [27, 28]}
    
    atoms_abs_all : absolute index for atoms in the universe
    atoms_abs_gas : absolute index for atoms in gas phase only
    atoms_vesta : atoms index in vesta order
    
    """
    u_eles = get_count_ind(u_i)
    atoms_abs_all = [ u_g.names[i]+str(u_g.indices[i]+1) for i in range(len(u_g.names))]  
    atoms_abs_gas = [ u_g.names[i]+str(i) for i in range(len(u_g.names))]
    atoms_vesta = []
    
    for i in range(len(u_g.names)):
        starting_ind = u_eles[u_g.names[i]]['new_start_ind']
        relative_ind = u_g.indices[i] - starting_ind + 1
        atoms_vesta.append(u_g.names[i]+str(relative_ind))
    return atoms_abs_all,atoms_abs_gas,atoms_vesta

#####
def get_count_ind(u_g):
    """
    get count and index of the gas
    
    Parameters
    -----
    u : universe
    
    Return
    -----
    g_eles_new : dict, key is ele, value dis dict with 'count' and 'new_start_ind'
    
    example
    {'H': {'count': 5, 'new_start_ind': 13},
     'Mg': {'count': 2, 'new_start_ind': 0},
     'O': {'count': 10, 'new_start_ind': 3},
     'Si': {'count': 1, 'new_start_ind': 2}}
    """        
    seen = set()
    result = []
    count  = []
    for item in u_g.atoms.names:
        if item not in seen:
            seen.add(item)
            result.append(item)
            count.append(u_g.select_atoms('type '+ item).n_atoms)
    g_eles_new = {}
    for i in range(len(count)):
        g_eles_new[result[i]] = {'new_start_ind':sum(count[:i]), 'count': count[i]}
    return g_eles_new
    

def reshape_pairs(pairs,g_eles_new):
    _clean_pairs(pairs)
    pairs_reshaped = {}
    for ele in g_eles_new:
        pairs_reshaped[ele] = [[]for i in range(g_eles_new[ele]['count'])]
        for key in pairs:
            pair  = key.replace(" ","").split(sep='-') 
            host  = pair[0]
            guest = pair[1]    
            if ele == host:            
#                host_ind  = g_eles_new[host]['new_start_ind']
                guest_ind           = g_eles_new[guest]['new_start_ind']
                guest_abs_ind       = _get_abs_ind(pairs[key],guest_ind)
                pairs_reshaped[ele] = _merge_two_ind(pairs_reshaped[ele],guest_abs_ind) 
    return pairs_reshaped

def build_nn(u_g,pairs_reshaped,g_eles_new):
    """
    build neareast neighbor table
    """
    pos = u_g.positions
    n_g = len(u_g.atoms)    
    nn = np.zeros((n_g,n_g)).astype('int')
    box = u_g.dimensions[:3]   
        
    for ele in g_eles_new:
        host = ele
        host_ind  = g_eles_new[host]['new_start_ind']
        for i in range(len(pairs_reshaped[host])):            
            n_nn = len(pairs_reshaped[host][i])     
            if n_nn == 1:
                nn[host_ind+i, pairs_reshaped[host][i]] = 1
    #            if host_ind+i == guest_ind+pairs[key][i][0]:
    #                print(key)
            elif n_nn >1 : ## find 1st and 2nd NN
                host_pos = pos[host_ind+i]
    #            for j in range(n_nn):
                guest_inds = pairs_reshaped[host][i]
                guest_pos  = pos[guest_inds]
                ### consider periodic boundary condition!!   
                dis_vector = host_pos - guest_pos
                for num in range(len(dis_vector)):
                    if dis_vector[num][0] > box[0]/2:
                        dis_vector[num][0] = box[0] -  dis_vector[num][0]
                    if dis_vector[num][1] > box[1]/2:
                        dis_vector[num][1] = box[1] -  dis_vector[num][1]
                    if dis_vector[num][2] > box[2]/2:
                        dis_vector[num][2] = box[2] -  dis_vector[num][2]                    
                dis  = np.sum(dis_vector**2,axis=1)
                rank  = rankdata(dis)            
                nn[host_ind+i, guest_inds] = rank
    ### nn_dataframe provide a more intuitive way of checking nn
#    import pandas as pd
#    symbols = u_g.names              
#    nn_dataframe = pd.DataFrame(nn,columns=symbols,index=symbols)
    return nn

def build_graph_from_nn(nn,index=None):
    """
    build graph from nn
    index : atoms_abs,atoms_gas,atoms_vesta
    """
    if index is None:
        index = [str(i) for i in range(len(nn))]
        
    G = nx.Graph()
    ## process nonzero values first
    row,col=nn.nonzero()
    
    for i in range(len(row)):
        nn_row = row[i]; nn_col = col[i]
        ## nn_col is the 1st nn of n_row
        ## nn_col --1--> n_row
        ## nn_row --1--> n_col => connect n_col and n_row
        ## nn_row --2--> n_col => connect n_col and n_row
        ## nn_row --0--> n_col is 0, add n_col only
        if nn[nn_row,nn_col] == 1:
            if (nn[nn_col,nn_row] == 1) or (nn[nn_col,nn_row] == 2):
    #            nodes = [(symbols[nn_row]+str(nn_row),symbols[nn_col]+str(nn_col))]
                nodes = [(index[nn_row],index[nn_col])]
                G.add_edges_from(nodes)
            else:
                G.add_node(index[nn_col]) 
        else:
            G.add_node(index[nn_col])
    
    ## process the zeros
    for i in range(len(nn)):
        if not (i in row):        
            G.add_node(index[i])
    return G

def speciation_from_G(G,draw_species=False):
    species_count = {}
    if draw_species:                  
        nx.draw(G,with_labels=True)                    
    species = sorted(nx.connected_components(G), key=len, reverse=True)
#    print(species)
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
#    out = copy.deepcopy
#    print(species_count)
    return species_count

def speciation_from_G_abs(G,draw_species=False):
    """
    extract species from G
    G must be index with atoms_abs_all
    for lifetime statistics, H1H2O1 and H1H2O2 are treat as different species
    
    """
    if draw_species:                  
        nx.draw(G,with_labels=True)                    
    species = sorted(nx.connected_components(G), key=len, reverse=True)
#    print(species)
#    for specie in species:   
#        species_count_abs[specie] = 
    return species

##### life time analysis ####

def check_species_in_life(species,life):
    """
    life has key as tuple, species is set
    
    Return
    -------
    flag : True, specie in life
    key  : None, specie not in life
           any other tuples, key corresponding to the specie
    """
    flag = False
    key  = None
    tuple_species = life.keys()
    set_species   =[]
    ## transfer life's tuple list keys to set list
    
    for tmp in tuple_species:
        set_species.append(set(tmp))
    for i in range(len(tuple_species)):
        if species == set_species[i]:
            flag = True
            key  = list(tuple_species)[i]
            return flag, key
    return flag, key

def life_time(species_sum_abs):
    """
    analyze life life of species
    
    Notes
    -----
    Three case
    1) species in species_sum_abs, not in life  => create it in life and give it value of [1]
    2) species in species_sum_abs, also in life => 
       check if the last value is 0 =>
       if so,  it means it has die out, append a new value [1]
       if not, it means it is still alive, last value + 1
    3) species not in species_sum_abs, in life =>
       check if the last value is 0 =>
       if so, it means it was dead, do nothing
       if not, it means it was alive in the last step, append [0]
    """

    life = {}  
    for i in range(len(species_sum_abs)):
        for species in species_sum_abs[i]:
            flag, key = check_species_in_life(species,life)
            if not flag:
                life[tuple(i for i in species)] = [1] 
                ## [3,4] represents in total appears for two times, 1st stay for 3 fs and 2nd stays for 4 fs
            else:
                if life[key][-1] == 0: # this key dies out before
                    life[key].append(1)
                else:
                    life[key][-1] += 1
        for key2 in life.keys():
            if not (set(key2) in species_sum_abs[i]):
                ### key2 dies out 
                if life[key2][-1] != 0:
                    life[key2].append(0)
    return life

def count_species_time(time):
    """
    calculate average time period and frequency of species
    
    Params
    ------
    time : dict, key is species in tuple, value is list of sperated with 0, 0 means die out
    
    Returns:
    count_time : pd DataFrame, indexes are count and time, cols are species
    """
    import pandas as pd
    
    time_ave = {}
    time_sum = {}
    for key,value in time.items():
        value_array   = np.array(value)
        time_ave[key] = value_array[value_array.nonzero()].mean()
        time_sum[key] = sum(value_array)

    species_count = {}
    species_time  = {}
    for key in time.keys():
        key_index_removed = [re.sub('[^a-zA-Z]+', '', i) for i in key]
        key_index_removed.sort() # aftr sorting the MgOH and HMgO should equal
        key_normal = ""
        for i in key_index_removed:
            key_normal += i
        if key_normal in species_count:
            species_count[key_normal] += time_sum[key]
            species_time[key_normal].append(time_ave[key])
        else:
            species_count[key_normal] = time_sum[key]
            species_time[key_normal] = [time_ave[key]]
    
    species_time_ave = {}
    for key,value in species_time.items():
        value_array   = np.array(value)
        species_time_ave[key] = value_array.mean()
    count_time = pd.DataFrame([species_count,species_time_ave],index=['count','time'])
    count_time.sort_values(by = ['count'],axis=1,ascending=False,inplace=True)    
    return count_time

##### post-analysis on speciation ####
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

def ele_stat(eles,df):
    """
    get a dict, with # of elements for each species, which can be used for ele_stat_all_elements
    
    Param
    ------
    eles : dict, key is all eles, value ele number
    df   : dataframe, iloc[0] must be ['count']!!
           pd.read_csv('species_' + str(begin) + '_' + str(end),sep=',',index_col=[0])
    returns
    -------
    ele_stats : dictionary, key is all eles, value ele # stats for each species
    
    example
    -------
    ele_stats         = ele_stat_all_species({'H','O','Si','Mg'},df)
    ele_counts  : dict of counts of elements

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
    ele_counts = _ele_stat_all_elements(ele_stats,df)
    return ele_counts


def ele_time(species_sum_abs,eles,plot=True):
    """
    calcualte the # of ele vs. for each step
    
    Parameters
    ----------
    species_sum_abs : list of dict
        species of each step.
    eles : list of ele
        eles of system.
    plot : boolean, optional
        plot the result. The default is True.

    Returns
    -------
    dic : dict
        key is ele, value is count of ele for each step.

    """
    dic ={}
    for ele in eles:
        dic[ele] = []
    for i in range(len(species_sum_abs)):
        for key in dic.keys():
            dic[key].append(0)
        for species in species_sum_abs[i]: #{'O226', 'Si108'}
            for atom in species:
                ele = re.sub('[^a-zA-Z]+','',atom)
                dic[ele][-1] += 1 
    if plot:
        for ele in eles:           
            plt.plot(dic[ele],label=ele)
            plt.xlabel('NSW')
            plt.ylabel('# atoms')
            plt.legend()
    return dic

def _ele_stat_all_elements(ele_stats,df):
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
    tots = []
    for ele in ele_stats:
        tmp = ele_stats[ele]*df.loc['count']
        ele_counts[ele] = tmp.sum(axis=0)
    return ele_counts


### subroutine of reshape_pairs ###
def _get_abs_ind(inds,start_ind):
    """
    subroutine of reshape_pairs    
    """
    out = []
    for ind in inds:
        out.append([item+start_ind for item in ind])
    return out

def _merge_two_ind(inds1,inds2):
    """
    subroutine of reshape_pairs
    """
    out = []
    for i in range(len(inds1)):
        tmp = (inds1[i]+inds2[i])
        out.append(tmp)
    return out

def _clean_pairs(pairs):
    """
    subroutine of reshape_pairs,remove the self pairing
    
    Parameters
    -----
    pairs : output of analyze_gas
    
    Return
    -----
    None, pairs are modified    
    
    
    """
    for key in pairs:
        pair  = key.replace(" ","").split(sep='-') 
        host  = pair[0]
        guest = pair[1]
        if host == guest:
            for i in range(len(pairs[key])):
                try:
                    pairs[key][i].remove(i)
                except:
                    pass     

#
#u_g = get_gas(u_i,prox_i,cutoff_prox)
#
#pairs,eles_num,subgroups = analyze_gas(u_g,cutoffs_rdf,show_vapor=True)
#
#atoms_abs,atoms_gas,atoms_vesta = get_atom_index_gas(u_i,u_g)
#
#g_eles_new = get_count_ind_order(u_g)
#speciation_from_G(G,species_count = {},draw_species=True)


######
#def get_atom_index_all(u_i): 
#    """    
#    MERGE TO get_count_ind()
#    index a whole frame, find atom starting index
#    
#    Parameters
#    ------------
#    u_i : universe, atoms must be consecutive
#    
#    return
#    ------
#    u_eles : dict, key is ele in ith frame, value are list, 
#             list[0] is the count of atmos
#             list[1] is the starting index, starting from 1, not 0
#    {'H': [18, 136], 'Mg': [27, 1], 'O': [81, 55], 'Si': [27, 28]}
#    """
#    
#    u_eles = {}
#    for i in u_i.atoms.names:
#        if i not in u_eles:
#            tmp = u_i.select_atoms('type ' + i)
#            ind = tmp.indices[0]+1
#            u_eles[i] = [u_i.select_atoms('type ' + i).atoms.n_atoms,ind]  
#    return u_eles
