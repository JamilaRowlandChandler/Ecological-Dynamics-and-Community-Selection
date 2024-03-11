# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:13:49 2024

@author: Jamila
"""

################## Packages #############

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pickle
from copy import deepcopy

from community_dynamics_and_properties import *

########################

############################ Community Selection Protocols ####################

def community_selection_protocal1(no_communities, no_selection_cycles,
                        no_species, interaction_matrix, interact_args,
                        newborn_biomass=0.5, cells_per_unit_biomass=1000):
    
    community_dict = {}
    lyapunov_exponent_dict = {}
    
    lineages = np.arange(no_communities)
    
    community_dynamics0 = community(no_species,
                                    "fixed", None,
                                    None, interact_args, usersupplied_interactmat=interaction_matrix)
    community_dynamics0.generate_community_function()
    community_dynamics0.simulate_community_with_community_function("Default",lineages,init_cond_func_name="Mallmin")

    community_dict["cycle 0"] = community_dynamics0
    
    ######### Community class object after initialising community ############
    
    community_dynamics = community(no_species,
                                   None, None,
                                   None, interact_args,
                                   usersupplied_growth=community_dynamics0.growth_rates,
                                   usersupplied_interactmat=community_dynamics0.interaction_matrix)
    community_dynamics.generate_community_function(usersupplied_community_function=community_dynamics0.species_contribute_community_function)
    
    for cycle in np.arange(1,no_selection_cycles):
        
        #### Propogation ###########
        
        adults = np.vstack([community_dict["cycle " + str(cycle-1)].ODE_sols[lineage].y[:,-1] \
                  for lineage in community_dict["cycle " + str(cycle-1)].ODE_sols.keys()]).T
            
        adult_newborn_biomass_ratio = np.sum(adults,axis=0)/newborn_biomass
        adults_without_enough_biomass = np.where(adult_newborn_biomass_ratio < 1.0)
        
        if adults_without_enough_biomass[0].size != 0:
        
            lineages = np.delete(lineages,adults_without_enough_biomass)
            adults = np.delete(adults,adults_without_enough_biomass,axis=1)
            
          # 1 newborn per adult
        newborns = np.hstack([propogate_adult_no_replacement(adults[:,i],1) \
                               for i in range(adults.shape[1])])
            
        #### Maturation ##############

        community_dynamics.simulate_community_with_community_function("Supply initial conditions",
                                                                      lineages,
                                                                      array_of_init_conds=newborns)
        deepcopy_community_dynamics = deepcopy(community_dynamics)
        community_dict["cycle " + str(cycle)] = deepcopy_community_dynamics
        
        if cycle == 5 or cycle == 12:
            
            community_dynamics.repeat_lyapunov(lineages)
        
            lyapunov_exponent_dict["cycle " + str(cycle)] = \
                deepcopy(community_dynamics.lyapunov_exponents)
                
    community_dict["Lyapunov Exponents"] = lyapunov_exponent_dict
        
    return community_dict

###################### Migration and Dilution ###################

def dilution(species_abundances, biomass_after_dilution):
    '''
    

    Parameters
    ----------
    species_abundances : TYPE
        DESCRIPTION.
    biomass_after_dilution : TYPE
        DESCRIPTION.

    Returns
    -------
    diluted_community : TYPE
        DESCRIPTION.

    '''
    
    dilution_factor = np.sum(species_abundances)/biomass_after_dilution
    
    diluted_community = species_abundances/dilution_factor
    
    return diluted_community

def migration():
    
    ...
    
def mix_communities(communities_to_mix):
    '''
    

    Parameters
    ----------
    communities_to_mix : TYPE
        DESCRIPTION.

    Returns
    -------
    mixed_communities : TYPE
        DESCRIPTION.

    '''
    
    mixed_communities = np.sum(communities_to_mix)
    
    return mixed_communities
    
##################### Selecting communities with high community function #########

def selection_algorithm(selection_function,function_args):
    '''

    Parameters
    ----------
    selection_function : TYPE
        DESCRIPTION.
    function_args : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    return selection_function(function_args)

def select_top_percent(all_communities_function,top_percent_selected=10):
    '''
    

    Parameters
    ----------
    all_communities_function : TYPE
        DESCRIPTION.
    top_percent_selected : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    top_communities : TYPE
        DESCRIPTION.

    '''
    
    top_percentile = np.percentile(all_communities_function,100-top_percent_selected)
    
    top_communities = np.where(all_communities_function >= top_percentile)
    
    return top_communities
 
###################### Propogration ##############################

# n_tot = no. communities under selection (no. newborns)
# n_chosen = no. adults chosen to reproduce
# If chosen adults all make the same no. offspring, no. offspring newborns per 
#   adult (n_per_a) = n_tot/n_chosen
# We often assign an ideal newborn biomass, bm_target. n_d = bm_adult/bm_target.
# If n_d > n_per_a, the remaining newborns are disgarded. 
# If n_d < n_per_a, the other chosen adult communities,
#    or the adults with the next highest function not chosen to reproduce, 
#   make up the remaining newborns until n_tot is reached.

def propogation(communities_to_propogate,no_communities):
    '''
    

    Parameters
    ----------
    communities_to_propogate : TYPE
        DESCRIPTION.
    no_communities : TYPE
        DESCRIPTION.

    Returns
    -------
    newborn_communities : TYPE
        DESCRIPTION.

    '''
    
    newborns_per_adult = no_newborns_per_adult(communities_to_propogate, no_communities)
    
    newborn_communities = [propogate_adult_no_replacement(communities_to_propogate[:,i], newborns_per_adult[i]) \
                           for i in range(communities_to_propogate.shape(1))]
        
    return newborn_communities

def no_newborns_per_adult(communities_to_propogate,no_communities,newborn_biomass=1.0):
    '''
    

    Parameters
    ----------
    communities_to_propogate : TYPE
        DESCRIPTION.
    no_communities : TYPE
        DESCRIPTION.
    newborn_biomass : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    newborns_per_adult : TYPE
        DESCRIPTION.

    '''
    
    # communities_to_propogate will be some list or a 2d matrix (rows=no. species, cols = no.communities)
    
    breakpoint()

    no_newborn_per_adult = no_communities/communities_to_propogate.shape(1)
    newborns_per_adult = np.repeat(no_newborn_per_adult,communities_to_propogate.shape(1))

    max_newborns_per_adult = np.sum(communities_to_propogate,axis=1)/newborn_biomass

    if np.all(max_newborns_per_adult >= no_newborn_per_adult) != True:
        
        adults_w_not_enough_newborn = np.where(max_newborns_per_adult < no_newborn_per_adult)
        adults_w_extra = np.where(max_newborns_per_adult > no_newborn_per_adult)
        
        newborns_per_adult[adults_w_not_enough_newborn] = max_newborns_per_adult[adults_w_not_enough_newborn]
        
        extra_newborns_needed = np.sum(max_newborns_per_adult[adults_w_not_enough_newborn])
        
        potential_newborns_leftover = max_newborns_per_adult[adults_w_extra] - no_newborn_per_adult
        newborns_to_sample = np.repeat(adults_w_extra,potential_newborns_leftover)
        
        extra_newborns = np.random.choice(newborns_to_sample,extra_newborns_needed,replace=False)
        adults_selected, no_newborns_sampled = np.unique(extra_newborns,return_counts=True)
        
        newborns_per_adult[adults_selected] += no_newborns_sampled
            
    return newborns_per_adult
    
def propogate_adult_replace(community_to_propogate,
                              no_newborns,newborn_biomass=1.0,
                              cells_per_unit_biomass=100):
    '''
    
    Inspired by Xie, Yuan, and Shou (2019) and Indra's algorithms.
        Sampling of biomass WITH replacement.

    Parameters
    ----------
    community_to_propogate : TYPE
        DESCRIPTION.
    no_newborns : TYPE
        DESCRIPTION.
    newborn_biomass : TYPE, optional
        DESCRIPTION. The default is 100.
    cells_per_unit_biomass : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    species_abundance_per_newborn : TYPE
        DESCRIPTION.

    '''
    no_cells_newborn = np.multiply(newborn_biomass,cells_per_unit_biomass)
    
    probability_of_selecting_species = community_to_propogate/ \
    np.sum(community_to_propogate)
    
    cells_per_newborn = np.random.multinomial(no_cells_newborn,
                                              probability_of_selecting_species,
                                              size=no_newborns)
    species_abundance_per_newborn = cells_per_newborn/cells_per_unit_biomass
    
    return species_abundance_per_newborn

def propogate_adult_no_replacement(community_to_propogate,
                              no_newborns,newborn_biomass=0.5,
                              cells_per_unit_biomass=1000):
    '''
    

    Parameters
    ----------
    community_to_propogate : TYPE
        DESCRIPTION.
    no_newborns : TYPE
        DESCRIPTION.
    newborn_biomass : TYPE, optional
        DESCRIPTION. The default is 100.
    cells_per_unit_biomass : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    newborns_by_biomass : TYPE
        DESCRIPTION.

    '''
    
    newborns_by_biomass = np.zeros((len(community_to_propogate),no_newborns))
    no_cells_newborn = np.round(np.multiply(newborn_biomass,cells_per_unit_biomass),0)
    no_cells_adult = np.round(np.multiply(community_to_propogate,
                                          cells_per_unit_biomass),0)
    no_cells_per_species = np.repeat(np.arange(len(community_to_propogate)),no_cells_adult.astype(int))
    
    newborns_cells = np.random.choice(no_cells_per_species,
                                      (no_cells_newborn.astype(int),no_newborns),
                                      replace=False)
    
    for i in range(newborns_cells.shape[1]):
        
        spec_present, cell_count = np.unique(newborns_cells[:,i],return_counts=True)
        
        newborns_by_biomass[spec_present,i] += cell_count/cells_per_unit_biomass
    
    return newborns_by_biomass

############################ Community Properties ###############################     

def heritability_during_selection(community_functions,no_cycles):
    
    no_communities = np.round(community_functions.shape[0]/no_cycles).astype(int)
    
    parent_offspring_array = \
        sliding_window_view(community_functions, no_communities*2)[::no_communities,:]
        
    heritability_over_selection_cycles = \
        np.apply_along_axis(heritability,1,parent_offspring_array,no_communities)
    
    heritability_over_selection_cycles = \
        np.stack((np.arange(1,len(heritability_over_selection_cycles)+1),heritability_over_selection_cycles)).T
    
    return heritability_over_selection_cycles

def heritability(parent_offspring_1darray,no_communities):
    
    parent = parent_offspring_1darray[:no_communities]
    offspring = parent_offspring_1darray[no_communities:]
    
    covariance_matrix = np.cov(parent,offspring)
    var_parent = covariance_matrix[0,0]
    cov_parent_offspring = covariance_matrix[0,1]
    
    h_squared = cov_parent_offspring/var_parent
    
    return h_squared

####################### Other ###########################

