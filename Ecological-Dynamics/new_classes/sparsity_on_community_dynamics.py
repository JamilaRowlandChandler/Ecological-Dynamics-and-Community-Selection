# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:19:22 2024

@author: jamil
"""

# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\new_classes

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy

from model_classes import gLV

from utility_functions import community_object_to_df
from utility_functions import pickle_dump

#######################################

def custom_sparse_matrix(random_interaction_matrix,connectance):
    
    n = random_interaction_matrix.shape[0]
    
    are_species_interacting = \
        np.random.binomial(1,connectance,size=n*n).reshape((n,n))
        
    interact_mat = random_interaction_matrix * are_species_interacting
    
    np.fill_diagonal(interact_mat, 1)
    
    return interact_mat
    
def sparsity_effect_community_dynamics(i,connectances,
                                       no_species,mu_a,sigma_a,no_lineages,
                                       no_sparse_communities):
    
    print({'Community':i,'mu_a':mu_a,'sigma_a':sigma_a,
           'no_species':no_species}, end = '\n')
    
    gLV_dense = gLV(no_species = no_species, growth_func = 'fixed', growth_args = None,
                   interact_func = 'random',interact_args = {'mu_a':mu_a,'sigma_a':sigma_a})
    gLV_dense.simulate_community(np.arange(no_lineages),t_end=10000)
    gLV_dense.calculate_community_properties(np.arange(no_lineages),from_which_time=7000)
    
    def same_interaction_strength_different_connectance(connectance,no_sparse_communities):
        
        def create_and_simulate_sparse_community():
            
            sparse_interactmat = custom_sparse_matrix(gLV_dense.interaction_matrix,connectance)
        
            gLV_sparse = gLV(no_species = no_species, growth_func = 'fixed', growth_args = None,
                           interact_func = None,interact_args = {'mu_a':mu_a,'sigma_a':sigma_a,
                           'connectance':connectance},
                           usersupplied_interactmat = sparse_interactmat)
            gLV_sparse.simulate_community(np.arange(no_lineages),t_end=10000)
            gLV_sparse.calculate_community_properties(np.arange(no_lineages),from_which_time=7000)
    
            return deepcopy(gLV_sparse)
        
        gLV_sparse_communities = [create_and_simulate_sparse_community() \
                                  for i in range(no_sparse_communities)]
            
        return gLV_sparse_communities
    
    output = {str(connectance) : \
              same_interaction_strength_different_connectance(connectance,no_sparse_communities) \
              for connectance in connectances}
        
    output['1.0'] = deepcopy(gLV_dense)
    
    return output

connectances = np.array([0.1,0.3,0.5,0.8])

species_range = np.arange(20,55,5)

interaction_distributions = [{'mu_a':0.7,'sigma_a':0.15},{'mu_a':0.7,'sigma_a':0.2},
                             {'mu_a':0.9,'sigma_a':0.1},{'mu_a':0.9,'sigma_a':0.2},
                             {'mu_a':1,'sigma_a':0.1},{'mu_a':1,'sigma_a':0.2},
                             {'mu_a':1.2,'sigma_a':0.1},{'mu_a':1.2,'sigma_a':0.2}]

no_lineages = 5
no_communities = 5
no_sparse_communities = 5

#community_dynamics_with_sparsity = \
#    {str(i_d['mu_a']) + str(i_d['sigma_a']) : {
#        str(no_species) : {
#            str('Community ') + str(i) : sparsity_effect_community_dynamics(i,connectances,
#                                                   no_species,i_d['mu_a'],i_d['sigma_a'],
#                                                   no_lineages,no_sparse_communities) \
#                for i in range(no_communities)}
#            for no_species in species_range}
#        for i_d in interaction_distributions}
            
community_dynamics_with_sparsity_01 = \
    {str(i_d['mu_a']) + str(i_d['sigma_a']) : {
        str(no_species) : {
            str('Community ') + str(i) : sparsity_effect_community_dynamics(i,connectances,
                                                   no_species,i_d['mu_a'],i_d['sigma_a'],
                                                   no_lineages,no_sparse_communities) \
                for i in range(no_communities)}
            for no_species in species_range}
        for i_d in interaction_distributions[4:6]}
        
pickle_dump('C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_with_sparsity_01.pkl',
            community_dynamics_with_sparsity_01)
           
community_dynamics_with_sparsity_012 = \
    {str(i_d['mu_a']) + str(i_d['sigma_a']) : {
        str(no_species) : {
            str('Community ') + str(i) : sparsity_effect_community_dynamics(i,connectances,
                                                   no_species,i_d['mu_a'],i_d['sigma_a'],
                                                   no_lineages,no_sparse_communities) \
                for i in range(no_communities)}
            for no_species in species_range}
        for i_d in interaction_distributions[6:]}
 
pickle_dump('C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_with_sparsity_012.pkl',
            community_dynamics_with_sparsity_012)

for community_dynamics_with_sparsity in [community_dynamics_with_sparsity_01,community_dynamics_with_sparsity_012]:
    for key0, communities_i_d in community_dynamics_with_sparsity.items():
        for key1, communities_no_species in communities_i_d.items():
            for key2, community_connectances in communities_no_species.items():
                for connectance, communities_connectance in community_connectances.items():
                    if connectance == '1.0':
                        
                        community_dynamics_with_sparsity[key0][key1][key2][connectance].connectance = 1
                        
                        community_dynamics_with_sparsity[key0][key1][key2][connectance] = \
                            [community_dynamics_with_sparsity[key0][key1][key2][connectance]]

    
communities_with_sparsity_df = \
    pd.concat([community_object_to_df(community_object,community_label = community_label,
                                      community_attributes=['mu_a','sigma_a',
                                                            'no_species','connectance',
                                                            'final_diversity','invasibility']) \
               for community_dynamics_with_sparsity in [community_dynamics_with_sparsity_01,community_dynamics_with_sparsity_012]
                   for communities_i_d in community_dynamics_with_sparsity.values()
                       for communities_no_species in communities_i_d.values()
                           for community_label, community_connectances in communities_no_species.items()
                               for communities_connectance in community_connectances.values()
                                   for community_object in communities_connectance])
     
communities_with_sparsity_df['survival_fraction'] = \
    communities_with_sparsity_df['final_diversity']/communities_with_sparsity_df['no_species']
        
communities_with_sparsity_df.to_csv('C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_with_sparsity.csv')