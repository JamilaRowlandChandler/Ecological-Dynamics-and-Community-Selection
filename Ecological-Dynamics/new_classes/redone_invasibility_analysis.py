# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:37:41 2024

@author: Jamila
"""

# cd C:\Users\Jamila\Documents\PhD\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\new_classes

import numpy as np
import pandas as pd
from copy import deepcopy

from model_classes import gLV

from utility_functions import generate_distribution
from utility_functions import community_object_to_df
from utility_functions import pickle_dump

#######################################

def community_simulations_fixed_std(std):
 
    min_species = 4
    max_species = 50
    no_species_to_test = np.arange(min_species,max_species,3)
    
    interaction_distributions = generate_distribution([0.1,1.1], [std,std+0.04])
    
    no_communities = 10
    no_lineages = 5
     
    def interaction_strength_community_dynamics(no_species_to_test,mu_a,sigma_a,no_lineages,
                                                no_communities):
        
        def create_and_simulate_community(i,no_species):
            
            gLV_dynamics = gLV(no_species = no_species, growth_func = 'fixed', growth_args = None,
                           interact_func = 'random',interact_args = {'mu_a':mu_a,'sigma_a':sigma_a})
            gLV_dynamics.simulate_community(np.arange(no_lineages),t_end = 10000)
            gLV_dynamics.calculate_community_properties(np.arange(no_lineages),from_which_time = 7000)
            
            if i == 0:
            
                print({'mu_a':gLV_dynamics.mu_a,'sigma_a':gLV_dynamics.sigma_a,
                       'no_species':gLV_dynamics.no_species}, end = '\n')
            
            return deepcopy(gLV_dynamics)
            
        output = {str(no_species) : 
                  [create_and_simulate_community(i,no_species) for i in range(no_communities)] \
                  for no_species in no_species_to_test}
            
        return output
    
    community_dynamics_interact_dist = {str(i_d['mu_a']) + str(i_d['sigma_a']) : \
                                        interaction_strength_community_dynamics(no_species_to_test,i_d['mu_a'],
                                                                                i_d['sigma_a'],no_lineages,no_communities) \
                                        for i_d in interaction_distributions}
    
    return community_dynamics_interact_dist
        
community_dynamics_invasibility_005 = community_simulations_fixed_std(0.05)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005_new.pkl",
            community_dynamics_invasibility_005)        

community_dynamics_invasibility_01 = community_simulations_fixed_std(0.1)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_01_new.pkl",
            community_dynamics_invasibility_01)        
        
community_dynamics_invasibility_015 = community_simulations_fixed_std(0.15)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_015_new.pkl",
            community_dynamics_invasibility_015)        

community_dynamics_invasibility_02 = community_simulations_fixed_std(0.2)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_02_new.pkl",
            community_dynamics_invasibility_02)        