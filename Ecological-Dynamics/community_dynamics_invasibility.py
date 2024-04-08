# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:20:11 2024

@author: jamil
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
import itertools

from community_dynamics_and_properties_v2 import *

#################################################

def community_simulations_fixed_std(std):

    community_dynamics_invasibility = {}
    
    min_species = 4
    max_species = 50
    no_species_to_test = np.arange(min_species,max_species,3)
    
    interaction_distributions = generate_distribution([0.1,1.1], [std,std+0.04])
    
    no_communities = 10
    no_lineages = 5
    t_end = 10000
    
    for interact_dist in interaction_distributions:
        
        print(interact_dist)
        
        community_dynamics_interact_dist = {}
        
        for no_species in no_species_to_test:
            
            print(str(no_species) + ' species')
        
            def generate_and_simulate_communities(interact_dist,no_species,no_lineages,t_end):
                
                community_dynamics = community(no_species,"fixed", None,'random',
                                               interact_dist)
                community_dynamics.simulate_community(t_end,"Default",np.arange(no_lineages),
                                                      init_cond_func_name="Mallmin")
                
                return deepcopy(community_dynamics)
            
            communities = [generate_and_simulate_communities(interact_dist,no_species,no_lineages,t_end) \
             for comm in range(no_communities)] 
        
            community_dynamics_interact_dist[(str(no_species) + ' species')] = communities
        
        community_dynamics_invasibility[str(interact_dist['mu_a']) + str(interact_dist['sigma_a'])] = \
            community_dynamics_interact_dist
            
    return community_dynamics_invasibility

community_dynamics_invasibility_005 = community_simulations_fixed_std(0.05)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005_2.pkl",
            community_dynamics_invasibility_005)        

community_dynamics_invasibility_01 = community_simulations_fixed_std(0.1)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_01_2.pkl",
            community_dynamics_invasibility_01)        
        
community_dynamics_invasibility_015 = community_simulations_fixed_std(0.15)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_015_2.pkl",
            community_dynamics_invasibility_015)        

community_dynamics_invasibility_02 = community_simulations_fixed_std(0.2)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_02_2.pkl",
            community_dynamics_invasibility_02)        
        



















      