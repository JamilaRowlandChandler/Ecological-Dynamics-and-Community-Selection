# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 23:28:05 2024

@author: jamil

"""
# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\Chemically-mediated models

import numpy as np
import sys
from copy import deepcopy

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Lotka-Volterra models/model_modules')
from model_classes import gLV
from utility_functions import pickle_dump
from community_properties import max_le_gLV

# %%

def fluctuation_coefficient(times, dynamics, from_which_time, extinction_threshold = 1e-3):
     
    last_500_t = np.argmax(times >= from_which_time)
    final_diversity = np.any(dynamics[:, last_500_t:] > extinction_threshold, axis=1)

    extant_species = dynamics[final_diversity, last_500_t:]

    return np.count_nonzero(np.std(extant_species, axis=1)/np.mean(extant_species, axis=1) > 5e-2)

# %%

def gLV_dynamics(mu_a, sigma_a, mu_g, sigma_g, no_species):
    
    no_communities = 25
    no_lineages = 5
   
    def simulate_community(i, no_lineages, no_species, mu_a, sigma_a, mu_g, sigma_g):
        
        print({'mu': mu_a, 'sigma' : sigma_a, 'Community' : i}, '\n')
         
        gLV_dynamics = gLV(no_species = no_species, growth_func = 'normal',
                           growth_args = {'mu_g' : mu_g, 'sigma_g' : sigma_g},
                           interact_func = 'random', 
                           interact_args = {'mu_a' : mu_a, 'sigma_a' : sigma_a},
                           dispersal = 1e-8)
        gLV_dynamics.simulate_community(np.arange(no_lineages),t_end = 3500)
        gLV_dynamics.calculate_community_properties(np.arange(no_lineages), from_which_time = 2500)
        
        gLV_dynamics.fluctuation_coefficient = \
            {'lineage ' + str(i) : fluctuation_coefficient(simulation.t,
                                                           simulation.y,
                                                           2500,
                                                           extinction_threshold = 1e-3)
                 for i, simulation in enumerate(gLV_dynamics.ODE_sols.values())}
            
        gLV_dynamics.lyapunov_exponent = \
            {'lineage ' + str(i) : max_le_gLV(gLV_dynamics, 1000, simulation.y[:,-1],
                                              1e-3, dt = 20, separation = 1e-3)
                  for i, simulation in enumerate(gLV_dynamics.ODE_sols.values())}
        
        return deepcopy(gLV_dynamics)
    
    communities_list = [simulate_community(i, no_lineages, no_species, mu_a, sigma_a, mu_g, sigma_g)
                        for i in range(no_communities)]
    
    return communities_list

# %%

mu_as = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_as = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

mu_g = 1
sigma_g = 0

gLV_communities_fixed_growth = [gLV_dynamics(mu_a, sigma_a, mu_g, sigma_g, no_species = 50) 
                                for sigma_a in sigma_as for mu_a in mu_as]
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_d_s_3.pkl",
            gLV_communities_fixed_growth)

# %%

mu_as = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_as = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

mu_g = 1
sigma_g = 0

gLV_communities_more_species = [gLV_dynamics(mu_a, sigma_a, mu_g, sigma_g, no_species = 250) 
                                for sigma_a in sigma_as for mu_a in mu_as]

pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_more_species.pkl",
            gLV_communities_more_species)
