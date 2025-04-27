# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:53:39 2024

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

def CR_to_gLV_parameters(mu_a, mu_b, sigma_a, sigma_b, no_species, model):
    
    no_resources = no_species
    
    # initialise parameters for the C-R model
    
    a = np.abs(mu_b + sigma_b*np.random.randn(no_species, no_resources))
    b = np.abs(mu_a + sigma_a*np.random.randn(no_resources, no_species))
    
    match model:
        
        case '1':
            
            growth = b
            consumption = b * a
            
        case '2':
            
            growth = a * b
            consumption =  a
    
    # convert into gLV
    
    effective_growth_rates = (1/no_resources) * np.sum(growth, axis = 1)
    effective_interaction_matrix = effective_growth_rates @ np.sum(consumption.T, axis = 1) # this might be incorrect, might need meshgrid
    
    return effective_growth_rates, effective_interaction_matrix
    

# %%

def gLV_dynamics(mu_a, sigma_a, no_species, model):
    
    no_communities = 25
    no_lineages = 5
   
    def simulate_community(i, no_lineages, no_species, mu_a, sigma_a, model):
        
        print({'mu': mu_a, 'sigma' : sigma_a, 'Community' : i}, '\n')
        
        growth_rates, interaction_matrix = CR_to_gLV_parameters(mu_a, 1, sigma_a, sigma_a,
                                                                no_species, model)
         
        gLV_dynamics = gLV(no_species = no_species, growth_func = None,
                           growth_args = None, usersupplied_growth = growth_rates,
                           interact_func = None, interact_args = None,
                           usersupplied_interactmat = interaction_matrix,
                           dispersal = 1e-8)
        gLV_dynamics.simulate_community(np.arange(no_lineages),t_end = 3500)
        gLV_dynamics.calculate_community_properties(np.arange(no_lineages), from_which_time = 2500)
             
        gLV_dynamics.lyapunov_exponent = \
            {'lineage ' + str(i) : max_le_gLV(gLV_dynamics, 1000, simulation.y[:,-1],
                                              1e-3, dt = 20, separation = 1e-3)
                  for i, simulation in enumerate(gLV_dynamics.ODE_sols.values())}
        
        return deepcopy(gLV_dynamics)
    
    communities_list = [simulate_community(i, no_lineages, no_species, mu_a, sigma_a, model)
                        for i in range(no_communities)]
    
    return communities_list