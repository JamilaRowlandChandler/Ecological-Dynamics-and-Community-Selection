# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:57:46 2024

@author: jamil
"""

################ Import packages ########

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
import importlib

from community_dynamics_and_properties import *

############ community_parameters class tutorial ##############

# Example 1
community_parameters_object = community_parameters(no_species=3, growth_func_name='fixed', growth_args=None,
                                                   interact_func_name='random', interact_args={'mu_a':0.9,'sigma_a':0.15},
                                                   usersupplied_growth=None, usersupplied_interactmat=None,
                                                   dispersal=1e-8)

## Alternative representation
no_species = 3
interaction_distribution = {'mu_a':0.9,'sigma_a':0.15}
dispersal=1e-8

community_parameters_object = community_parameters(no_species, 'fixed', None, 'random', interaction_distribution,
                                                   None, None, 1e-8)

# Example 2
community_parameters_object = community_parameters(no_species=3, growth_func_name='normal', growth_args={'mu_g':1,'sigma_g':3},
                                                   interact_func_name='random', interact_args={'mu_a':0.9,'sigma_a':0.15},
                                                   usersupplied_growth=None, usersupplied_interactmat=None,
                                                   dispersal=1e-8)

# Example 3
user_growth_rates = np.array([1.1,0.7,1.5]) # growth rates = 1.1, 0.7 and 1.5
user_interaction_matrix = np.array([[1,0.2,1.4],[0,1,0.6],[1.1,0.9,1]])

community_parameters_object = community_parameters(no_species=3, growth_func_name=None, growth_args=None, interact_func_name=None, interact_args=None,
                                                   usersupplied_growth=user_growth_rates, usersupplied_interactmat=user_interaction_matrix,
                                                   dispersal=1e-8)

# Inspect class attributes
print('Initial number of species =', community_parameters_object.no_species, end='\n')
print('Growth rates =', community_parameters_object.growth_rates, end='\n')
print('Interaction matrix =', community_parameters_object.interaction_matrix, end='\n')
print('Dispersal =', community_parameters_object.dispersal, end='\n')

####################### gLV class tutorial ####################

########### Simulations ###########

# Example 1
community_parameters_object = community_parameters(no_species=50, growth_func_name='fixed', growth_args=None,
                                                   interact_func_name='random', interact_args={'mu_a':0.9,'sigma_a':0.15},
                                                   usersupplied_growth=None, usersupplied_interactmat=None,
                                                   dispersal=1e-8)

gLV_object = gLV(community_parameters_object, t_end=5000, init_cond_func_name='Mallmin')

# Example 2
community_parameters_object = community_parameters(no_species=50, growth_func_name='fixed', growth_args=None,
                                                   interact_func_name='random', interact_args={'mu_a':0.9,'sigma_a':0.15},
                                                   usersupplied_growth=None, usersupplied_interactmat=None,
                                                   dispersal=1e-8)

initial_species_abundances = np.repeat(0.5,50) # all species start with abundances of 0.5

gLV_object = gLV(community_parameters_object, t_end=5000, usersupplied_init_cond=initial_species_abundances)

# Plot simulations
plt.plot(gLV_object.ODE_sol.t,gLV_object.ODE_sol.y.T)
plt.xlabel('time (t)',fontsize=14)
plt.ylabel('Species abundance',fontsize=14)
plt.show()

# Inspect class attributes
print('Initial species abundances = ', gLV_object.initial_abundances, end= '\n')

######### Analysing community properties #############

gLV_object.identify_community_properties()

print('Species diversity at the end of simulation = ', gLV_object.final_diversity, end='\n')
print('Species composition at the end of simulation = ', np.trim_zeros(gLV_object.final_composition,'b').astype(int), end='\n')
print('Invasibility = ', gLV_object.invasibility, end='\n')

######################### community class tutorial #####################

# Example 1
community_dynamics = community(no_species=50, growth_func_name='fixed', growth_args=None, interact_func_name='random',
                               interact_args={'mu_a':0.9,'sigma_a':0.15}, dispersal=1e-8) # very similar to creating a community_parameters object

no_lineages = 5
community_dynamics.simulate_community(lineages=np.arange(no_lineages), t_end = 10000, func_name='Generate initial conditions',
                                      init_cond_func_name ='Mallmin') # np.arange(no_lineages) generates lineage 0, lineage 1, ... lineage 4.

# Example 2
community_dynamics = community(no_species=50, growth_func_name='fixed', growth_args=None, interact_func_name='random',
                               interact_args={'mu_a':0.9,'sigma_a':0.15}, dispersal=1e-8) # very similar to creating a community_parameters object

no_lineages = 5
initial_species_abundances = 0.5 + 0.3*np.random.randn(50,no_lineages) # matrix dimensions no_species x no_Lineages
community_dynamics.simulate_community(lineages=np.arange(no_lineages), t_end = 10000, func_name='Supply initial conditions',
                                      array_of_init_conds=initial_species_abundances)

# Plotting simulations
plt.plot(community_dynamics.ODE_sols['lineage 0'].t,community_dynamics.ODE_sols['lineage 0'].y.T)
plt.xlabel('time (t)',fontsize=14)
plt.ylabel('Species abundance',fontsize=14)
plt.title('Population dynamics of lineage 0',fontsize=16)
plt.show()

# Inspecting community properties/attributes
print('Species diversity at the end of simulation = ', community_dynamics.diversity, end='\n')
print('Invasibilities = ', community_dynamics.invasibilities, end='\n')

tidied_final_compositions = {lineage : np.trim_zeros(composition,'b').astype(int) for lineage, composition in community_dynamics.final_composition.items()}
print('Species composition at the end of simulation = ', tidied_final_compositions , end='\n')

print('Number of unique species compositions = ', community_dynamics.no_unique_compositions, end = '\n')
print('Compositions = ', community_dynamics.unique_composition_label, end = '\n')








