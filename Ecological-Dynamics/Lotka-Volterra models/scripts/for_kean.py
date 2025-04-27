# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:23:43 2025

@author: jamil
"""

# %%

##################################### ignore this bit ################

import numpy as np
from matplotlib import pyplot as plt
import os
import sys

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Lotka-Volterra models/scripts')
sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Lotka-Volterra models/model_modules_2')

from model_classes import gLV
from community_properties import max_le_gLV 

##################################################################

########################## This is what I would do on my computer ########################

no_lineages = 1

mu_a = 0.5 # or some other value
sigma_a = 0.1 # or some other value
no_species = 50 # or some other value

t_end = 3500 # or some other value
from_time = 2500 # or some other value

# generate community and simulate dynamics

gLV_dynamics = gLV(no_species = no_species, growth_func = 'fixed',
                   growth_args = None,
                   interact_func = 'random', 
                   interact_args = {'mu_a' : mu_a, 'sigma_a' : sigma_a},
                   dispersal = 1e-8)

gLV_dynamics.simulate_community(np.arange(no_lineages), t_end = t_end)
gLV_dynamics.calculate_community_properties(from_time = from_time)
    
gLV_dynamics.lyapunov_exponent = \
    {'lineage ' + str(i) : max_le_gLV(gLV_dynamics, 1000, simulation.y[:,-1],
                                      1e-3, dt = 20, separation = 1e-3)
          for i, simulation in enumerate(gLV_dynamics.ODE_sols.values())}

# plot dynamics

plt.plot(gLV_dynamics.ODE_sols['lineage 0'].t, gLV_dynamics.ODE_sols['lineage 0'].y.T)

# Save randomly generated variables - the interaction matrix and initial conditions

same_interaction_matrix = gLV_dynamics.interaction_matrix
same_initial_conditions = gLV_dynamics.ODE_sols['lineage 0'].y[:,0] # species abundances at t = 0 are the initial conditions

np.save('interaction_matrix.npy', same_interaction_matrix)
np.save('initial_conditions.npy', same_initial_conditions)

# Check if it worked
# clear the gLV object, interaction matrix and initial conditions from the system 
del gLV_dynamics, same_interaction_matrix, same_initial_conditions

########################## This is what you would do on your computer ########################

# Reload data 

same_interaction_matrix = np.load('interaction_matrix.npy')
same_initial_conditions = np.load('initial_conditions.npy')

# generate community and simulate dynamics

# supply same interaction matrix from gLV_dynamics to the new gLV object 
gLV_dynamics_repeat = gLV(no_species = no_species, growth_func = 'fixed',
                          growth_args = None,
                          interact_func = None, 
                          interact_args = None,
                          dispersal = 1e-8,
                          usersupplied_interactmat = same_interaction_matrix)

# Run simulations from the same initial conditions
gLV_dynamics_repeat.simulate_community(np.arange(no_lineages), t_end = t_end,
                                       init_cond_func = None,
                                       usersupplied_init_conds = same_initial_conditions.reshape(len(same_initial_conditions), 1))
gLV_dynamics_repeat.calculate_community_properties(from_time = from_time)
    
gLV_dynamics_repeat.lyapunov_exponent = \
    {'lineage ' + str(i) : max_le_gLV(gLV_dynamics_repeat, 1000, simulation.y[:,-1],
                                      1e-3, dt = 20, separation = 1e-3)
          for i, simulation in enumerate(gLV_dynamics_repeat.ODE_sols.values())}

# plot dynamics to check if they're the same

plt.plot(gLV_dynamics_repeat.ODE_sols['lineage 0'].t, gLV_dynamics_repeat.ODE_sols['lineage 0'].y.T) # yay, they are


