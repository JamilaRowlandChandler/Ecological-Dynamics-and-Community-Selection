# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:56:24 2024

@author: Jamila
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.integrate import solve_ivp
from copy import deepcopy

from model_classes import gLV
from model_classes import gLV_allee

#######################################

# effect = log((median growth of focal in coculture)/(median growth of focal in monoculture))

growth_rates = np.array([0.3,1.2])
interaction_matrix = np.array([[1,0.9],[0.9,1]])

weak_monoculture = gLV(no_species = 1, growth_func = None, growth_args = None,
                       interact_func = None, interact_args = None,
                       usersupplied_growth = growth_rates[0], usersupplied_interactmat = np.array([interaction_matrix[0,0]]),
                       dispersal = 0)
weak_monoculture.simulate_community(np.arange(5), t_end = 1000)

plt.plot(weak_monoculture.ODE_sols['lineage 0'].t,weak_monoculture.ODE_sols['lineage 0'].y.T)

coculture = gLV(no_species = 2, growth_func = None, growth_args = None,
                interact_func = None, interact_args = None,
                usersupplied_growth = growth_rates, usersupplied_interactmat = interaction_matrix,
                dispersal = 0)
coculture.simulate_community(np.arange(5), t_end = 1000)

plt.plot(coculture.ODE_sols['lineage 1'].t,coculture.ODE_sols['lineage 1'].y.T)

######

competition_matrix = interaction_matrix
cooperation_matrix = np.array([[0,2],[0,0]])

coculture_allee = gLV_allee(no_species = 2, growth_func = None, growth_args = None,
                            competition_func = None, competition_args = None,
                            cooperation_func = None, cooperation_args = None,
                            usersupplied_growth = growth_rates,
                            usersupplied_competition = competition_matrix,
                            usersupplied_cooperation = cooperation_matrix,
                            dispersal = 0)
coculture_allee.simulate_community(np.arange(5), t_end = 1000)

plt.plot(coculture_allee.ODE_sols['lineage 1'].t,coculture_allee.ODE_sols['lineage 1'].y.T)

##################################################################

for i in range(20):

    gLV_allee_dynamics = gLV_allee(no_species = 50, growth_func = 'fixed', growth_args = None,
                                   competition_func = 'random', competition_args = {'mu_comp':0.9,'sigma_comp':0.15},
    #                            cooperation_func = 'random', cooperation_args = {'mu_coop':0.9,'sigma_coop':0.15},
                                    cooperation_func = None, cooperation_args = None,
                                    usersupplied_cooperation = np.zeros((50,50)))
    gLV_allee_dynamics.simulate_community(np.arange(5), t_end = 7000)
    gLV_allee_dynamics.calculate_community_properties(np.arange(5))
    
    print('invasibilities = ', list(gLV_allee_dynamics.invasibility.values()),end='\n')
    print('mean invasibility = ', np.mean(list(gLV_allee_dynamics.invasibility.values())),end='\n')

plt.plot(gLV_allee_dynamics.ODE_sols['lineage 0'].t,gLV_allee_dynamics.ODE_sols['lineage 0'].y.T)

###################

for i in range(20):

    gLV_test = gLV(no_species = 50, growth_func = 'fixed', growth_args = None,
                   interact_func = 'random', interact_args = {'mu_a':0.9,'sigma_a':0.15})
    gLV_test.simulate_community(np.arange(5), t_end = 10000)
    gLV_test.calculate_community_properties(np.arange(5))
    
    print('invasibilities = ', list(gLV_test.invasibility.values()),end='\n')
    print('mean invasibility = ', np.mean(list(gLV_test.invasibility.values())),end='\n')


   

gLV_test = gLV(no_species = 50, growth_func = 'fixed', growth_args = None,
               interact_func = 'random', interact_args = {'mu_a':0.9,'sigma_a':0.15})
gLV_test.simulate_community(np.arange(5), t_end = 10000)
gLV_test.calculate_community_properties(np.arange(5))

print('invasibilities = ', list(gLV_test.invasibility.values()),end='\n')
print('mean invasibility = ', np.mean(list(gLV_test.invasibility.values())),end='\n')

plt.plot(gLV_test.ODE_sols['lineage 0'].t,gLV_test.ODE_sols['lineage 0'].y.T)


gLV_allee_dynamics = gLV_allee(no_species = 50, growth_func = 'fixed', growth_args = None,
                               competition_func = None, competition_args = None,
                               cooperation_func = 'sparse', cooperation_args = {'mu_coop':0,'sigma_coop':0,'connectance_coop':0},
                               #cooperation_func = None, cooperation_args = None,
                               usersupplied_competition = gLV_test.interaction_matrix)
                               #usersupplied_cooperation = np.zeros((50,50)))
gLV_allee_dynamics.simulate_community(np.arange(5), t_end = 10000)
gLV_allee_dynamics.calculate_community_properties(np.arange(5))
    
plt.plot(gLV_allee_dynamics.ODE_sols['lineage 0'].t,gLV_allee_dynamics.ODE_sols['lineage 0'].y.T)























