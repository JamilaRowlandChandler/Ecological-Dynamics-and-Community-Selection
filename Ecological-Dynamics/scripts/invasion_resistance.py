# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:32:49 2024

@author: jamil
"""

# cd C:\Users\Jamila\Documents\PhD\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\new_classes
# cd C:\Users\jamil\Documents\PhD\Github Projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\scripts

import numpy as np
from matplotlib import pyplot as plt
import sys
from copy import deepcopy
from tqdm import tqdm
from time import sleep
import pandas as pd

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/model_modules')
from model_classes import gLV

###################

def invasion_resistance(mu_a,sigma_a,no_species,no_invasion_tests):
    
    ###############
    
    original_dynamics = gLV(no_species = no_species, growth_func = 'fixed', growth_args = None,
                            interact_func = 'random',interact_args = {'mu_a':mu_a,'sigma_a':sigma_a})
    original_dynamics.simulate_community(np.arange(5),t_end = 5000)
    
    initial_abundances = np.vstack([ode_sol.y[:,-1] for ode_sol in original_dynamics.ODE_sols.values()])
    
    original_dynamics.simulate_community(np.arange(5),t_end = 10000,init_cond_func=None,
                                     usersupplied_init_conds=initial_abundances.T)
    original_dynamics.calculate_community_properties(np.arange(5),from_which_time = 7000)
    
    print(original_dynamics.invasibility)
    
    original_survival_fraction = \
        np.mean(np.array([diversity/no_species for diversity in original_dynamics.final_diversity.values()]))
        
    #################################
    
    def invade(pbar):
        
        extinct_thresh = 1e-4
    
        one_more_species_matrix = mu_a + sigma_a*np.random.randn(no_species + 1, no_species + 1)
        one_more_species_matrix[:no_species,:no_species] = \
            deepcopy(original_dynamics.interaction_matrix)
        one_more_species_matrix[-1,-1] = 1
            
        abundances_before_invasion = \
            np.vstack([ode_sol.y[:,-1] for ode_sol in original_dynamics.ODE_sols.values()])
        new_species_abundances = np.random.uniform(1e-8,1e-5,5).reshape((5,1))
        new_initial_abundances = np.concatenate((abundances_before_invasion,
                                                 new_species_abundances),
                                                axis=1)
        
        community_to_invade = gLV(no_species = no_species + 1, growth_func = 'fixed', growth_args = None,
                                  interact_func = None, interact_args = {'mu_a':mu_a,'sigma_a':sigma_a},
                                  usersupplied_interactmat = one_more_species_matrix)
        community_to_invade.simulate_community(np.arange(5),t_end = 10000,init_cond_func=None,
                                               usersupplied_init_conds=new_initial_abundances.T)
        
        invasion_success = \
            np.any(np.vstack([deepcopy(simulation.y[-1,140:]) \
                              for simulation in community_to_invade.ODE_sols.values()]) \
                   > extinct_thresh,axis = 1)
                
        sleep(0.01)
        pbar.update(1)
                
        return np.count_nonzero(invasion_success)/5
    
    with tqdm(total=no_invasion_tests) as pbar:
        invasion_successes = \
            np.array([invade(pbar) for i in range(no_invasion_tests)])
        
    #return {'Original community' : deepcopy(original_dynamics),
    #        'Original survival fraction' : original_survival_fraction,
    #        'Invasion probability' : invasion_successes}

    return deepcopy(original_dynamics), original_survival_fraction, invasion_successes    

community_01, survival_fraction_01, invasion_successes_01 = \
    invasion_resistance(0.1, 0.05, 49, 49)
print(survival_fraction_01,np.sum(invasion_successes_01)/49)
plt.plot(community_01.ODE_sols['lineage 0'].t,community_01.ODE_sols['lineage 0'].y.T)

community_07, survival_fraction_07, invasion_successes_07 = \
    invasion_resistance(0.7, 0.05, 49, 49)
print(survival_fraction_07,np.sum(invasion_successes_07)/49)
plt.plot(community_07.ODE_sols['lineage 0'].t,community_07.ODE_sols['lineage 0'].y.T)

community_092, survival_fraction_092, invasion_successes_092 = \
    invasion_resistance(0.9, 0.15, 49, 49)
print(survival_fraction_092,np.sum(invasion_successes_092)/49)
plt.plot(community_092.ODE_sols['lineage 0'].t,community_092.ODE_sols['lineage 0'].y.T)
plt.plot(community_092.ODE_sols['lineage 1'].t,community_092.ODE_sols['lineage 1'].y.T)

community_12, survival_fraction_12, invasion_successes_12 = \
    invasion_resistance(1, 0.15, 49, 49)
print(survival_fraction_12,np.sum(invasion_successes_12)/49)
plt.plot(community_12.ODE_sols['lineage 1'].t,community_12.ODE_sols['lineage 1'].y.T)

invasion_resistance_df = \
    pd.DataFrame([[0.1,0.7,0.9,1],[0.05,0.05,0.15,0.15],
                  ['stable','stable','stable','fluctuating'],
                  [survival_fraction_01,survival_fraction_07,survival_fraction_092,survival_fraction_12],
                  [np.sum(invasion_successes_01)/49,np.sum(invasion_successes_07)/49,
                   np.sum(invasion_successes_092)/49,np.sum(invasion_successes_12)/49]]).T  
invasion_resistance_df.rename(columns={0:'avg. interaction',1:'std. interaction',
                                       2:'ecological dynamics',3:'survival fraction in original community',
                                       4:'invasion success'},inplace=True)
print(invasion_resistance_df.head())

####################################################################

mu_as = [0.1,0.9,1]
sigma_a = 0.15
no_species = 49
no_communities = 15

community_invasion_resistance = \
    {str(mu_a): [invasion_resistance(mu_a, sigma_a, no_species, no_species) \
                 for i in range(no_communities)]
                     for mu_a in mu_as}






