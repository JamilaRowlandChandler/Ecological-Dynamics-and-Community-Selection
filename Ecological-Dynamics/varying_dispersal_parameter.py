# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:04:10 2024

@author: jamil
"""


##############################

# Home - cd Documents/PhD/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/
# Work - cd "Documents/PhD for github/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/" 

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
import itertools

from community_dynamics_and_properties_v2 import *

#################################################

no_communities = 10
no_lineages = 5
no_species = 50
t_end = 10000
interact_dist = {'mu_a':0.9,'sigma_a':0.15}

def vary_community_dispersal_rate(dispersal_rates):
    
    initial_community_dynamics = community(no_species, 'fixed', None, 'random', interact_dist)
    initial_community_dynamics.simulate_community(t_end, 'Default', np.arange(no_lineages),
                                                  init_cond_func_name='Mallmin')
    
    initial_conditions = np.stack(list(initial_community_dynamics.initial_abundances.values()),axis=1)
    
    def initialise_and_simulate_community(dispersal_rate):
        
        community_dynamics = community(no_species,'fixed', None,None,interact_dist,
                                       usersupplied_interactmat=initial_community_dynamics.interaction_matrix,
                                       dispersal=dispersal_rate)
        community_dynamics.simulate_community(t_end, 'Supply initial conditions',
                                              np.arange(no_lineages),
                                              array_of_init_conds=initial_conditions)
        return community_dynamics
    
    community_dynamics_different_migration_rates = [deepcopy(initialise_and_simulate_community(dispersal)) \
                                                    for dispersal in dispersal_rates]
    
    return community_dynamics_different_migration_rates

communities_migration_rates = [vary_community_dispersal_rate(np.array([0,1e-8,1e-7,1e-6,
                                                                      1e-5,1e-3,1e-1])) \
                               for i in range(no_communities)]
    
plt.plot(communities_migration_rates[1][5].ODE_sols['lineage 0'].t,
          communities_migration_rates[1][5].ODE_sols['lineage 0'].y.T)