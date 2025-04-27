# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:18:34 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import sys
from copy import deepcopy
from copy import copy
import pickle
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import colormaps as cmaps
from matplotlib.colors import TwoSlopeNorm

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/self_limiting_alternative_growth_consumption_coupling')

from simulation_functions import create_and_delete_CR, \
    create_df_and_delete_simulations, prop_chaotic, distance_from_instability, \
    consumer_resource_model_dynamics

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/consumer_resource_modules_2')
from models import Consumer_Resource_Model

# %%

system_size = 200

fixed_parameters = {'mu_c' : 10/system_size, 'm' : 1, 'K' : 1,
                    'gamma' : system_size/system_size}

options_for_mu_g = [0.1, 3]
options_for_sigma_c = [0.05/np.sqrt(system_size), 0.2/np.sqrt(system_size)]
mu_g_sigma_combinations = sce.parameter_combinations([options_for_mu_g,
                                                      options_for_sigma_c], 2)

variable_parameters = np.vstack([mu_g_sigma_combinations,
                                 mu_g_sigma_combinations[1, :]*np.sqrt(system_size)])

parameters = sce.variable_fixed_parameters(variable_parameters, ['mu_g', 'sigma_c', 'sigma_g'],
                                           fixed_parameters)

# %%

ecological_dynamics_test = [consumer_resource_model_dynamics(system_size, system_size,
                                                           parms,
                                                           'growth function of consumption',
                                                           no_communities = 1,
                                                           no_lineages = 1)
                            for parms in tqdm(parameters)]

ecological_dynamics_test = [item[0] for item in ecological_dynamics_test]

# %%

plt.plot(ecological_dynamics_test[0].ODE_sols['lineage 0'].t, 
         ecological_dynamics_test[0].ODE_sols['lineage 0'].y[:system_size, :].T)

# %%

plt.plot(ecological_dynamics_test[1].ODE_sols['lineage 0'].t, 
         ecological_dynamics_test[1].ODE_sols['lineage 0'].y[:system_size, :].T)

# %%

plt.plot(ecological_dynamics_test[2].ODE_sols['lineage 0'].t, 
         ecological_dynamics_test[2].ODE_sols['lineage 0'].y[:system_size, :].T)

# %%

plt.plot(ecological_dynamics_test[3].ODE_sols['lineage 0'].t, 
         ecological_dynamics_test[3].ODE_sols['lineage 0'].y[:system_size, :].T)
