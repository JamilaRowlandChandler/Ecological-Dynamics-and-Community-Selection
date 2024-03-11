# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:59:50 2024

@author: jamil
"""

################## Packages #############

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import pandas as pd

from community_dynamics_and_properties_v2 import *

########################################

interation_matrix_chaos = np.load('chaos_09_005.npy')
interation_matrix_oscillations = np.load('oscillations_09_005.npy')
interation_matrix_stable = np.load('stable_09_005.npy')

########################################

no_species = 50
lineages = np.arange(5)
interact_func = 'random'
interact_func_args = {'mu_a':0.9,'sigma_a':0.05}

community_dynamics_chaos = community(no_species,
                                "fixed", None,
                                interact_func, interact_func_args,
                                usersupplied_interactmat=interation_matrix_chaos)
community_dynamics_chaos.simulate_community(30000,'Default',lineages,init_cond_func_name="Mallmin")
community_dynamics_chaos.calculate_lyapunov_exponents(lineages,separation=1e-3,dt=7000)

community_dynamics_oscillations = community(no_species,
                                "fixed", None,
                                interact_func, interact_func_args,
                                usersupplied_interactmat=interation_matrix_oscillations)
community_dynamics_oscillations.simulate_community(30000,"Default",lineages,init_cond_func_name="Mallmin")
community_dynamics_oscillations.calculate_lyapunov_exponents(lineages,separation=1e-3,dt=7000)

community_dynamics_stable = community(no_species,
                                "fixed", None,
                                interact_func, interact_func_args,
                                usersupplied_interactmat=interation_matrix_stable)
community_dynamics_stable.simulate_community(30000,"Default",lineages,init_cond_func_name="Mallmin")
community_dynamics_stable.calculate_lyapunov_exponents(lineages,separation=1e-3,dt=7000)

for value in community_dynamics_chaos.lyapunov_exponents.values():
     print(value[0]*1e4)
for value in community_dynamics_oscillations.lyapunov_exponents.values():
     print(value[0]*1e4)
for value in community_dynamics_stable.lyapunov_exponents.values():
     print(value[0]*1e4)

for value in community_dynamics_chaos.lyapunov_exponents.values():
     print(np.round(value[1]*1e4,1))
for value in community_dynamics_oscillations.lyapunov_exponents.values():
     print(np.round(value[1]*1e4,1))
for value in community_dynamics_stable.lyapunov_exponents.values():
     print(np.round(value[1]*1e4,1))

#####################################


#############################

three_species_community_chaos = pd.read_pickle('three_species_community_chaos.pkl')

three_species_community = community(3,"fixed", None, interact_func, {'mu_a':0.9,'sigma_a':0.15})
three_species_community.simulate_community(30000,'Default',lineages,init_cond_func_name="Mallmin")
three_species_community.calculate_lyapunov_exponents(lineages,separation=1e-3,dt=7000)
