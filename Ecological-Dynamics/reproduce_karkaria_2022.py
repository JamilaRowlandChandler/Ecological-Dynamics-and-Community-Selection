# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:46:37 2024

@author: jamil
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from community_dynamics_and_properties_v2 import *

####################

no_lineages = 1

no_species = 4

growth_rates = np.array([1,0.72,1.53,1.27])

interaction_matrix = np.array([[1,1.09,1.52,0],
                               [0,1,0.44,1.36],
                               [2.33,0,1,0.47],
                               [1.21,0.51,0.35,1]])

karkaria_community = community(no_species,
                               None, None,
                               None, {'mu_a':None,'sigma_a':None},
                               usersupplied_growth=growth_rates,
                               usersupplied_interactmat=interaction_matrix,
                               dispersal=0)
karkaria_community.simulate_community(1000, 'Supply initial conditions',
                                      np.arange(no_lineages),
                                      array_of_init_conds=np.reshape(np.repeat(0.5,4),(4,1)))
karkaria_community.calculate_lyapunov_exponents(np.arange(no_lineages),dt=1000,
                                                separation=1e-5)
karkaria_community.lyapunov_exponents

#####################

interaction_matrix_chaos = np.load('chaos_09_005.npy')
interaction_matrix_oscillations = np.load('oscillations_09_005.npy')
interaction_matrix_stable = np.load('stable_09_005.npy')

initial_conditions = np.tile(0.2,(50,4))

community_dynamics_chaos = community(50,
                                "fixed", None,
                                None, {'mu_a':None,'sigma_a':None},
                                usersupplied_interactmat=interaction_matrix_chaos)
community_dynamics_chaos.simulate_community(10000,'Supply initial conditions',
                                      np.arange(4),
                                      array_of_init_conds=initial_conditions)
community_dynamics_chaos.calculate_lyapunov_exponents(np.arange(4),dt=2000,separation=1e-3)

###

community_dynamics_oscillations = community(50,
                                "fixed", None,
                                None, {'mu_a':None,'sigma_a':None},
                                usersupplied_interactmat=interaction_matrix_oscillations)
community_dynamics_oscillations.simulate_community(10000,'Supply initial conditions',
                                      np.arange(4),
                                      array_of_init_conds=initial_conditions)
community_dynamics_oscillations.calculate_lyapunov_exponents(np.arange(4),dt=2000,separation=1e-3)

###

community_dynamics_stable = community(50,
                                "fixed", None,
                                None, {'mu_a':None,'sigma_a':None},
                                usersupplied_interactmat=interaction_matrix_stable)
community_dynamics_stable.simulate_community(10000,'Supply initial conditions',
                                      np.arange(4),
                                      array_of_init_conds=initial_conditions)
community_dynamics_stable.calculate_lyapunov_exponents(np.arange(4),dt=2000,separation=1e-3)

community_dynamics_chaos.lyapunov_exponents
community_dynamics_oscillations.lyapunov_exponents
community_dynamics_stable.lyapunov_exponents
