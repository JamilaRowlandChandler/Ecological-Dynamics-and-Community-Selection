# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:40:03 2024

@author: jamil
"""

import numpy as np
from matplotlib import pyplot as plt

from community_dynamics_and_properties_new_version_test2 import *
from community_dynamics_and_properties import *

#######################

gLV_parameters_test = gLV_parameters(no_species = 50,
                                     growth_func = 'fixed', growth_args = None,
                                     interact_func = 'random', interact_args = {'mu_a':0.9,'sigma_a':0.15},
                                     usersupplied_growth = None, usersupplied_interactmat = None,
                                     dispersal = 1e-8)

gLV_test = gLV(gLV_parameters_test, t_end = 10000, init_cond_func = 'Mallmin')
plt.plot(gLV_test.ODE_sol.t,gLV_test.ODE_sol.y.T)

gLV_test2 = gLV(gLV_parameters_test, t_end = 10000, usersupplied_init_cond=np.repeat(0.1,50))
plt.plot(gLV_test2.ODE_sol.t,gLV_test2.ODE_sol.y.T)

gLV_community_test = gLV_community(no_species = 50,
                                     growth_func = 'fixed', growth_args = None,
                                     interact_func = 'random', interact_args = {'mu_a':0.9,'sigma_a':0.15})
gLV_community_test.simulate_community(np.arange(5), t_end = 10000,
                                      func_name = 'Generate initial conditions',
                                      init_cond_func = 'Mallmin')
plt.plot(gLV_community_test.ODE_sols['lineage 0'].t,
         gLV_community_test.ODE_sols['lineage 0'].y.T)

######################

gLV_community_sparse = gLV_community(no_species = 50,
                                     growth_func = 'fixed', growth_args = None,
                                     interact_func = 'sparse',
                                     interact_args = {'mu_a':0.9,'sigma_a':0.15,'connectance':0.7})
gLV_community_sparse.simulate_community(np.arange(5), t_end = 10000,
                                      func_name = 'Generate initial conditions',
                                      init_cond_func = 'Mallmin')
plt.plot(gLV_community_sparse.ODE_sols['lineage 0'].t,
         gLV_community_sparse.ODE_sols['lineage 0'].y.T)

gLV_community_modular = gLV_community(no_species = 50,
                                     growth_func = 'fixed', growth_args = None,
                                     interact_func = 'modular',
                                     interact_args = {'no_modules':3,'p_mu_a':0.9,'p_sigma_a':0.15,
                                                      'p_connectance':1,'q_mu_a':0.6,'q_sigma_a':0.15,
                                                      'q_connectance':0.1})
gLV_community_modular.simulate_community(np.arange(5), t_end = 10000,
                                      func_name = 'Generate initial conditions',
                                      init_cond_func = 'Mallmin')
plt.plot(gLV_community_modular.ODE_sols['lineage 0'].t,
         gLV_community_modular.ODE_sols['lineage 0'].y.T)


