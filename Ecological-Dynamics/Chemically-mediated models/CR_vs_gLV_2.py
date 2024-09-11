# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:59:26 2024

@author: jamil
"""

# %%

# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\Chemically-mediated models

########################

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects as pe
import pandas as pd
import seaborn as sns
import sys

from CR_vs_gLV_functions_2 import *

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Lotka-Volterra models/model_modules')
from utility_functions import pickle_dump

# %%

####################################### SIMULATIONS ##################################

############################ generalised Lotka-Volterra dynamics ##############################

# Classic gLV with fixed self-interactions and growth rate.

mu_as = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_as = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

mu_g = 1
sigma_g = 0

gLV_communities_fixed_growth = [gLV_dynamics(mu_a, sigma_a, mu_g, sigma_g, no_species = 50) 
                                for sigma_a in sigma_as for mu_a in mu_as]
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_d_s_3.pkl",
            gLV_communities_fixed_growth)

# %%

# gLV with fixed self-interaction and slightly variable growth rate

mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_cs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

mu_g = 1
sigma_gs = sigma_cs

mu_as = mu_cs
sigma_as = [[np.sqrt((sigma**2 + mu_c**2) * (sigma**2 + mu_g**2) \
                - (mu_c**2 * mu_g**2)) for sigma in sigma_cs]
            for mu_c in mu_cs]

gLV_communities_fixed_self = [gLV_dynamics(mu_a, sigma_a, mu_g, sigma_g, no_species = 50) 
                               for mu_a, sigma_mu_a in zip(mu_as, sigma_as) 
                               for sigma_a, sigma_g in zip(sigma_mu_a, sigma_gs)]
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_fixedself_kindvariableg_2.pkl",
            gLV_communities_fixed_self)

# %%

###################################### Consumer - Resource dynamics ###############################

# C-R model where growth does not scale with consumption, but does have the same variance

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_string = ['002','004','006','008','01','015','02']
sigma_cs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

no_species = 50

for sigma_c, sigma_s in zip(sigma_cs, sigma_string):
    
    for mu_c, mu_s in zip(mu_cs, mu_string):
        
        filename_CR = "CR_d_s_small_3_" + str(mu_s) + "_" + str(sigma_s)

        create_and_delete_CR(filename_CR, {'mu_c' : mu_c, 'sigma_c' : sigma_c,
                                           'mu_g' : 1, 'sigma_g' : sigma_c,
                                           'no_species' : no_species,
                                           'no_resources' : no_species})

#%%

# C-R model with growth scaled with consumption

mu_c = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_c = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

no_species = 50

for sigma in sigma_c :
    
    for mu in mu_c:
        
        filename_CR = "cr_growth_scaled_consumption_3_" + str(mu) + "_" + str(sigma)

        create_and_delete_CR(filename_CR, {'mu_c' : mu, 'sigma_c' : sigma,
                                           'mu_g' : mu, 'sigma_g' : sigma,
                                           'no_species' : no_species,
                                           'no_resources' : no_species})
        
# %%

# New C-R model where growth is a function of consumption

mu_c = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_c = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

no_species = 50

for sigma in sigma_c :
    
    for mu in mu_c:
        
        filename_CR = "cr_growth_scaled_consumption_new_model_" + str(mu) + "_" + str(sigma)

        create_and_delete_CR(filename_CR, {'mu_c' : mu, 'sigma_c' : sigma,
                                           'mu_g' : 1, 'sigma_g' : sigma,
                                           'no_species' : no_species,
                                           'no_resources' : no_species},
                             function_option = '2')
        
#%%

#################################### Dataframes ############################################

################################## Consumer - Resource models ######################

# C-R model where growth does not scale with consumption, but does have the same variance

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_string = ['005','01','015','02']
sigma_cs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

data_mu_sigma_s = pd.concat([create_df_and_delete_simulations("CR_d_s_small_3_" + str(mu_s) + "_" + str(sigma_s), mu_c, sigma_c)
                              for sigma_c, sigma_s in zip(sigma_cs, sigma_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

data_mu_sigma_s['Model'] = np.repeat('CR', data_mu_sigma_s.shape[0])

# %%

# C-R model with growth scaled with consumption (1)

mu_c = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_c = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

data_mu_sigma_gc = pd.concat([create_df_and_delete_simulations("cr_growth_scaled_consumption_3_" + str(mu) + "_" + str(sigma),
                                                               mu, sigma)
                              for sigma in sigma_c for mu in mu_c])

data_mu_sigma_gc['Model'] = np.repeat('CR', data_mu_sigma_gc.shape[0])

# %%

# C-R model with growth scaled with consumption, but growth is a function of consumption (2)

mu_c = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_c = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

data_mu_sigma_gc_2 = pd.concat([create_df_and_delete_simulations("cr_growth_scaled_consumption_new_model_" + str(mu) + "_" + str(sigma),
                                                               mu, sigma)
                              for sigma in sigma_c for mu in mu_c])

data_mu_sigma_gc_2['Model'] = np.repeat('CR', data_mu_sigma_gc_2.shape[0])

#%%

gLV_communities_fixed_growth = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_d_s_3.pkl")

mu_as = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_as = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

data_gLV_mu_sigma_unscaled = pd.concat([gLV_dynamics_df(simulation_data, mu, sigma)
                                        for simulation_data, mu, sigma in 
                                        zip(gLV_communities_fixed_growth, np.tile(mu_as, len(sigma_as)),
                                            np.repeat(sigma_as, len(mu_as)))])
    
data_gLV_mu_sigma_unscaled['Model'] = np.repeat('gLV', data_gLV_mu_sigma_unscaled.shape[0])

# %%

############################################# Phase diagrams ###############################################



