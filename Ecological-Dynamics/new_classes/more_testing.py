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

from utility_functions import community_object_to_df
from utility_functions import pickle_dump

#######################################

chaotic_interaction_matrix = np.load('chaos_09_005.npy')
   
average_cooperations = np.array([0,0.05,0.1,0.2,0.3,0.5,0.7,0.9])
connectances = np.arange(0,1,step=0.1)

def gLV_allee_simulate_calculate_properties(cooperation, connectance):
    
    print('Cooperation = ', cooperation,', Connectance = ', connectance, end = '\n')


    gLV_allee_dynamics = gLV_allee(no_species = 50, growth_func = 'fixed', growth_args = None,
                                   competition_func = None, competition_args = None,
                                   cooperation_func = 'sparse',
                                   cooperation_args = {'mu_coop':cooperation,'sigma_coop':0.0,'connectance_coop':connectance},
                                   usersupplied_competition = chaotic_interaction_matrix,
                                   dispersal=0)
    gLV_allee_dynamics.simulate_community(np.arange(5), t_end = 5000)
    gLV_allee_dynamics.calculate_community_properties(np.arange(5),3000)
    
    return deepcopy(gLV_allee_dynamics)
    
cooperation_rescue = [{'Parameters':{'Cooperation':cooperation, 'Connectance': connectance},
                      'Simulations':[gLV_allee_simulate_calculate_properties(cooperation, connectance) \
                                     for i in range(10)]} \
                      for cooperation in average_cooperations \
                          for connectance in connectances]

df_list = [community_object_to_df(gLV_allee_object,community_label=i,
                                  community_attributes=['mu_coop','connectance_coop','no_species',
                                                        'no_unique_compositions','unique_composition_label',
                                                        'final_diversity','invasibility']) \
           for parameters_communities in cooperation_rescue \
               for i, gLV_allee_object in enumerate(parameters_communities['Simulations'])]
df = pd.concat(df_list)

pickle_dump('C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/cooperaction_rescue_attempt1.pkl',
            cooperation_rescue)













