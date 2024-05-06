# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:57:18 2024

@author: Jamila
"""

#cd Documents/PhD/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/new_classes/

import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from itertools import chain

from model_classes import gLV_allee

from utility_functions import pickle_dump

###############################################################

def interactions_scaled_with_growth(no_species,growth_rates,
                                    average_degree,
                                    max_comp,comp_noise,
                                    max_coop,coop_noise,
                                    beta=7):
    
    # create i's
    species = np.arange(1,no_species+1)
    
    # calculate node weights, used to calculate the probability species i interacts with j.
    weights = \
        average_degree*((beta-2)/(beta-1))*((no_species/species)**(1/(beta-1)))
    
    # sort growth rates
    indices_sorted_growth_rates = np.flip(np.argsort(growth_rates))
    
    weights_reordered_by_growth = weights[indices_sorted_growth_rates]
    
    # calculate the probability species i interacts with j.
    probability_of_interactions = \
        (np.outer(weights_reordered_by_growth,
                  weights_reordered_by_growth)/np.sum(weights_reordered_by_growth)).flatten()
    
    # set probabilities > 1 to 1.
    probability_of_interactions[probability_of_interactions > 1] = 1
    
    are_species_interacting = \
        np.random.binomial(1,probability_of_interactions,
                           size=no_species*no_species).reshape((no_species,no_species))
    
    def competition_strength(growth_i,growth_j,
                             max_a,sigma_a,niche_width=0.5):
        
        expected_interaction_strength = max_a*np.exp(-((growth_i - growth_j)**2)/(2*niche_width**2))
        
        actual_interaction_strength = np.random.normal(expected_interaction_strength,sigma_a)
        
        return actual_interaction_strength
    
    def cooperation_strength(growth_i,growth_j,
                             max_a,sigma_a,niche_width=0.5):
        
        #expected_interaction_strength = \
        #    max_a / (1 + np.exp(-(growth_j - growth_i)/(niche_width)))
        
        expected_interaction_strength = max_a*(1-np.exp(-((growth_i - growth_j)**2)/(2*niche_width**2)))
        
        actual_interaction_strength = np.random.normal(expected_interaction_strength,sigma_a)
        
        return actual_interaction_strength
        
    growth_rates_i, growth_rates_j = np.meshgrid(growth_rates,growth_rates)

    competition_strengths = competition_strength(growth_rates_i,growth_rates_j,
                                                 max_comp,comp_noise)
    
    cooperation_strengths = competition_strength(growth_rates_i,growth_rates_j,
                                                 max_coop,coop_noise)
    
    competition_mat = are_species_interacting * competition_strengths
    np.fill_diagonal(competition_mat, 1)
    
    cooperation_mat = are_species_interacting * cooperation_strengths
    np.fill_diagonal(cooperation_mat, 0)
    
    return [competition_mat, cooperation_mat]

def species_interaction_effect():
    
    pass

no_species = 50
no_lineages = 5

growth_rates_uniform = np.linspace(0.1,2,no_species)

growth_rates_clustered_weakstrong = np.random.rand(no_species)
growth_rates_clustered_weakstrong[:np.round(no_species/2).astype(int)] = \
    0.3 + 0.1*growth_rates_clustered_weakstrong[:np.round(no_species/2).astype(int)]
growth_rates_clustered_weakstrong[np.round(no_species/2).astype(int):] = \
    1.5 + 0.3*growth_rates_clustered_weakstrong[np.round(no_species/2).astype(int):]

competition_matrix1, cooperation_matrix1 = \
    interactions_scaled_with_growth(no_species, growth_rates_uniform, 10, 1, 0.15, 1, 0)
    
competition_matrix2, cooperation_matrix2 = \
    interactions_scaled_with_growth(no_species, growth_rates_clustered_weakstrong, 10, 1, 0.15, 1, 0.15)
    
for i in range(no_species):
    
    monoculture_dynamics = gLV_allee(no_species = 1, growth_func = None, growth_args = None,
                                     competition_func = None, competition_args = None,
                                     cooperation_func = None, cooperation_args = None,
                                     usersupplied_growth = growth_rates_uniform[i],
                                     usersupplied_competition = np.array([-1]).reshape((1,1)),
                                     usersupplied_cooperatition = np.array([0]).reshape((1,1)))
    monoculture_dynamics.simulate_community(np.arange(no_lineages),t_end = 10000)
    
    for j in range(no_species):
        
        if i is not j:
            
            pass
    
    