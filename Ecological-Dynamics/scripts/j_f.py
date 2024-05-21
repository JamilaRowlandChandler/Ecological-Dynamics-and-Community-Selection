# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:57:18 2024

@author: Jamila
"""

#cd Documents/PhD/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/scripts/

import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm import tqdm
from time import sleep
import sys
from itertools import chain

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/model_modules')
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
                             max_a,sigma_a,niche_width):
        
        expected_interaction_strength = max_a*np.exp(-((growth_i - growth_j)**2)/(2*niche_width**2))
        
        actual_interaction_strength = np.random.normal(expected_interaction_strength,sigma_a)
        
        return actual_interaction_strength
    
    def cooperation_strength(growth_i,growth_j,
                             max_a,sigma_a,niche_width):
        
        #expected_interaction_strength = \
        #    max_a / (1 + np.exp(-(growth_j - growth_i)/(niche_width)))
        
        expected_interaction_strength = max_a*(1-np.exp(-((growth_i - growth_j)**2)/(2*niche_width**2)))
        
        actual_interaction_strength = np.random.normal(expected_interaction_strength,sigma_a)
        
        return actual_interaction_strength
        
    growth_rates_i, growth_rates_j = np.meshgrid(growth_rates,growth_rates)

    competition_strengths = competition_strength(growth_rates_i,growth_rates_j,
                                                 max_comp,comp_noise,0.5*growth_rates_j)
    
    cooperation_strengths = cooperation_strength(growth_rates_i,growth_rates_j,
                                                 max_coop,coop_noise,0.5*growth_rates_j)
    
    competition_mat = are_species_interacting * competition_strengths
    np.fill_diagonal(competition_mat, 1)
    competition_mat[competition_mat < 1e-3] = 0
    
    cooperation_mat = are_species_interacting * cooperation_strengths
    np.fill_diagonal(cooperation_mat, 0)
    cooperation_mat[cooperation_mat < 1e-3] = 0
    
    return [competition_mat, cooperation_mat]

def species_interaction_effect(monoculture_dynamics,coculture_dynamics,species_index):
    
    monoculture_yeild = np.median(np.array([simulation.y[:,-1] \
                                            for simulation in monoculture_dynamics.ODE_sols.values()]))
    
    coculture_yeild = np.median(np.array([simulation.y[species_index,-1] \
                                            for simulation in coculture_dynamics.ODE_sols.values()]))
    
    coculture_effect = np.log2(coculture_yeild/monoculture_yeild)
    
    return coculture_effect

    
def model_test(no_species, growth_rates):
    
    no_lineages = 5

    competition_matrix, cooperation_matrix = \
        interactions_scaled_with_growth(no_species, growth_rates, 10, 1, 0, 1, 0)
        
    def simulate_monoculture(i):
        
        monoculture_dynamics = gLV_allee(no_species = 1, growth_func = None, growth_args = None,
                                         competition_func = None, competition_args = None,
                                         cooperation_func = None, cooperation_args = None,
                                         usersupplied_growth = growth_rates[i],
                                         usersupplied_competition = np.array([1]),
                                         usersupplied_cooperation = np.array([0]))
        monoculture_dynamics.simulate_community(np.arange(no_lineages),t_end = 10000)
        
        pbar.update(1)
        
        return monoculture_dynamics
    
    with tqdm(total=no_species) as pbar:
        monocultures = [deepcopy(simulate_monoculture(i)) for i in range(no_species)]
    
    def simulate_coculture(i,j):
        
        
        coculture_dynamics = gLV_allee(no_species = 2, growth_func = None, growth_args = None,
                                         competition_func = None, competition_args = None,
                                         cooperation_func = None, cooperation_args = None,
                                         usersupplied_growth = growth_rates[[i,j]],
                                         usersupplied_competition = competition_matrix[np.ix_([i,j],[i,j])],
                                         usersupplied_cooperation = cooperation_matrix[np.ix_([i,j],[i,j])])
        coculture_dynamics.simulate_community(np.arange(no_lineages),t_end = 10000)
        
        pbar.update(1)
        
        return coculture_dynamics
    
    cocultures = {}
    species_interaction_effect_matrix = np.zeros((no_species,no_species))
    
    with tqdm(total=no_species*no_species) as pbar:
        for i in range(no_species):
               
            for j in range(no_species):
                
                #breakpoint()
                
                species_combo = str(i) + str(j)
                
                if (i is not j) and \
                    (cocultures.get(species_combo) is None or cocultures.get(species_combo[::-1]) is None):
                    
                    cocultures[species_combo] = deepcopy(simulate_coculture(i, j))
                    
                    species_interaction_effect_matrix[i,j] = species_interaction_effect(monocultures[i],
                                                                                        cocultures[species_combo],
                                                                                        0)
                    species_interaction_effect_matrix[j,i] = species_interaction_effect(monocultures[j],
                                                                                        cocultures[species_combo],
                                                                                        1)
                    
    return monocultures, cocultures, species_interaction_effect_matrix

no_species = 10

growth_rates_uniform = np.linspace(0.1,2,no_species)

growth_rates_clustered_weakstrong = np.random.rand(no_species)
growth_rates_clustered_weakstrong[:np.round(no_species/2).astype(int)] = \
    0.3 + 0.1*growth_rates_clustered_weakstrong[:np.round(no_species/2).astype(int)]
growth_rates_clustered_weakstrong[np.round(no_species/2).astype(int):] = \
    1.5 + 0.3*growth_rates_clustered_weakstrong[np.round(no_species/2).astype(int):]

mono1, co1, effect1 = model_test(no_species, growth_rates_uniform)
mono2, co2, effect2 = model_test(no_species, growth_rates_clustered_weakstrong)


dynamics = gLV_allee(no_species = 10, growth_func = 'fixed', growth_args = None,
                     competition_func = 'random',
                     competition_args = {'mu_comp':0.9,'sigma_comp':0.15},
                     cooperation_func = 'random',
                     cooperation_args = {'mu_coop':0.1,'sigma_coop':0})
dynamics.simulate_community(np.arange(5),t_end = 10000)

plt.plot(dynamics.ODE_sols['lineage 0'].t,dynamics.ODE_sols['lineage 0'].y.T)

monoculture_dynamics = gLV_allee(no_species = 1, growth_func = None, growth_args = None,
                                 competition_func = None, competition_args = None,
                                 cooperation_func = None, cooperation_args = None,
                                 usersupplied_growth = 0.1,
                                 usersupplied_competition = np.array([1]),
                                 usersupplied_cooperation = np.array([0]))
monoculture_dynamics.simulate_community(np.arange(5),t_end = 10000)

plt.plot(monoculture_dynamics.ODE_sols['lineage 0'].t,
         monoculture_dynamics.ODE_sols['lineage 0'].y.T)


