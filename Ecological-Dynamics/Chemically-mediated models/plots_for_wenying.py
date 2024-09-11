# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:58:43 2024

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

from CR_vs_gLV_functions import *

###################

# %%

def plot_gLV_dynamics(simulations, no_species = 50):
    
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom YlGBl',
                                                        ['#e9a100ff','#1fb200ff',
                                                         '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                        N = 50)
    
    fig, axs = plt.subplots(2,3,figsize=(11,5),layout='constrained')

    for d_i, (ax, data) in enumerate(zip(axs.flatten(), simulations)):
        
        if d_i == len(simulations) - 1: 
        
            for i in range(50):
                
                ax.plot(data.t[40:], data.y[i,40:].T, color = 'black', linewidth = 3.75)
                ax.plot(data.t[40:], data.y[i,40:].T, color = cmap(i), linewidth = 3)
                
        else: 
            
            for i in range(50):
                
                ax.plot(data.t[:-40], data.y[i,:-40].T, color = 'black', linewidth = 3.75)
                ax.plot(data.t[:-40], data.y[i,:-40].T, color = cmap(i), linewidth = 3)

    for ax in axs.flatten():
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    sns.despine()
     
    return fig, axs

#%%

gLV_communities_fixed_growth = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_d_s_2.pkl")

#%%

mu_as = [0.3, 0.5, 0.7, 0.9, 1.1]
sigma_as = [0.05, 0.1, 0.15, 0.2]

data_gLV_mu_sigma_unscaled = pd.concat([gLV_dynamics_df(simulation_data, mu, sigma)
                                        for simulation_data, mu, sigma in 
                                        zip(gLV_communities_fixed_growth, np.tile(mu_as, len(sigma_as)),
                                            np.repeat(sigma_as, len(mu_as)))])
    
data_gLV_mu_sigma_unscaled['Model'] = np.repeat('gLV', data_gLV_mu_sigma_unscaled.shape[0])

#%%

chaotic_gLV = gLV_communities_fixed_growth[8]['Simulations'][20]
#multistable_gLV_1 = gLV_communities_fixed_growth[8]['Simulations'][50]
#multistable_gLV_2 = gLV_communities_fixed_growth[8]['Simulations'][52]
multistable_gLV_1 = gLV_communities_fixed_growth[8]['Simulations'][69]
multistable_gLV_2 = gLV_communities_fixed_growth[8]['Simulations'][70]
limit_cycle_gLV = gLV_communities_fixed_growth[8]['Simulations'][9] 

stable_gLV = gLV_communities_fixed_growth[5]['Simulations'][0]

#%%

fig_p, axs_p = plot_gLV_dynamics([stable_gLV, chaotic_gLV, multistable_gLV_1, multistable_gLV_2,
                                  limit_cycle_gLV])

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/glv_simulations_for_wenying.svg",
            bbox_inches='tight')
