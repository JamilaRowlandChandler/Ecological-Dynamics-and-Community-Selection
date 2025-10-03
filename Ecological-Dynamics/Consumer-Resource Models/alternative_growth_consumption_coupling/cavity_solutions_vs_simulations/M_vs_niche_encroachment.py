# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 12:25:10 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from copy import copy
from matplotlib import pyplot as plt

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling/cavity_solutions_vs_simulations')

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling")
from simulation_functions import generic_heatmaps, pickle_dump

from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine

# %%

def niche_encroach(n,
                   mu_c = 145, sigma_c = 1.6, mu_y = 1, sigma_y = 0.13):
   
    C = (mu_c/n + (sigma_c/np.sqrt(n))*np.random.randn(n**2)).reshape(n, n)
    G = (mu_y + sigma_y*np.random.randn(n**2)).reshape(n, n) * C
    
    quick_niche = 1 - cdist(G, C, 'cosine')
    
    #breakpoint()
    
    mask = np.zeros(quick_niche.shape, dtype=bool)
    np.fill_diagonal(mask, 1)
    between_niche = np.ma.masked_array(quick_niche, mask).max(axis=0).mean()
    
    within_niche = np.round(np.mean(np.diag(quick_niche)), 5)
    
    return {'M' : n, 'between niche' : between_niche,
            'within niche' : within_niche}

# %%
    
resource_pool_sizes = np.arange(50, 275, 25)
n_rep = 100

data = pd.DataFrame([niche_encroach(M) for M in resource_pool_sizes for _ in range(n_rep)])
data = data.groupby('M').apply('mean')
data.rename(columns = {'between niche' : 'Distance between niches',
                       'within niche' : 'Own niche similarity (baseline)'},
            inplace = True)

data.reset_index(inplace = True)

# %%

dfl = pd.melt(data[['M',
                    'Distance between niches',
                    'Own niche similarity (baseline)']], ['M'])

sns.set_style('ticks')

fig, ax = plt.subplots(1, 1, figsize = (2.75, 2.5), layout = 'constrained')

sns.lineplot(dfl, x = 'M', y = 'value', hue = 'variable', ax = ax,
             palette = ['black', 'gray'], linewidth = 2.5)

ax.set_xlabel('resource pool size, ' + r'$M$', fontsize = 10, weight = 'bold')
ax.set_ylabel('niche encroachment\n(cosine similarity between\nnearest consumer)',
              fontsize = 10)

ax.get_legend().remove()

ax.set_title("Increasing the resource pool size decreases\nniche encroachment",
             fontsize = 11, weight = 'bold')

sns.despine()

plt.show()