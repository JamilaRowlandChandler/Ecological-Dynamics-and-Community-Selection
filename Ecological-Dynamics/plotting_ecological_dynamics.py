# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:59:26 2024

@author: jamil
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
import itertools

from community_dynamics_and_properties_v2 import *

#######################################################

chaotic_interaction_matrix = np.load('chaos_09_005.npy')
oscillating_interaction_matrix = np.load('oscillations_09_005.npy')
stable_interaction_matrix = np.load('stable_09_005.npy')

no_species = 50
no_lineages = 10
interaction_distribution = {'mu_a':0.9,'sigma_a':0.05}
t_end = 10000

chaotic_community = community(no_species, 'fixed', None, None, interaction_distribution,
                              usersupplied_interactmat=chaotic_interaction_matrix)
chaotic_community.simulate_community(t_end,'Default',np.arange(no_lineages),
                                      init_cond_func_name='Mallmin')

colours = plt.cm.jet(np.linspace(0,1,50))

fig, ax = plt.subplots(1,1)
fig.supxlabel('time (t)',fontsize=14)
fig.supylabel('Species abundance',fontsize=14)

for i in range(50):
    
    ax.plot(chaotic_community.ODE_sols['lineage 0'].t,
               chaotic_community.ODE_sols['lineage 0'].y[i,:].T,
             color=colours[i])

ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax.set_xlim(-100,10100)
ax.set_ylim(0,0.4)

fig, ax = plt.subplots(1,1)
fig.supxlabel('time (t)',fontsize=14)
fig.supylabel('Species abundance',fontsize=14)

ax.plot(chaotic_community.ODE_sols['lineage 0'].t,
        chaotic_community.ODE_sols['lineage 0'].y[5,:].T,
        color=colours[5])

ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax.set_xlim(-100,10100)
ax.set_ylim(0,0.4)

plt.savefig("Figures/large_chaotic_community_species5.png", dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1,1)
fig.supxlabel('time (t)',fontsize=14)
fig.supylabel('Species abundance',fontsize=14)

ax.plot(chaotic_community.ODE_sols['lineage 0'].t,
        chaotic_community.ODE_sols['lineage 0'].y[45,:].T,
        color=colours[45])

ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax.set_xlim(-100,10100)
ax.set_ylim(0,0.4)

plt.savefig("Figures/large_chaotic_community_species45.png", dpi=300, bbox_inches='tight')

find_period_ode_system(chaotic_community.ODE_sols['lineage 0'], 7000)

oscillating_community = community(no_species, 'fixed', None, None, interaction_distribution,
                              usersupplied_interactmat=oscillating_interaction_matrix)
oscillating_community.simulate_community(t_end,'Default',np.arange(no_lineages),
                                      init_cond_func_name='Mallmin')

stable_community = community(no_species, 'fixed', None, None, interaction_distribution,
                              usersupplied_interactmat=stable_interaction_matrix)
stable_community.simulate_community(t_end,'Default',np.arange(no_lineages),
                                      init_cond_func_name='Mallmin')

print(mean_std_deviation(list(itertools.chain(chaotic_community.diversity.values()))))

oscillating_chaos = [oscillating_community.diversity['lineage 3'],oscillating_community.diversity['lineage 9']]
oscillating_osc = deepcopy(oscillating_community.diversity)
for key in ['lineage 3','lineage 9']: oscillating_osc.pop(key)
oscillating_osc = list(itertools.chain(oscillating_osc.values()))
print(mean_std_deviation(oscillating_osc))
print(mean_std_deviation(oscillating_chaos))

print(mean_std_deviation(list(itertools.chain(stable_community.diversity.values()))))









