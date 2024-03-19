# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:04:10 2024

@author: jamil
"""


##############################

# Home - cd Documents/PhD/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/
# Work - cd "Documents/PhD for github/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/" 

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
import itertools

from community_dynamics_and_properties_v2 import *

#################################################

####### Home #######

community_dynamics_invasibility_005 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005_2.pkl")
community_dynamics_invasibility_01 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_01_2.pkl")
community_dynamics_invasibility_015 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_015_2.pkl")
community_dynamics_invasibility_02 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_02_2.pkl")

####### Work #######

#community_dynamics_invasibility_005 = pd.read_pickle("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005_2.pkl")
#community_dynamics_invasibility_01 = pd.read_pickle("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_01_2.pkl")
#community_dynamics_invasibility_015 = pd.read_pickle("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_015_2.pkl")
#community_dynamics_invasibility_02 = pd.read_pickle("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_02_2.pkl")

######################################################

t_end = 10000
interact_dist = {'mu_a':0.9,'sigma_a':0.15}
dispersal_rates = np.array([0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-3,1e-1])

def vary_community_dispersal_rate(dispersal_rates,initial_community_dynamics,count=0):
    
    #initial_community_dynamics = community(no_species, 'fixed', None, 'random', interact_dist)
    #initial_community_dynamics.simulate_community(t_end, 'Default', np.arange(no_lineages),
    #                                              init_cond_func_name='Mallmin')
    
    print('Community ' + str(count) + ', ' + str(initial_community_dynamics.no_species) + ' species.',
          end='\n')
    
    initial_conditions = np.stack(list(initial_community_dynamics.initial_abundances.values()),axis=1)
    no_lineages = initial_conditions.shape[1]
    
    def initialise_and_simulate_community(dispersal_rate):
        
        community_dynamics = community(initial_community_dynamics.no_species,'fixed',
                                       None,None,interact_dist,
                                       usersupplied_interactmat=initial_community_dynamics.interaction_matrix,
                                       dispersal=dispersal_rate)
        community_dynamics.simulate_community(t_end, 'Supply initial conditions',
                                              np.arange(no_lineages),
                                              array_of_init_conds=initial_conditions)
        return community_dynamics
    
    community_dynamics_different_migration_rates = [deepcopy(initialise_and_simulate_community(dispersal)) \
                                                    for dispersal in dispersal_rates]
    
    return community_dynamics_different_migration_rates

communities_migration_rates = {key : [vary_community_dispersal_rate(dispersal_rates,community_dynamics,count) \
                               for count, community_dynamics in enumerate(communities_no_species)] \
                               for key, communities_no_species in community_dynamics_invasibility_015['0.90.15'].items()}
    
dispersal_rate_df_list = [community_object_to_df(community_obj,community_label=i,dispersal=True) \
                             for communities_no_species in communities_migration_rates.values() \
                                 for i, community_different_dispersal in enumerate(communities_no_species) \
                                     for community_obj in community_different_dispersal]
    
dispersal_rate_df = pd.concat(dispersal_rate_df_list,ignore_index=True)
dispersal_rate_df['no_species'] = dispersal_rate_df['no_species'].astype(int)
dispersal_rate_df['survival_fraction'] = dispersal_rate_df['diversity']/dispersal_rate_df['no_species']

###

dispersal_rate_df_list2 = [community_object_to_df2(community_obj,
                                                   community_attributes = ['mu_a',
                                                        'sigma_a','dispersal',
                                                        'no_species',
                                                        'no_unique_compositions',
                                                        'unique_composition_label',
                                                        'diversity','fluctuations',
                                                        'invasibilities'],
                                                   community_label=i) \
                             for communities_no_species in communities_migration_rates.values() \
                                 for i, community_different_dispersal in enumerate(communities_no_species) \
                                     for community_obj in community_different_dispersal]
    
dispersal_rate_df2 = pd.concat(dispersal_rate_df_list2,ignore_index=True)
dispersal_rate_df2['no_species'] = dispersal_rate_df2['no_species'].astype(int)
dispersal_rate_df2['survival_fraction'] = dispersal_rate_df2['diversity']/dispersal_rate_df2['no_species']

dispersal_rate_df2['community_lineage_label'] = [str(int(dispersal_rate_df2.iloc[i]['community'])) + \
                                                 str(int(dispersal_rate_df2.iloc[i]['lineage'])) + \
                                                 str(int(dispersal_rate_df2.iloc[i]['no_species'])) \
                                                     for i in range(dispersal_rate_df2.shape[0])]
fig, ax = plt.subplots(1,1)           
sns.lineplot(dispersal_rate_df2.iloc[np.where(dispersal_rate_df2['no_species'] == 49)],
             x='dispersal',y='invasibilities',hue='community_lineage_label',
             palette=sns.color_palette("icefire",n_colors=50),ax=ax)                                            
plt.xscale('log')
ax.get_legend().remove()

sns.lineplot(dispersal_rate_df2,x='dispersal',y='invasibilities',hue='no_species',estimator=None)                                            
plt.xscale('log')




dispersal_rate_49species = dispersal_rate_df2.iloc[np.where(dispersal_rate_df2['no_species'] == 49)]

fig, ax = plt.subplots(1,1)
fig.supxlabel('dispersal rate',fontsize=14)
fig.supylabel('invasibility',fontsize=14)

for i in np.unique(dispersal_rate_49species['community_label']):
    
    ax.plot(dispersal_rate_49species.iloc[np.where(dispersal_rate_49species['community_label'] == i)]['dispersal'],
            dispersal_rate_49species.iloc[np.where(dispersal_rate_49species['community_label'] == i)]['invasibilities'])
    plt.xscale('log')


