# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:49:48 2024

@author: Jamila
"""

##############################

# cd Documents/PhD/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
import itertools

from community_dynamics_and_properties_v2 import *

#################################################

community_dynamics_invasibility_005 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005.pkl")
community_dynamics_invasibility_01 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_01.pkl")
community_dynamics_invasibility_015 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_015.pkl")
community_dynamics_invasibility_02 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_02.pkl")

#################################################

community_df_list = [community_object_to_df(community_obj,no_unique_compositions=False) \
                     for community_dict in [community_dynamics_invasibility_005,community_dynamics_invasibility_01,
                                            community_dynamics_invasibility_015,community_dynamics_invasibility_02] \
                         for community_interaction_dist in community_dict.values() \
                             for communities_no_species in community_interaction_dist.values() \
                                 for community_obj in communities_no_species]
    
communities_df = pd.concat(community_df_list,ignore_index=True)

###############################################

def proportion_communities_invadable(data):
    
    return np.count_nonzero(data > 0.6)/len(data)


groupby_invasibilities = communities_df.groupby(['mu_a','sigma_a','no_species'])['invasibilities'].apply(proportion_communities_invadable)
groupby_invasibilities = groupby_invasibilities.to_frame()
groupby_invasibilities.reset_index(inplace=True)

groupby_invasibilities_005 = groupby_invasibilities.iloc[np.where(groupby_invasibilities['sigma_a'] == 0.05)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

sns.heatmap(groupby_invasibilities_005)

groupby_invasibilities_01 = groupby_invasibilities.iloc[np.where(groupby_invasibilities['sigma_a'] == 0.1)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

sns.heatmap(groupby_invasibilities_01)

groupby_invasibilities_015 = groupby_invasibilities.iloc[np.where(groupby_invasibilities['sigma_a'] == 0.15)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

sns.heatmap(groupby_invasibilities_015)

groupby_invasibilities_02 = groupby_invasibilities.iloc[np.where(groupby_invasibilities['sigma_a'] == 0.2)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

sns.heatmap(groupby_invasibilities_02)

##################################

groupby_invasibilities2 = communities_df.groupby(['mu_a','sigma_a','no_species'])['invasibilities'].mean()
groupby_invasibilities2 = groupby_invasibilities2.to_frame()
groupby_invasibilities2.reset_index(inplace=True)

groupby_invasibilities2_005 = groupby_invasibilities2.iloc[np.where(groupby_invasibilities2['sigma_a'] == 0.05)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

sns.heatmap(groupby_invasibilities2_005)

groupby_invasibilities2_01 = groupby_invasibilities2.iloc[np.where(groupby_invasibilities2['sigma_a'] == 0.1)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

sns.heatmap(groupby_invasibilities2_01)

groupby_invasibilities2_015 = groupby_invasibilities2.iloc[np.where(groupby_invasibilities2['sigma_a'] == 0.15)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

sns.heatmap(groupby_invasibilities2_015)

groupby_invasibilities2_02 = groupby_invasibilities2.iloc[np.where(groupby_invasibilities2['sigma_a'] == 0.2)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

sns.heatmap(groupby_invasibilities2_02)

