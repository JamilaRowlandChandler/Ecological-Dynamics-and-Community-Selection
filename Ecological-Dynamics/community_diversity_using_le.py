# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:51:33 2024

@author: jamil
"""
######################

# Jamila: for console - cd "Documents/PhD for github/Ecological dynamics and community selection"

#########################

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from copy import deepcopy

from community_dynamics_and_properties import *
from community_selection_protocols_and_analysis import *

################################## Functions ##########################

def community_properties(no_pools_per_dist,
                         no_lineages,
                         no_species,
                         interact_dist_minmax):
    
    community_dict = {}
    
    mu_maxmin = interact_dist_minmax['mean']
    std_maxmin = interact_dist_minmax['std']
    
    interaction_distributions = generate_distribution(mu_maxmin,std_maxmin)
    pools = np.arange(no_pools_per_dist)
    lineages = np.arange(no_lineages)
    
    for i, interact_dist in enumerate(interaction_distributions):
        
        interact_dist_dict = {}
        interact_dist_dict['Interaction distribution'] = {'mean':interact_dist[0],
                                                          'std':interact_dist[1]}
        
        for pool in pools:
            
            interact_dist_used = deepcopy(interact_dist)
        
            community_dynamics = community(no_species,
                                           'fixed', None,
                                           'random', {'mu_a':interact_dist_used[0],
                                                      'sigma_a':interact_dist_used[1]})
            community_dynamics.simulate_community('Default', lineages, init_cond_func_name='Mallmin')
            
            community_dynamics.repeat_lyapunov(lineages)
            
            interact_dist_dict['Species pool ' + str(pool)] = deepcopy(community_dynamics)
        
        community_dict['Distribution ' + str(i)] = interact_dist_dict
   
    return community_dict  
        
def community_dict_to_df(community_dict):
    
    community_selection_dynamics = []
    
    for distribution in community_dict.keys():
        
        for pool in community_dict[distribution].keys():
        
            if pool != "Interaction distribution":
                
                no_lineages = len(community_dict[distribution].keys())-1
                
                df = pd.DataFrame([community_dict[distribution][pool].diversity,
                                   community_dict[distribution][pool].fluctuations,
                                   community_dict[distribution][pool].final_composition,
                                   community_dict[distribution][pool].lyapunov_exponents]).T
                df.columns = ["Diversity","Fluctuations","Final Composition",
                              "Lyapunov Exponents"]
                
                df['Species pool'] = np.repeat(pool,no_lineages)
                df['Interaction mean'] = np.repeat(community_dict[distribution]['Interaction distribution']['mean'],no_lineages)
                df['Interaction std'] = np.repeat(community_dict[distribution]['Interaction distribution']['std'],no_lineages)
                df['No. Unique Compositions'] = np.repeat(community[distribution][pool].no_unique_compositions,lineages)
                
                community_selection_dynamics.append(df)

    community_selection_dynamics_df = pd.concat(community_selection_dynamics)
    
    community_selection_dynamics_df[['Lyapunov Exponent mean','Lyapunoc Exponent std']] = \
        pd.DataFrame(community_selection_dynamics_df['Lyapunov Exponents'].to_list())
    community_selection_dynamics_df.drop(columns=['Lyapunov Exponents'],inplace=True)
    
    community_selection_dynamics_df.reset_index(inplace=True)
    community_selection_dynamics_df = community_selection_dynamics_df.rename(columns={"index":"Lineage"})
    community_selection_dynamics_df["Lineage"] = \
        community_selection_dynamics_df["Lineage"].str.replace('lineage ','').astype(int)
        
    identify_ecological_dynamics(community_selection_dynamics_df,
                                 "Lyapunov Exponent mean",
                                 "Lyapunov Exponent std",
                                 predicted_dynamics_col="Predicted Dynamics")
    
    df = df[['Species pool','Interaction mean','Interaction std','No. Unique Compositions',
             'Lineage','Predicted Dynamics','Diversity','Lyapunov Exponent mean',
             'Lyapunov Exponent std','Fluctuations','Final Composition']]
    
    return df
        
############################################## Main #########################

no_pools_per_dist = 10
no_lineages = 10
no_species = 50
interact_dist_minmax = {'mean':[0.1,1.2],'std':[0.05,0.2]}

community_dynamics_dict = community_properties(no_pools_per_dist, no_lineages,
                                               no_species, interact_dist_minmax)  
        
        
        
        
        
        
        
        
        