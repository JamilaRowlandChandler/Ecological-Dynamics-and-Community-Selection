# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:39:21 2024

@author: Jamila
"""

################## Packages #############

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import itertools

from community_dynamics_and_properties_v2 import *

########################################

def min_no_species_for_dynamics(min_species,max_species,no_communities,no_lineages,t_end,
                                dynamics_function,**dynamics_function_args):
    
    print(dynamics_function_args['interact_args'])
    
    no_species_range = np.arange(min_species,max_species)
    
    lineages = np.arange(no_lineages)
    
    for no_species in no_species_range:
        
        print(str(no_species),' species.',end='\n')
        
        no_species_with_dynamics = dynamics_function(lineages,no_species,no_communities,t_end,
                                                     **dynamics_function_args)
        
        if no_species_with_dynamics:
            
            print('Communities with ',str(no_species),' species can exhibit these dynamics.', end='\n')
            
            return {'Number of species':no_species,'Community':no_species_with_dynamics}
        
    print('No communities with ',str(min_species),'-',str(max_species),
          ' species exhibit these dynamics.', end='\n')

def find_invasibility(lineages,no_species,no_communities,t_end,**dynamics_function_args):
    
    for comm in range(no_communities):
        
        community_dynamics = community(no_species,"fixed", None,**dynamics_function_args)
        community_dynamics.simulate_community(t_end,"Default",lineages,init_cond_func_name="Mallmin")
        
        comm_invasibilities = np.array(list(community_dynamics.invasibilities.values()))
        
        if np.any(comm_invasibilities > 0.6):
            
            print('Community ',str(comm),' is invadable.',end='\n')
            return community_dynamics
        
    print('No invadable communities have been identified.')
            
        
def find_chaos(lineages,no_species,no_communities,t_end,**dynamics_function_args):
    
    for comm in range(no_communities):
        
        community_dynamics = community(no_species,"fixed", None,**dynamics_function_args)
        community_dynamics.simulate_community(t_end,"Default",lineages,init_cond_func_name="Mallmin")
        
        fluctuating_lineages = [lineage for lineage, fluctuations in community_dynamics.fluctuations.items() \
                               if fluctuations > 0.2]
            
        if fluctuating_lineages:
            
            fluctuating_lineage_dynamics = deepcopy(community_dynamics)
            
            fluctuating_lineage_dynamics.final_composition = \
                {lineage: community_dynamics.final_composition[lineage] for lineage in fluctuating_lineages}
                
            no_uniq_fluct_compositions, lineages_no = fluctuating_lineage_dynamics.unique_compositions()
            
            for lineage_no in lineages_no:
                
                lineage_period = find_period_ode_system(community_dynamics.ODE_sols['lineage ' + str(lineage_no)])
                
                if not np.isnan(lineage_period).all():
                
                    community_dynamics.calculate_lyapunov_exponents([lineage_no],separation=1e-3,
                                                          dt=np.nanmax(np.array(lineage_period)))
                    
                else:
                    
                    community_dynamics.calculate_lyapunov_exponents([lineage_no],separation=1e-3,
                                                          dt=10000)
                    
                le_lineage = list(deepcopy(community_dynamics.lyapunov_exponents).values())[0]
                
                predicted_dynamics = identify_ecological_dynamics(np.array(le_lineage[0]),
                                                                  np.array(le_lineage[1]))
                
                if predicted_dynamics == 'chaotic-like':
                    
                    print('Chaos has been identified in lineage ',str(lineage_no),'.', end='\n')
                    print('This community exhibits chaos.', end='\n')
                
                    return community_dynamics
          
    print('No lineages are chaotic.', end='\n')

def find_multiple_regimes(lineages,no_species,
               interact_func=None,interact_func_args=None,interaction_matrix=None):
    
    community_dynamics = community(no_species,
                                    "fixed", None,
                                    interact_func, interact_func_args, usersupplied_interactmat=interaction_matrix)
    community_dynamics.simulate_community(t_end,"Default",lineages,init_cond_func_name="Mallmin")
    
    if community_dynamics.no_unique_compositions > 1:
        
        print('This community has multiple compositions.', end='\n')
        
        return community_dynamics
    
    else:
    
        print('This community has one composition.', end='\n')


def find_multistability(lineages,no_species,
               interact_func=None,interact_func_args=None,interaction_matrix=None):
    
    community_dynamics = community(no_species,
                                    "fixed", None,
                                    interact_func, interact_func_args, usersupplied_interactmat=interaction_matrix)
    community_dynamics.simulate_community(t_end,"Default",lineages,init_cond_func_name="Mallmin")
    
    stable_lineages = [lineage for lineage, fluctuations in community_dynamics.fluctuations.items() \
                           if fluctuations == False]
        
    stable_lineage_dynamics = deepcopy(community_dynamics)
    
    stable_lineage_dynamics.final_composition = \
        {lineage: community_dynamics.final_composition[lineage] for lineage in stable_lineages}
        
    no_uniq_stable_compositions = stable_lineage_dynamics.unique_compositions()[0]
    
    if no_uniq_stable_compositions > 1:
        
        print('This community may be multistable.', end='\n')
        
        return community_dynamics
    
    else:
        
        print('This community is not multistable.', end='\n')
        
def find_fluctuations(lineages,no_species,t_end,**dynamics_function_args):
        
        community_dynamics = community(no_species,"fixed", None,**dynamics_function_args)
        community_dynamics.simulate_community(t_end,"Default",lineages,init_cond_func_name="Mallmin")
        
        fluctuating_lineages = [lineage for lineage, fluctuations in community_dynamics.fluctuations.items() \
                               if fluctuations > 0.2]
            
        if fluctuating_lineages:
            
            return community_dynamics



################################# Main ################################        

interaction_distributions = generate_distribution([0.8,1.1], [0.1,0.25])

min_species = 3
max_species = 15
no_communities = 30
no_lineages = 5
t_end = 10000

min_species_for_invasibility_per_dist = {(str(interact_dist['mu_a']) + str(interact_dist['sigma_a'])) :
                                  min_no_species_for_dynamics(min_species,max_species,
                                                     no_communities, no_lineages,
                                                     t_end, find_invasibility,
                                                     **{'interact_func_name':'random','interact_args':interact_dist}) \
                                  for interact_dist in interaction_distributions}
    
pickle_dump('min_species_for_invasibility_per_dist.pkl',min_species_for_invasibility_per_dist )
    
community_column = np.zeros((len(min_species_for_invasibility_per_dist)-1)*no_lineages)
lineage_column = np.tile(np.arange(no_lineages),len(min_species_for_invasibility_per_dist)-1)
no_species_column = []
invasibility_column = []
diversity_column = []

for community_info in min_species_for_invasibility_per_dist.values():
    
    if community_info is not None:
    
        no_species_column.append(np.repeat(community_info['Number of species'].astype(int),no_lineages))
        
        community_extracted = deepcopy(community_info['Community'])
        
        diversity_column.append(list(community_extracted.diversity.values()))
        invasibility_column.append(list(community_extracted.invasibilities.values()))

min_species_invasibility_df = pd.DataFrame(np.stack((np.concatenate(no_species_column),
                                                     community_column,lineage_column,
                                                     np.concatenate(invasibility_column),
                                                     np.concatenate(diversity_column))).T,
                                           columns=['No_Species','Community','Lineage',
                                              'Invasibility','Diversity'])
