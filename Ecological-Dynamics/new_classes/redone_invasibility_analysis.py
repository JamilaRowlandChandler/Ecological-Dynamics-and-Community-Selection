# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:37:41 2024

@author: Jamila
"""

# cd C:\Users\Jamila\Documents\PhD\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\new_classes

import numpy as np
import pandas as pd
from copy import deepcopy
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

from model_classes import gLV

from utility_functions import generate_distribution
from utility_functions import community_object_to_df
from utility_functions import pickle_dump

#######################################

def community_simulations_fixed_std(std):
 
    min_species = 4
    max_species = 50
    no_species_to_test = np.arange(min_species,max_species,3)
    
    interaction_distributions = generate_distribution([0.1,1.1], [std,std+0.04])
    
    no_communities = 10
    no_lineages = 5
     
    def interaction_strength_community_dynamics(no_species_to_test,mu_a,sigma_a,no_lineages,
                                                no_communities):
        
        def create_and_simulate_community(i,no_species):
            
            gLV_dynamics = gLV(no_species = no_species, growth_func = 'fixed', growth_args = None,
                           interact_func = 'random',interact_args = {'mu_a':mu_a,'sigma_a':sigma_a})
            gLV_dynamics.simulate_community(np.arange(no_lineages),t_end = 5000)
            
            initial_abundances = np.vstack([ode_sol.y[:,-1] for ode_sol in gLV_dynamics.ODE_sols.values()])
            
            gLV_dynamics.simulate_community(np.arange(5),t_end = 10000,init_cond_func=None,
                                             usersupplied_init_conds=initial_abundances.T)
            gLV_dynamics.calculate_community_properties(np.arange(no_lineages),from_which_time = 7000)
            
            if i == 0:
            
                print({'mu_a':gLV_dynamics.mu_a,'sigma_a':gLV_dynamics.sigma_a,
                       'no_species':gLV_dynamics.no_species}, end = '\n')
            
            return deepcopy(gLV_dynamics)
            
        output = {str(no_species) : 
                  [create_and_simulate_community(i,no_species) for i in range(no_communities)] \
                  for no_species in no_species_to_test}
            
        return output
    
    community_dynamics_interact_dist = {str(i_d['mu_a']) + str(i_d['sigma_a']) : \
                                        interaction_strength_community_dynamics(no_species_to_test,i_d['mu_a'],
                                                                                i_d['sigma_a'],no_lineages,no_communities) \
                                        for i_d in interaction_distributions}
    
    return community_dynamics_interact_dist
        
community_dynamics_invasibility_005 = community_simulations_fixed_std(0.05)
#pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005_new.pkl",
#            community_dynamics_invasibility_005)  
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005_new.pkl",
            community_dynamics_invasibility_005)        
      
community_dynamics_invasibility_01 = community_simulations_fixed_std(0.1)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_01_new.pkl",
            community_dynamics_invasibility_01)        
        
community_dynamics_invasibility_015 = community_simulations_fixed_std(0.15)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_015_new.pkl",
            community_dynamics_invasibility_015)        

community_dynamics_invasibility_02 = community_simulations_fixed_std(0.2)
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_02_new.pkl",
            community_dynamics_invasibility_02)

#######################################

community_dynamics_invasibility_005 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005_new.pkl")
community_dynamics_invasibility_01 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_01_new.pkl")
community_dynamics_invasibility_015 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_015_new.pkl")
community_dynamics_invasibility_02 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_02_new.pkl")

communities_dynamics_df = \
    pd.concat([community_object_to_df(community_object,community_label = i,
                                      community_attributes=['mu_a','sigma_a',
                                                            'no_species','final_diversity',
                                                            'invasibility']) \
               for community_dynamics in [community_dynamics_invasibility_005,community_dynamics_invasibility_01,
                                          community_dynamics_invasibility_015,community_dynamics_invasibility_02]
                   for communities_i_d in community_dynamics.values()
                       for communities_no_species in communities_i_d.values()
                           for i, community_object in enumerate(communities_no_species)],ignore_index=True)
      
communities_dynamics_df['no_species'] = communities_dynamics_df['no_species'].astype(int)        
communities_dynamics_df['survival_fraction'] = \
    communities_dynamics_df['final_diversity']/communities_dynamics_df['no_species']
    
##########################################

sns.set_style('white')

cmap = mpl.cm.plasma_r
bounds = np.append(np.sort(np.unique(communities_dynamics_df['no_species'])),52)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, axs = plt.subplots(4,4,sharex=True,sharey=True,figsize=(11.5,9),layout='constrained')
fig.suptitle('Effect of invasibility on community diversity \n',fontsize=28)
fig.supxlabel('Invasibility',fontsize=24)
fig.supylabel('Survival fraction',fontsize=24)

plt.gcf().text(0.5, 0.93,'Avg. interaction strength',fontsize=18,horizontalalignment='center',
               verticalalignment='center')

plt.gcf().text(0.85, 0.15, '0.2', fontsize=14,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.85, 0.37, '0.15', fontsize=14,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.85, 0.6, '0.1', fontsize=14,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.85, 0.8, '0.05', fontsize=14,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.89, 0.5, 'Std. in interaction strength', fontsize=18,
               horizontalalignment='center',verticalalignment='center',
               rotation=90,rotation_mode='anchor')

clb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs,shrink=0.8,
                   pad=0.16)
clb.ax.set_title('Initial number \n of species',fontsize=18,
                 pad=7.5)

mu_as = np.unique(communities_dynamics_df['mu_a'])[[0,6,8,9]]
no_species_test = len(np.unique(communities_dynamics_df['no_species']))
sigma_as = np.unique(communities_dynamics_df['sigma_a'])

sigma_a_plot = np.repeat(sigma_as,len(mu_as))
mu_a_plot = np.tile(mu_as,len(sigma_as))

for i, ax in enumerate(axs.flat):
    
    ax.axvline(0.6,color='grey',ls='--')
    subfig = sns.scatterplot(data=communities_dynamics_df.iloc[np.where((communities_dynamics_df['sigma_a'] == sigma_a_plot[i]) & \
                                                               (communities_dynamics_df['mu_a'] == mu_a_plot[i]))],
                          x='invasibility',y='survival_fraction',hue='no_species',
                          ax=ax,palette='plasma_r',hue_norm=norm,s=60)
    subfig.set(xlabel=None,ylabel=None)
    subfig.set_xticks(range(2))
    subfig.set_yticks(range(2))
    
    if i < 4:
        
        subfig.set_title(str(mu_a_plot[i]),fontsize=14,pad=4)
        
    ax.get_legend().remove()
    
sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/invasibility_survivalfraction_new.png",
            dpi=300,bbox_inches='tight')

###########################################
