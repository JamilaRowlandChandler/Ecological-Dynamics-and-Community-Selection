# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:19:22 2024

@author: jamil
"""

# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\new_classes

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from model_classes import gLV

from utility_functions import community_object_to_df
from utility_functions import pickle_dump

#######################################

def custom_sparse_matrix(random_interaction_matrix,connectance):
    
    n = random_interaction_matrix.shape[0]
    
    are_species_interacting = \
        np.random.binomial(1,connectance,size=n*n).reshape((n,n))
        
    interact_mat = random_interaction_matrix * are_species_interacting
    
    np.fill_diagonal(interact_mat, 1)
    
    return interact_mat
    
def sparsity_effect_community_dynamics(i,connectances,
                                       no_species,mu_a,sigma_a,no_lineages,
                                       no_sparse_communities):
    
    print({'Community':i,'mu_a':mu_a,'sigma_a':sigma_a,
           'no_species':no_species}, end = '\n')
    
    gLV_dense = gLV(no_species = no_species, growth_func = 'fixed', growth_args = None,
                   interact_func = 'random',interact_args = {'mu_a':mu_a,'sigma_a':sigma_a})
    gLV_dense.connectance = 1
    gLV_dense.simulate_community(np.arange(no_lineages),t_end=5000)
    
    initial_abundances = np.vstack([ode_sol.y[:,-1] for ode_sol in gLV_dense.ODE_sols.values()])
    
    gLV_dense.simulate_community(np.arange(no_lineages),t_end = 10000,init_cond_func=None,
                                     usersupplied_init_conds=initial_abundances.T)
    gLV_dense.calculate_community_properties(np.arange(no_lineages),from_which_time=7000)
    
    def same_interaction_strength_different_connectance(connectance,no_sparse_communities):
        
        def create_and_simulate_sparse_community():
            
            sparse_interactmat = custom_sparse_matrix(gLV_dense.interaction_matrix,connectance)
        
            gLV_sparse = gLV(no_species = no_species, growth_func = 'fixed', growth_args = None,
                           interact_func = None,interact_args = {'mu_a':mu_a,'sigma_a':sigma_a,
                           'connectance':connectance},
                           usersupplied_interactmat = sparse_interactmat)
            gLV_sparse.simulate_community(np.arange(no_lineages),t_end=5000)
            
            initial_abundances = np.vstack([ode_sol.y[:,-1] for ode_sol in gLV_sparse.ODE_sols.values()])
            
            gLV_sparse.simulate_community(np.arange(no_lineages),t_end = 10000,init_cond_func=None,
                                             usersupplied_init_conds=initial_abundances.T)
            gLV_sparse.calculate_community_properties(np.arange(no_lineages),from_which_time=7000)
            
            return deepcopy(gLV_sparse)
        
        gLV_sparse_communities = [create_and_simulate_sparse_community() \
                                  for i in range(no_sparse_communities)]
            
        return gLV_sparse_communities
    
    output = {str(connectance) : \
              same_interaction_strength_different_connectance(connectance,no_sparse_communities) \
              for connectance in connectances}
        
    output['1.0'] = deepcopy(gLV_dense)
    
    return output

connectances = np.array([0.1,0.3,0.5,0.8,0.9])

species_range = np.arange(20,55,5)

interaction_distributions = [{'mu_a':0.7,'sigma_a':0.15},{'mu_a':0.7,'sigma_a':0.2},
                             {'mu_a':0.9,'sigma_a':0.1},{'mu_a':0.9,'sigma_a':0.2},
                             {'mu_a':1,'sigma_a':0.1},{'mu_a':1,'sigma_a':0.2},
                             {'mu_a':1.2,'sigma_a':0.1},{'mu_a':1.2,'sigma_a':0.2}]

no_lineages = 5
no_communities = 5
no_sparse_communities = 5

community_dynamics_with_sparsity = \
    {str(i_d['mu_a']) + str(i_d['sigma_a']) : {
        str(no_species) : {
            str('Community ') + str(i) : sparsity_effect_community_dynamics(i,connectances,
                                                   no_species,i_d['mu_a'],i_d['sigma_a'],
                                                   no_lineages,no_sparse_communities) \
                for i in range(no_communities)}
            for no_species in species_range}
        for i_d in interaction_distributions}
        
pickle_dump('C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_sparsity.pkl',
            community_dynamics_with_sparsity)

###########################################################################################################

def custom_nested_matrix(random_interaction_matrix,average_degree,beta=7):
    
    n = random_interaction_matrix.shape[0]
    
    #############################
    
    # create i's
    species = np.arange(1,n+1)
    
    # calculate node weights, used to calculate the probability species i interacts with j.
    weights = \
        average_degree*((beta-2)/(beta-1))*((n/species)**(1/(beta-1)))
    
    # calculate the probability species i interacts with j.
    probability_of_interactions = \
        (np.outer(weights,weights)/np.sum(weights)).flatten()
    
    # set probabilities > 1 to 1.
    probability_of_interactions[probability_of_interactions > 1] = 1
    
    are_species_interacting = \
        np.random.binomial(1,probability_of_interactions,size=n*n).reshape((n,n))
    
    interact_mat = random_interaction_matrix * are_species_interacting
    
    np.fill_diagonal(interact_mat, 1)
    
    return interact_mat
    
def degree_effect_community_dynamics(i,average_proportion_of_species_interacted,
                                       no_species,mu_a,sigma_a,no_lineages,
                                       no_nested_communities):
    
    print({'Community':i,'mu_a':mu_a,'sigma_a':sigma_a,
           'no_species':no_species}, end = '\n')
    
    gLV_dense = gLV(no_species = no_species, growth_func = 'fixed', growth_args = None,
                   interact_func = 'random',interact_args = {'mu_a':mu_a,'sigma_a':sigma_a})
    gLV_dense.simulate_community(np.arange(no_lineages),t_end=5000)
    
    initial_abundances = np.vstack([ode_sol.y[:,-1] for ode_sol in gLV_dense.ODE_sols.values()])
    
    gLV_dense.simulate_community(np.arange(no_lineages),t_end = 10000,init_cond_func=None,
                                     usersupplied_init_conds=initial_abundances.T)
    gLV_dense.calculate_community_properties(np.arange(no_lineages),from_which_time=7000)
 
    average_degrees = no_species * average_proportion_of_species_interacted
    
    def same_interaction_strength_different_average_degree(average_degree,no_nested_communities):
        
        def create_and_simulate_nested_community():
            
            nested_interactmat = custom_nested_matrix(gLV_dense.interaction_matrix,average_degree)
        
            gLV_nested = gLV(no_species = no_species, growth_func = 'fixed', growth_args = None,
                           interact_func = None,interact_args = {'mu_a':mu_a,'sigma_a':sigma_a,
                           'average_degree':average_degree},
                           usersupplied_interactmat = nested_interactmat)
            gLV_nested.simulate_community(np.arange(no_lineages),t_end=5000)
            
            initial_abundances = np.vstack([ode_sol.y[:,-1] for ode_sol in gLV_nested.ODE_sols.values()])
            
            gLV_nested.simulate_community(np.arange(no_lineages),t_end = 10000,init_cond_func=None,
                                             usersupplied_init_conds=initial_abundances.T)
            gLV_nested.calculate_community_properties(np.arange(no_lineages),from_which_time=7000)
         
            return deepcopy(gLV_nested)
        
        gLV_nested_communities = [create_and_simulate_nested_community() \
                                  for i in range(no_nested_communities)]
            
        return gLV_nested_communities
    
    output = {str(average_degree) : \
              same_interaction_strength_different_average_degree(average_degree,no_nested_communities) \
              for average_degree in average_degrees}
        
    output['1.0'] = [deepcopy(gLV_dense)]
    
    return output

###########################################

average_proportion_of_species_interacted = np.array([0.05,0.1,0.3,0.5,0.8,0.9])

species_range = np.arange(20,55,5)

interaction_distributions = [{'mu_a':0.7,'sigma_a':0.15},{'mu_a':0.7,'sigma_a':0.2},
                             {'mu_a':0.9,'sigma_a':0.1},{'mu_a':0.9,'sigma_a':0.2},
                             {'mu_a':1,'sigma_a':0.1},{'mu_a':1,'sigma_a':0.2},
                             {'mu_a':1.2,'sigma_a':0.1},{'mu_a':1.2,'sigma_a':0.2}]

no_lineages = 5
no_communities = 5
no_nested_communities = 5

community_dynamics_with_nestedness = \
    {str(i_d['mu_a']) + str(i_d['sigma_a']) : {
        str(no_species) : {
            str('Community ') + str(i) : degree_effect_community_dynamics(i,average_proportion_of_species_interacted,
                                                   no_species,i_d['mu_a'],i_d['sigma_a'],
                                                   no_lineages,no_nested_communities) \
                for i in range(no_communities)}
            for no_species in species_range}
        for i_d in interaction_distributions}
            
pickle_dump('C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_nestedness.pkl',
            community_dynamics_with_nestedness)

#############################################################################

for key0, communities_i_d in community_dynamics_with_sparsity.items():
    for key1, communities_no_species in communities_i_d.items():
        for key2, community_connectances in communities_no_species.items():
            for c, communities_connectance in community_connectances.items():
                
                if c == '1.0':
                    
                    community_dynamics_with_sparsity[key0][key1][key2][c][0].connectance = 1


community_dynamics_with_sparsity = \
    pd.read_pickle('C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_sparsity.pkl')

communities_with_sparsity_df = \
    pd.concat([community_object_to_df(community_object,community_label = community_label,
                                      community_attributes=['mu_a','sigma_a',
                                                            'no_species','connectance',
                                                            'final_diversity','invasibility']) \
                for communities_i_d in community_dynamics_with_sparsity.values()
                    for communities_no_species in communities_i_d.values()
                        for community_label, community_connectances in communities_no_species.items()
                            for communities_connectance in community_connectances.values()
                                for community_object in communities_connectance])
  
communities_with_sparsity_df['no_species'] = \
    communities_with_sparsity_df['no_species'].astype(int)        
communities_with_sparsity_df['survival_fraction'] = \
    communities_with_sparsity_df['final_diversity']/communities_with_sparsity_df['no_species']
        
communities_with_sparsity_df.to_csv('C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_with_sparsity.csv')

communities_with_sparsity_df = \
    pd.read_csv('C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_with_sparsity.csv',
                index_col=0)

#################################################################

sns.set_style('white')

cmap = mpl.cm.plasma_r
bounds = np.append(np.sort(np.unique(communities_with_sparsity_df['no_species'])),55)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, axs = plt.subplots(2,2,sharex=True,sharey=True,figsize=(8.5,7.5),layout='constrained')
fig.suptitle('Effect of invasibility on community diversity \n',fontsize=28)
fig.supxlabel('Invasibility',fontsize=24)
fig.supylabel('Survival fraction',fontsize=24)

plt.gcf().text(0.5, 0.91,'Connectance',fontsize=18,horizontalalignment='center',
               verticalalignment='center')

clb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs,shrink=0.8,
                   pad=0.11)
clb.ax.set_title('Species \n pool size',fontsize=18,pad=7.5)
spacing = np.linspace(bounds[0],bounds[-1],len(bounds))
add_on = np.diff(spacing)[0]/2
clb.set_ticks(spacing[:-1] + add_on)
clb.set_ticklabels(bounds[:-1])

connectances = np.flip(np.unique(communities_with_sparsity_df['connectance']))[[0,2,4,5]]
no_species_test = len(np.unique(communities_with_sparsity_df['no_species']))

sigma_a_plot = 0.2
mu_a_plot = 0.9

data_to_plot = \
    communities_with_sparsity_df.iloc[np.where((communities_with_sparsity_df['sigma_a'] == sigma_a_plot) & \
                                  (communities_with_sparsity_df['mu_a'] == mu_a_plot))]
#data_to_plot['community'] = \
#    [int(label.replace('Community ','')) for label in data_to_plot['community']]

for i, ax in enumerate(axs.flat):
    
    ax.axvline(0.6,color='grey',ls='--')
    subfig = sns.scatterplot(data = \
                             data_to_plot.iloc[np.where(data_to_plot['connectance'] == connectances[i])],
                             x='invasibility',y='survival_fraction',hue='no_species',
                             ax=ax,palette='plasma_r',hue_norm=norm,s=60)
    subfig.set(xlabel=None,ylabel=None)
    subfig.set_xticks(range(2))
    subfig.set_yticks(range(2))
    
    if i == 0:
        
        subfig.set_title('All species interact',fontsize=14,pad=4)
        
    else:
        
        subfig.set_title(str(connectances[i]),fontsize=14,pad=4)
      
    ax.get_legend().remove()
    
sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/invasibility_survivalfraction_connectance.png",
            dpi=300,bbox_inches='tight')

#################################################################

data_to_plot['community'] = \
    [int(label.replace('Community ','')) for label in data_to_plot['community']]
data_to_plot['community_label'] = data_to_plot['no_species'].astype(str) + data_to_plot['community'].astype(str)

p_corr = []
p_c_pval = []

for connectance in np.unique(data_to_plot['connectance']):

    for no_species in np.unique(data_to_plot['no_species']):
        
        data = data_to_plot.iloc[np.where((data_to_plot['connectance'] == connectance) &\
                                          (data_to_plot['no_species'] == no_species))]
        
        corr_test = pearsonr(data['invasibility'],data['survival_fraction'])
        
        p_corr.append(corr_test.statistic)
        p_c_pval.append(corr_test.pvalue)

connectances = np.repeat(np.unique(data_to_plot['connectance']),len(np.unique(data_to_plot['no_species'])))
species = np.tile(np.unique(data_to_plot['no_species']),len(np.unique(data_to_plot['connectance'])))

p_corr_df = pd.DataFrame([connectances,species,p_corr,p_c_pval]).T
p_corr_df.rename(columns={0:'connectance',1:'no_species',2:'correlation',3:'p_value'},
                 inplace=True)
p_corr_pivot = pd.pivot_table(p_corr_df,values='correlation',index='connectance',
                              columns='no_species')

sns.heatmap(p_corr_pivot)

###############

s_corr = []
s_c_pval = []

for connectance in np.unique(data_to_plot['connectance']):

    for no_species in np.unique(data_to_plot['no_species']):
        
        data = data_to_plot.iloc[np.where((data_to_plot['connectance'] == connectance) &\
                                          (data_to_plot['no_species'] == no_species))]
        
        corr_test = spearmanr(data['invasibility'],data['survival_fraction'])
        
        s_corr.append(corr_test.statistic)
        s_c_pval.append(corr_test.pvalue)

connectances = np.repeat(np.unique(data_to_plot['connectance']),len(np.unique(data_to_plot['no_species'])))
species = np.tile(np.unique(data_to_plot['no_species']),len(np.unique(data_to_plot['connectance'])))

s_corr_df = pd.DataFrame([connectances,species,s_corr,s_c_pval]).T
s_corr_df.rename(columns={0:'connectance',1:'no_species',2:'correlation',3:'p_value'},
                 inplace=True)
s_corr_pivot = pd.pivot_table(s_corr_df,values='correlation',index='connectance',
                              columns='no_species')

sns.heatmap(s_corr_pivot)

##############################

sns.scatterplot(data=s_corr_df,x='connectance',y='no_species',
                hue='correlation',size='correlation')