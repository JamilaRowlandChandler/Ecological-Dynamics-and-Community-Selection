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
from itertools import chain

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

cmap = mpl.cm.viridis_r
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
clb.ax.set_title('Species pool \n size',fontsize=18,
                 pad=7.5)
spacing = np.linspace(bounds[0],bounds[-1],len(bounds))
add_on = np.diff(spacing)[0]/2
clb.set_ticks(spacing[:-1] + add_on)
clb.set_ticklabels(bounds[:-1])

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
                          ax=ax,palette='viridis_r',hue_norm=norm,s=60)
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

colourmap_base = mpl.colormaps['viridis_r'](0.99)
light_dark_range = np.linspace(0.85,0,256)
lighten_func = lambda val, i : val + i*(1-val)
colours_list = [tuple([lighten_func(val,i) for val in colourmap_base[:-1]]) + (colourmap_base[-1],)
                for i in light_dark_range]
cmap = mpl.colors.ListedColormap(colours_list)
norm = mpl.colors.PowerNorm(2.1, vmin = 0, vmax = 1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

sns.set_style('white')

fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(9,7),layout='constrained')

clb = plt.colorbar(sm,ax=ax,shrink=0.8,pad=0.1)
clb.ax.set_title('Invasibility',fontsize=18,
                 pad=7.5)

subfig = sns.scatterplot(data=communities_dynamics_df.iloc[np.where((communities_dynamics_df['sigma_a'] == 0.15) & \
                                                           (communities_dynamics_df['mu_a'] == 0.9))],
                      x='no_species',y='final_diversity',hue='invasibility',
                      ax=ax,palette=cmap,hue_norm=sm.norm,s=80,edgecolor='black',
                      linewidth=0.5)
subfig.set_xlabel('Species pool size',fontsize=20)
subfig.set_ylabel('Species diversity (at t = 7000-10000)',fontsize=20)
subfig.set_title('Invasibility is a good measure of \n ecological dynamics',fontsize=28,
                 pad=10)
subfig.set_xticks(np.arange(np.min(communities_dynamics_df['no_species']),
                            np.max(communities_dynamics_df['no_species'])+3,3))
ax.set_ylim([0,np.max(communities_dynamics_df['no_species'])])
ax.get_legend().remove()
sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/invasibility_good_metric.png",
            dpi=300,bbox_inches='tight')

################## Retesting fluctuation coefficient ######################

def detect_fluctuations(gLV_object,lineages,from_which_time):
    
    fluct_prop_lineages = \
        [fluctuation_coefficient(gLV_object.ODE_sols['lineage ' + str(lineage)],from_which_time)
             for lineage in lineages]
    
    return fluct_prop_lineages

def fluctuation_coefficient(ode_sol,from_which_time,fluctuation_thresh = 5e-2):
    
    t_start_index = np.where(ode_sol.t >= from_which_time)[0]
    
    average_variation_coeff_per_spec = \
        np.std(ode_sol.y[:,t_start_index],axis=1)/np.mean(ode_sol.y[:,t_start_index],axis=1)
    
    # find the species with the average CV greater than the flucutation threshold
    species_fluctutating = np.count_nonzero(average_variation_coeff_per_spec > fluctuation_thresh)
    
    # find the proportion of species with fluctuating dynamics in the whole community.
    fluct_prop = species_fluctutating/ode_sol.y.shape[0]
    
    return fluct_prop

##########################################

communities_fluctuation_coefficient = \
    [detect_fluctuations(deepcopy(community_object),lineages = np.arange(5), from_which_time = 7000) \
        for communities_no_species in community_dynamics_invasibility_015['0.90.15'].values()
            for community_object in communities_no_species]

communities_fluctuation_coefficient = list(chain.from_iterable(communities_fluctuation_coefficient))

##########################

colourmap_base = mpl.colormaps['plasma_r'](0.6)
light_dark_range = np.linspace(0.85,0,256)
lighten_func = lambda val, i : val + i*(1-val)
colours_list = [tuple([lighten_func(val,i) for val in colourmap_base[:-1]]) + (colourmap_base[-1],)
                for i in light_dark_range]
cmap = mpl.colors.ListedColormap(colours_list)
norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

sns.set_style('white')

fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(9,7),layout='constrained')

clb = plt.colorbar(sm,ax=ax,shrink=0.8,pad=0.1)
clb.ax.set_title('Fluctuations',fontsize=18,
                 pad=7.5)

communities_dynamics_df['fluctuations'] = np.nan
communities_dynamics_df.loc[(communities_dynamics_df['sigma_a'] == 0.15) & \
                            (communities_dynamics_df['mu_a'] == 0.9),'fluctuations'] = \
    communities_fluctuation_coefficient

subfig = sns.scatterplot(data=communities_dynamics_df.iloc[np.where((communities_dynamics_df['sigma_a'] == 0.15) & \
                                                           (communities_dynamics_df['mu_a'] == 0.9))],
                      x='no_species',y='final_diversity',hue='fluctuations',
                      ax=ax,palette=cmap,hue_norm=sm.norm,s=80,edgecolor='black',
                      linewidth=0.5)
subfig.set_xlabel('Species pool size',fontsize=20)
subfig.set_ylabel('Species diversity (at t = 7000-10000)',fontsize=20)
subfig.set_title('Invasibility is a good measure of \n ecological dynamics',fontsize=28,
                 pad=10)
subfig.set_xticks(np.arange(np.min(communities_dynamics_df['no_species']),
                            np.max(communities_dynamics_df['no_species'])+3,3))
ax.set_ylim([0,np.max(communities_dynamics_df['no_species'])])
ax.get_legend().remove()
sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/fluct_bad_metric.png",
            dpi=300,bbox_inches='tight')

############################################################

custom_YlGrBl = \
    mpl.colors.LinearSegmentedColormap.from_list('custom YlGBl',
                                                 ['#e9a100ff','#1fb200ff',
                                                  '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                 N=49)
    
sns.set_style('white')
fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(9,7),layout='constrained')

simulation = community_dynamics_invasibility_01['0.90.1']['49'][6].ODE_sols['lineage 0']

for i in range(49):
    
    ax.plot(simulation.t[:100],simulation.y[i,:100].T,
            color = custom_YlGrBl(i),linewidth=2)
    
plt.xlabel('time',fontsize=32)
plt.ylabel('Species abundances',fontsize=32)
plt.xlim([simulation.t[0]-50,simulation.t[100]+50])
plt.ylim([-0.001,0.55])
plt.xticks([], [])
plt.yticks([], [])
sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/chaotic_community_new.png",
            dpi=300,bbox_inches='tight')

################

sns.set_style('white')
fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(9,7),layout='constrained')
    
ax.plot(simulation.t[:100],simulation.y[12,:100].T,color = custom_YlGrBl(12),
        linewidth=2)

plt.xlim([simulation.t[0]-50,simulation.t[100]+50])
plt.ylim([-0.001,0.55])
plt.xticks([], [])
plt.yticks([], [])
sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/chaotic_community_new_1_spec.png",
            dpi=300,bbox_inches='tight')















