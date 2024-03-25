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

dispersal_rate_df_list = [community_object_to_df2(community_obj,
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
    
dispersal_rate_df = pd.concat(dispersal_rate_df_list,ignore_index=True)
dispersal_rate_df['no_species'] = dispersal_rate_df['no_species'].astype(int)
dispersal_rate_df['survival_fraction'] = dispersal_rate_df['diversity']/dispersal_rate_df['no_species']

dispersal_rate_df['community_lineage_label'] = [str(int(dispersal_rate_df.iloc[i]['community'])) + \
                                                 str(int(dispersal_rate_df.iloc[i]['lineage'])) + \
                                                 str(int(dispersal_rate_df.iloc[i]['no_species'])) \
                                                     for i in range(dispersal_rate_df.shape[0])]

############################## Save files ############################

############ Home ###############

dispersal_rate_df.to_csv("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/varying_dispersal_rates.csv")
pickle_dump("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dispersal_rates.pkl",
            communities_migration_rates)

############################# Read files ##############################

############# Work #############

communities_migration_rates = pd.read_pickle("C:/Users/Jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dispersal_rates.pkl")
dispersal_rate_df = pd.read_csv("C:/Users/Jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/varying_dispersal_rates.csv",
                                index_col=0)

#######################################################

for no_species, communities in communities_migration_rates.items():
    
    print(no_species,'\n')
    
    for i, community_set in enumerate(communities):
        
        print('Commmunity ' + str(i),'\n')
        
        initial_conditions = np.stack(list(community_set[0].initial_abundances.values()),axis=1)
        no_lineages = initial_conditions.shape[1]
        
        community_set[0].simulate_community(t_end, 'Supply initial conditions',
                                              np.arange(no_lineages),
                                              array_of_init_conds=initial_conditions)

        
pickle_dump("C:/Users/Jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dispersal_rates_w_extinction.pkl",
            communities_migration_rates)        
        
communities_migration_rates_with_extinction_threshold = \
    pd.read_pickle("C:/Users/Jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dispersal_rates_w_extinction.pkl")

###############################################################

dispersal_rate_df_list = [community_object_to_df2(community_obj,
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
    
dispersal_rate_df = pd.concat(dispersal_rate_df_list,ignore_index=True)
dispersal_rate_df['no_species'] = dispersal_rate_df['no_species'].astype(int)
dispersal_rate_df['survival_fraction'] = dispersal_rate_df['diversity']/dispersal_rate_df['no_species']

dispersal_rate_df['community_lineage_label'] = [str(int(dispersal_rate_df.iloc[i]['community'])) + \
                                                 str(int(dispersal_rate_df.iloc[i]['lineage'])) + \
                                                 str(int(dispersal_rate_df.iloc[i]['no_species'])) \
                                                     for i in range(dispersal_rate_df.shape[0])]

#####################

og_fluctuating_lineages = dispersal_rate_df.iloc[np.where(\
                                            (dispersal_rate_df['dispersal'] == 1e-9) & \
                                            (dispersal_rate_df['invasibilities'] >= 0.6))][\
                                            'community_lineage_label']
                                                                                        
og_nonfluctuating_lineages = dispersal_rate_df.iloc[np.where(\
                                            (dispersal_rate_df['dispersal'] == 1e-9) & \
                                            (dispersal_rate_df['invasibilities'] < 0.6))][\
                                            'community_lineage_label']

original_dynamics = np.empty(dispersal_rate_df.shape[0],dtype=str)

for community_lineage_label in list(og_fluctuating_lineages):
    
    original_dynamics[np.where(dispersal_rate_df['community_lineage_label'] \
                                    == community_lineage_label)] = \
        'invadable'

for community_lineage_label in list(og_nonfluctuating_lineages):
    
    original_dynamics[np.where(dispersal_rate_df['community_lineage_label'] \
                                    == community_lineage_label)] = \
        'non_invadable'
                                                                                                                                                        
dispersal_rate_df['original_dynamics'] = original_dynamics

dispersal_rate_df.to_csv("C:/Users/Jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/varying_dispersal_rates_w_exinction")
dispersal_rate_df = pd.read_csv("C:/Users/Jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/varying_dispersal_rates_w_exinction",
                                index_col=0)

##############################################################################

colours = plt.cm.jet(np.linspace(0,1,2))

fig, ax = plt.subplots(1,1,figsize=(10,8),sharex=True,sharey=True,layout='constrained')
fig.suptitle("The effect of dispersal on ecological dynamics",fontsize=24)
fig.supxlabel('Dispersal rate (D)',fontsize=20)
fig.supylabel('Average invasibilty',fontsize=20)

for i, no_species in enumerate(np.unique(dispersal_rate_df['no_species'])):

    subfig = sns.lineplot(data=dispersal_rate_df.iloc[np.where(dispersal_rate_df['no_species'] == no_species)],
                 x='dispersal',y='invasibilities',hue='original_dynamics',
                 ax=ax,palette={'n':colours[0],'i':colours[1]})
    ax.set_xscale('log')
    
    if no_species == 49:
        
        handles, labels = ax.get_legend_handles_labels()
    
    ax.get_legend().remove()
    subfig.set(xlabel=None,ylabel=None)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    
plt.gcf().text(0.83, 0.6,'For all communities, \n invasibility = 0 \n when dispersal = 0',
               fontsize=16,horizontalalignment='center',
               verticalalignment='center',
               bbox=dict(facecolor='none', edgecolor='black'))

reduced_labels, label_position = np.unique(labels,return_index=True)
plt.legend([handles[label_pos] for label_pos in label_position],
           ['Invasibility > 0.6 \n at D = $10^{-9}$','Invasibility < 0.6 \n at D = $10^{-9}$'],
           fontsize=18,loc='upper right')

plt.savefig("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/dispersal_invasibility.png",
            dpi=300,bbox_inches='tight')

##########################

colours = plt.cm.jet(np.linspace(0,1,2))

fig, ax = plt.subplots(1,1,figsize=(10,7.5),sharex=True,sharey=True,layout='constrained')
fig.suptitle("The effect of dispersal on community diversity",fontsize=24)
fig.supxlabel('Dispersal rate (D)',fontsize=20)
fig.supylabel('Average survival fraction',fontsize=20)

for i, no_species in enumerate(np.unique(dispersal_rate_df['no_species'])):

    subfig = sns.lineplot(data=dispersal_rate_df.iloc[np.where(dispersal_rate_df['no_species'] == no_species)],
                 x='dispersal',y='survival_fraction',hue='original_dynamics',
                 ax=ax,palette={'n':colours[0],'i':colours[1]})
    ax.set_xscale('log')
    
    if no_species == 49:
        
        handles, labels = ax.get_legend_handles_labels()
    
    ax.get_legend().remove()
    subfig.set(xlabel=None,ylabel=None)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

reduced_labels, label_position = np.unique(labels,return_index=True)
plt.legend([handles[label_pos] for label_pos in label_position],
           ['Invasibility > 0.6 \n at D = $10^{-9}$','Invasibility < 0.6 \n at D = $10^{-9}$'],
           fontsize=18,loc='lower right')

plt.savefig("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/dispersal_survivalfraction.png",
            dpi=300,bbox_inches='tight')

##############################################################

colours = plt.cm.jet(np.linspace(0,1,2))

fig, ax = plt.subplots(2,1,figsize=(7.5,10),sharex=True,sharey=False,layout='constrained')
fig.suptitle("The effect of dispersal on ecological dynamics",fontsize=24)
fig.supxlabel('Dispersal rate (D)',fontsize=20)

for i, no_species in enumerate(np.unique(dispersal_rate_df['no_species'])):

    subfig = sns.lineplot(data=dispersal_rate_df.iloc[np.where(dispersal_rate_df['no_species'] == no_species)],
                 x='dispersal',y='invasibilities',hue='original_dynamics',
                 ax=ax[0],palette={'n':colours[0],'i':colours[1]})
    ax[0].set_xscale('log')
    
    if no_species == 49:
        
        handles, labels = ax[0].get_legend_handles_labels()
    
    ax[0].get_legend().remove()
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Average invasibility',fontsize=20)
    ax[0].xaxis.set_tick_params(labelsize=14)
    ax[0].yaxis.set_tick_params(labelsize=14)
    ax[0].set_ylim(-0.01,1.01)
    
for i, no_species in enumerate(np.unique(dispersal_rate_df['no_species'])):

    subfig = sns.lineplot(data=dispersal_rate_df.iloc[np.where(dispersal_rate_df['no_species'] == no_species)],
                 x='dispersal',y='survival_fraction',hue='original_dynamics',
                 ax=ax[1],palette={'n':colours[0],'i':colours[1]})
    ax[1].set_xscale('log')
    ax[1].get_legend().remove()
    ax[1].set_xlabel(None)
    ax[1].set_ylabel('Average survival fraction',fontsize=20)
    ax[1].xaxis.set_tick_params(labelsize=14)
    ax[1].yaxis.set_tick_params(labelsize=14)
    ax[1].set_ylim(-0.01,1.01)
       
       
ax[0].text(0.77, 0.75,'For all communities, \n invasibility = 0 \n when dispersal = 0',
               fontsize=16,horizontalalignment='center',
               verticalalignment='center',
               bbox=dict(facecolor='none', edgecolor='black'),
               transform=ax[0].transAxes)

reduced_labels, label_position = np.unique(labels,return_index=True)
plt.figlegend([handles[label_pos] for label_pos in label_position],
           ['Invasibility > 0.6 \n at D = $10^{-9}$','Invasibility < 0.6 \n at D = $10^{-9}$'],
           fontsize=18,bbox_to_anchor=(1.45, 0.5),loc='center right')

plt.savefig("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/dispersal_invasibility_survivalfraction.png",
            dpi=300,bbox_inches='tight')

########################################################################################
    
colours = plt.cm.jet(np.linspace(0,1,50))

fig, ax = plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True,layout='constrained')
fig.suptitle("The effect of dispersal on a single community's dynamics \n",fontsize=16)
fig.supxlabel('time (t)',fontsize=14)
fig.supylabel('Species abundance',fontsize=14)

for i in range(49):
    
    ax[0].plot(communities_migration_rates['49 species'][-1][1].ODE_sols['lineage 0'].t,
             communities_migration_rates['49 species'][-1][1].ODE_sols['lineage 0'].y[i,:].T,
             color=colours[i])
    
ax[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax[0].set_xlim(-100,7000)
ax[0].set_ylim(0,0.8)
ax[0].set_title('dispersal = 1e-9',fontsize=14)

for i in range(49):
    
    ax[1].plot(communities_migration_rates['49 species'][-1][0].ODE_sols['lineage 0'].t,
             communities_migration_rates['49 species'][-1][0].ODE_sols['lineage 0'].y[i,:].T,
             color=colours[i])

ax[1].set_title('no dispersal',fontsize=14)
ax[1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax[1].set_xlim(-100,7000)
ax[1].set_ylim(0,0.75)

plt.savefig("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/one_community_dispersal01e-9.png",
            dpi=300,bbox_inches='tight')

##########################################################

community0_49species_dispersal0 = communities_migration_rates_with_extinction_threshold['49 species'][-1][0]

def gLV_ode_with_extinction_threshold(t,spec,growth_r,interact_mat,dispersal,
                                      extinct_thresh=1e-9):
    
    '''
    
    ODE system from generalised Lotka-Volterra model. 
    
    Removes species below some extinction threshold to cap abundances species can
    reinvade from and removes very small values that could cause numerical instability.
    This is useful when dispersal = 0.
    

    Parameters
    ----------
    t : float
        time.
    spec : float
        Species population dynamics at time t.
    growth_r : np.array of float64, size (n,)
        Array of species growth rates.
    interact_mat : np.array of float64, size (n,n)
        Interaction maitrx.
    dispersal : float
        Dispersal or migration rate.
    extinct_thresh : float
        Extinction threshold.

    Returns
    -------
    dSdt : np.array of float64, size (n,)
        array of change in population dynamics at time t aka dS/dt.

    '''
    
    #breakpoint()
    
    spec[spec < extinct_thresh] = 0 # set species abundances below extinction threshold to 0
    
    dSdt = np.multiply(1 - np.matmul(interact_mat,spec), growth_r*spec) + dispersal
    
    return dSdt

result = solve_ivp(gLV_ode_with_extinction_threshold,[0,t_end],
                   community0_49species_dispersal0.initial_abundances['lineage 0'],
                     args=(community0_49species_dispersal0.growth_rates,
                           community0_49species_dispersal0.interaction_matrix,
                           community0_49species_dispersal0.dispersal),
                     method='RK45')


d = -np.linalg.inv(community0_49species_dispersal0.interaction_matrix) @ np.ones(49).T
community0_49species_dispersal0.interaction_matrix @ d

d = np.ones(49)
cooperative_interaction_matrix = community0_49species_dispersal0.interaction_matrix
np.fill_diagonal(cooperative_interaction_matrix,-community0_49species_dispersal0.mu_a * 49 * 0.99)
cooperative_interaction_matrix @ d

result = solve_ivp(gLV_ode_with_extinction_threshold,[0,1000],
                   community0_49species_dispersal0.initial_abundances['lineage 0'],
                     args=(community0_49species_dispersal0.growth_rates,
                           -cooperative_interaction_matrix,
                           1e-9),
                     method='RK45')

plt.plot(result.t,result.y.T)
plt.ylim(3.04,3.08)
plt.xlim(150,155)

cooperative_interaction_matrix2 = 0.5*np.random.randn(50,50)

for i in range(50):
    
    cooperative_interaction_matrix2[i,i] = -(np.sum(np.abs(cooperative_interaction_matrix2[i,:])) - \
                                                   np.abs(cooperative_interaction_matrix2[i,i]))
    
result2 = solve_ivp(gLV_ode_with_extinction_threshold,[0,1000],
                   0.5 + 0.1*np.random.randn(50),
                     args=(np.ones(50),
                           -cooperative_interaction_matrix2,1e-5),
                     method='RK45')

plt.plot(result2.t,result2.y.T)

cooperative_interaction_matrix3 = 0.9 + 0.15*np.random.randn(50,50)
cooperative_competitive = np.random.binomial(1,0.9,50*50).reshape((50,50))
cooperative_competitive[cooperative_competitive == 0] = -1

cooperative_interaction_matrix3 *= cooperative_competitive
np.fill_diagonal(cooperative_interaction_matrix3,1)

result3 = solve_ivp(gLV_ode_with_extinction_threshold,[0,1000],
                   0.5 + 0.1*np.random.randn(50),
                     args=(np.ones(50),
                           cooperative_interaction_matrix3,0),
                     method='RK45')

plt.plot(result3.t,result3.y.T)
#plt.ylim(0,10)
plt.xlim(800,1000)

plt.plot(communities_migration_rates_with_extinction_threshold['49 species'][-1][4].ODE_sols['lineage 0'].t,
         communities_migration_rates_with_extinction_threshold['49 species'][-1][4].ODE_sols['lineage 0'].y.T)

community_obj = communities_migration_rates_with_extinction_threshold['49 species'][9][3]
community_parm_obj = community_parameters(community_obj.no_species,
                                          'fixed', None,
                                          None, interact_dist,
                                          None, community_obj.interaction_matrix,
                                          community_obj.dispersal)
community_gLV_obj = gLV(community_parm_obj, 10000, usersupplied_init_cond=community_obj.initial_abundances['lineage 1'])
community_gLV_obj.detect_invasibility(7000)

plt.plot(communities_migration_rates_with_extinction_threshold['13 species'][4][6].ODE_sols['lineage 2'].t,
         communities_migration_rates_with_extinction_threshold['49 species'][4][6].ODE_sols['lineage 2'].y.T)
#plt.xlim(7000,10000)













