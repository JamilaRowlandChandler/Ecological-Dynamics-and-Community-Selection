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
fig, ax = plt.subplots(1,1)           
sns.lineplot(dispersal_rate_df.iloc[np.where(dispersal_rate_df['no_species'] == 49)],
             x='dispersal',y='invasibilities',hue='community_lineage_label',
             palette=sns.color_palette("icefire",n_colors=50),ax=ax)                                            
plt.xscale('log')
ax.get_legend().remove()

sns.lineplot(dispersal_rate_df,x='dispersal',y='invasibilities',hue='no_species',estimator=None)                                            
plt.xscale('log')

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

a = communities_migration_rates['49 species'][-1][1].ODE_sols['lineage 0'].y

def gLV_ode_rounded(t,spec,growth_r,interact_mat,dispersal,extinct_thresh=1e-9):

    spec[spec < extinct_thresh] = 0
    
    dSdt = np.multiply(1 - np.matmul(interact_mat,spec), growth_r*spec) + dispersal
    
    return dSdt

community_obj_1eminus9 = communities_migration_rates['49 species'][-1][1]
community_obj_0 = communities_migration_rates['49 species'][-1][0]

result_1eminus9 = solve_ivp(gLV_ode_rounded,[0,t_end],community_obj_1eminus9.initial_abundances['lineage 0'],
                 args=(community_obj_1eminus9.growth_rates,community_obj_1eminus9.interaction_matrix,
                       community_obj_1eminus9.dispersal,1e-10),method='RK45',
                 t_eval=np.linspace(0,t_end,2000))

result_0 = solve_ivp(gLV_ode_rounded,[0,t_end],community_obj_0.initial_abundances['lineage 0'],
                 args=(community_obj_0.growth_rates,community_obj_0.interaction_matrix,
                       community_obj_0.dispersal,1e-10),method='RK45',
                 t_eval=np.linspace(0,t_end,2000))

colours = plt.cm.jet(np.linspace(0,1,50))

fig, ax = plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True,layout='constrained')
fig.supxlabel('time (t)',fontsize=14)
fig.supylabel('Species abundance',fontsize=14)

for i in range(49):
    
    ax[0].plot(communities_migration_rates['49 species'][-1][1].ODE_sols['lineage 0'].t,
             communities_migration_rates['49 species'][-1][1].ODE_sols['lineage 0'].y[i,:].T,
             color=colours[i])
    
ax[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax[0].set_xlim(-100,7000)
ax[0].set_ylim(0,0.8)
ax[0].set_title('not rounded',fontsize=14)

for i in range(49):
    
    ax[1].plot(result_1eminus9.t,result_1eminus9.y[i,:].T,
             color=colours[i])

ax[1].set_title('rounded',fontsize=14)
ax[1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax[1].set_xlim(-100,7000)
ax[1].set_ylim(0,0.75)

####

fig, ax = plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True,layout='constrained')
fig.supxlabel('time (t)',fontsize=14)
fig.supylabel('Species abundance',fontsize=14)

for i in range(49):
    
    ax[0].plot(communities_migration_rates['49 species'][-1][0].ODE_sols['lineage 0'].t,
             communities_migration_rates['49 species'][-1][0].ODE_sols['lineage 0'].y[i,:].T,
             color=colours[i])
    
ax[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax[0].set_xlim(-100,10000)
ax[0].set_ylim(0,0.8)
ax[0].set_title('not rounded',fontsize=14)

for i in range(49):
    
    ax[1].plot(result_0.t,result_0.y[i,:].T,
             color=colours[i])

ax[1].set_title('rounded',fontsize=14)
ax[1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax[1].set_xlim(-100,10000)
ax[1].set_ylim(0,0.75)

################

community_obj_1eminus3 = communities_migration_rates['49 species'][-1][4]

result_1eminus3 = solve_ivp(gLV_ode_rounded,[0,t_end],community_obj_1eminus3.initial_abundances['lineage 0'],
                 args=(community_obj_1eminus3.growth_rates,community_obj_1eminus3.interaction_matrix,
                       community_obj_1eminus3.dispersal,1e-9),method='RK45',
                 t_eval=np.linspace(0,t_end,2000))

colours = plt.cm.jet(np.linspace(0,1,50))

fig, ax = plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True,layout='constrained')
fig.supxlabel('time (t)',fontsize=14)
fig.supylabel('Species abundance',fontsize=14)

for i in range(49):
    
    ax[0].plot(communities_migration_rates['49 species'][-1][4].ODE_sols['lineage 0'].t,
             communities_migration_rates['49 species'][-1][4].ODE_sols['lineage 0'].y[i,:].T,
             color=colours[i])
    
ax[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax[0].set_xlim(-100,7000)
ax[0].set_ylim(0,0.8)
ax[0].set_title('not rounded',fontsize=14)

for i in range(49):
    
    ax[1].plot(result_1eminus3.t,result_1eminus3.y[i,:].T,
             color=colours[i])

ax[1].set_title('rounded',fontsize=14)
ax[1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax[1].set_xlim(-100,7000)
ax[1].set_ylim(0,0.75)


def gLV_sde_additive_noise(t,spec,
                           growth_r,interact_mat,dispersal,
                           rng,sigma_noise):
    
    '''
    
    SDE system of generalised Lotka-Volterra model with gaussian white noise

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
    dispersal : float.
        Dispersal or migration rate.
    rng : np.random Generator
        Used to generate gaussian white noise.
    dt : float
        Time step used by the solver.

    Returns
    -------
    dSdt : np.array of float64, size (n,)
        array of change in population dynamics at time t aka dS/dt.

    '''

    dSdt = (np.multiply(1 - np.matmul(interact_mat,spec), growth_r*spec) + dispersal) + \
            np.multiply(rng.normal(loc=0,scale=sigma_noise,size=len(spec)),spec)
    
    return dSdt

def gLV_sde_simulation(community_obj,lineage,t_end,sigma_noise=0.001):
    
    '''
    
    Simulate generalised Lotka-Volterra dynamics.

    Parameters
    ----------
    growth_r : np.array of float64, size (n,)
        Array of species growth rates.
    interact_mat : np.array of float64, size (n,n)
        Interaction maitrx.
    dispersal : float.
        Dispersal or migration rate.
    t_end : int or float
        Time for end of simulation.
    init_abundance : np.array of float64, size (n,)
        Initial species abundances.

    Returns
    -------
     OdeResult object of scipy.integrate.solve_ivp module
        (Deterministic) Solution to gLV ODE system.

    '''
    
    rng = np.random.default_rng()
    
    return solve_ivp(gLV_sde_additive_noise,[0,t_end],community_obj.initial_abundances[lineage],
                     args=(community_obj.growth_rates,community_obj.interaction_matrix,
                           community_obj.dispersal,rng,sigma_noise),method='RK45',
                     t_eval=np.linspace(0,t_end,2000))

result = gLV_sde_simulation(communities_migration_rates['25 species'][1][2],'lineage 0',10000)

result2 = gLV_sde_simulation(communities_migration_rates['49 species'][-1][0],'lineage 0',10000,
                             sigma_noise=0.01)
result3 = gLV_sde_simulation(communities_migration_rates['49 species'][-1][1],'lineage 0',10000,
                             sigma_noise=0.01)

plt.plot(result2.t,result2.y.T)
plt.plot(result3.t,result3.y.T)
