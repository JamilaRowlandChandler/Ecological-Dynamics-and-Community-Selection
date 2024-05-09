# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:19:35 2024

@author: jamil
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy
import itertools
import sys

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/model_modules')
from model_classes import gLV

################### Functions ############################

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

def calculate_les(gLV_object,lineages):
    
    le_lineages = \
        {'lineage ' + str(lineage) : lyapunov_exponents(gLV_object,'lineage ' + str(lineage))
             for lineage in lineages}
    
    return le_lineages

def lyapunov_exponents(gLV_object,lineage,n=10,dt=7000,separation=1e-2,extinct_thresh=1e-4):
    
    #breakpoint()
    
    # Initialise list of max. lyapunov exponents
    log_d1d0_list = []
    
    # Set initial conditions as population abundances at the end of lineage simulations
    initial_conditions = gLV_object.ODE_sols[lineage].y[:,-1]
    
    # Set initial conditions of the original trajectory
    initial_conditions_no_sep = deepcopy(initial_conditions.reshape((len(initial_conditions),1)))
    
    # Set initial conditions of the perturbated communty as the same as the original trajectory
    initial_conditions_sep = deepcopy(initial_conditions.reshape((len(initial_conditions),1)))
    
    # Select an extant species
    species_to_perturbate = np.where(initial_conditions_sep > extinct_thresh)[0][0]
    # Perturbate the selected species by 'separation'. We now have a perturbated trajectory.
    initial_conditions_sep[species_to_perturbate,:] += separation
    
    gLV_object_copy = deepcopy(gLV_object)
     
    # Repeat process n times
    for step in range(n):
        
        #breakpoint()
         
        gLV_object_copy.simulate_community(np.arange(1),t_end = dt,init_cond_func=None,
                                         usersupplied_init_conds=initial_conditions_no_sep)
        
        # Get species abundances at the end of simulation from the original trajectory
        species_abundances_end = deepcopy(gLV_object_copy.ODE_sols['lineage 0'].y[:,-1])
        
        ############
        
        gLV_object_copy.simulate_community(np.arange(1),t_end = dt,init_cond_func=None,
                                         usersupplied_init_conds=initial_conditions_sep)
        
        # Get species abundances at the end of simulation from the original trajectory
        species_abundances_end_sep = deepcopy(gLV_object_copy.ODE_sols['lineage 0'].y[:,-1])
        
        # Calculated the new separation between the original and perturbated trajectory (d1)
        separation_dt = np.sqrt(np.sum((species_abundances_end - species_abundances_end_sep)**2))
        
        # Calculate the max. lyapunov exponent
        log_d1d0 = (1/dt)*np.log(np.abs(separation_dt/separation))
        # Add exponent to list
        log_d1d0_list.append(log_d1d0)
        
        # Reset the original trajectory's species abundances to the species abundances at dt.
        initial_conditions_no_sep = \
            species_abundances_end.reshape((len(species_abundances_end),1))
        # Reset the perturbated trajectory's species abundances so that the original
        #    and perturbated community are 'separation' apart.
        initial_conditions_sep = species_abundances_end + \
            (separation/separation_dt)*(species_abundances_end_sep-species_abundances_end)
        initial_conditions_sep = \
            initial_conditions_sep.reshape((len(initial_conditions_sep),1))
    # Calculate average max. lyapunov exponent
    #max_lyapunov_exponent = mean_std_deviation(np.array(log_d1d0_list))
    
    #return max_lyapunov_exponent
    
    return np.array(log_d1d0_list)

############################ Simulating and plotting community dynamics #############

chaos_im = np.load('chaos_09_005.npy')
oscillations_im = np.load('oscillations_09_005.npy')
community_dynamics_invasibility_005 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005_new.pkl")
stable_im = deepcopy(community_dynamics_invasibility_005['0.90.05']['49'][0].interaction_matrix)

chaos_dynamics = gLV(no_species = 50, growth_func = 'fixed', growth_args = None,
               interact_func = None,interact_args = {'mu_a':0.9,'sigma_a':0.05},
               usersupplied_interactmat = chaos_im)
chaos_dynamics.simulate_community(np.arange(5),t_end = 10000)
chaos_dynamics.calculate_community_properties(np.arange(5),from_which_time = 7000)

oscillations_dynamics = gLV(no_species = 50, growth_func = 'fixed', growth_args = None,
               interact_func = None,interact_args = {'mu_a':0.9,'sigma_a':0.05},
               usersupplied_interactmat = oscillations_im)
oscillations_dynamics.simulate_community(np.arange(5),t_end = 10000)
oscillations_dynamics.calculate_community_properties(np.arange(5),from_which_time = 7000)

stable_dynamics = gLV(no_species = 49, growth_func = 'fixed', growth_args = None,
               interact_func = None,interact_args = {'mu_a':0.9,'sigma_a':0.05},
               usersupplied_interactmat = stable_im)
stable_dynamics.simulate_community(np.arange(5),t_end = 10000)
stable_dynamics.calculate_community_properties(np.arange(5),from_which_time = 7000)

####### Plotting #########

sns.set_style('white')
fig, axs = plt.subplots(3,1,sharex=True,sharey=True,figsize=(5,7),layout='constrained')

colours_list = [['#ffe837ff','#e9a100ff','#b34b00ff'],
                ['#67c6fbff','#1d4bfaff','#001256fd'],
                ['#3db200ff','#1f5a00ff','#002802ff']]

for ax, simulation, colours in zip(axs.flatten(),[chaos_dynamics.ODE_sols['lineage 1'],
                                         oscillations_dynamics.ODE_sols['lineage 2'],
                                         stable_dynamics.ODE_sols['lineage 0']],
                                  colours_list):
    
    custom_cm = mpl.colors.LinearSegmentedColormap.from_list('custom_cm',
                                                             colours,
                                                             N=49)

    for i in range(49):
    
        ax.plot(simulation.t[50:],simulation.y[i,50:].T,
                color = custom_cm(i),linewidth=1.5)
        
    ax.set_xlim([2500-50,10000+50])
    ax.set_ylim([-0.01,0.4])
        
fig.supxlabel('time',fontsize=32)
fig.supylabel('Species abundances',fontsize=32)
plt.xticks([], [])
plt.yticks([], [])
sns.despine()

#plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/fluctuation_coefficient_comparison.png",
#            dpi=300,bbox_inches='tight')
plt.savefig("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/fluctuation_coefficient_comparison.png",
            dpi=300,bbox_inches='tight')

######

sns.set_style('white')
fig, axs = plt.subplots(1,1,sharex=True,sharey=True,figsize=(5,3),layout='constrained')

axs.plot(chaos_dynamics.ODE_sols['lineage 1'].t[50:],chaos_dynamics.ODE_sols['lineage 1'].y[31,50:].T,
         color='#e9a100ff',linewidth=2)
        
axs.set_xlim([4500-50,10000+50])
axs.set_ylim([-0.01,0.2])
        
plt.xticks([], [])
plt.yticks([], [])
sns.despine()

plt.savefig("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/single_chaos.png",
            dpi=300,bbox_inches='tight')

######################## Comparing re-invadability metric, fluctuation coefficients and lyapunov exponents #########

communities_fluctuation_coefficient = \
    [detect_fluctuations(deepcopy(community_object),lineages = np.arange(5), from_which_time = 7000) \
        for community_object in [chaos_dynamics,oscillations_dynamics,stable_dynamics]]
        
communities_les = \
    [calculate_les(deepcopy(community_object),lineages = np.arange(5)) \
        for community_object in [chaos_dynamics,oscillations_dynamics,stable_dynamics]]
        
invasibilities = [invasibility_l 
                  for community in [chaos_dynamics,oscillations_dynamics,stable_dynamics]
                      for invasibility_l in community.invasibility.values()]

#######################

sns.set_style('white')
hue_dict = {0:'#e9a100ff',1:'#001256fd',2:'#1f5a00ff'}

fig, axs = plt.subplots(1,3,sharey=True,figsize=(13,2.2),layout='constrained')

proportion_fluctuating = np.array(list(itertools.chain.from_iterable(communities_fluctuation_coefficient)))
community_label = np.array(['chaos','oscillations','stable'])
lineage_label = np.arange(3)

p1 = sns.stripplot(y=community_label,x=proportion_fluctuating[[1,7,10]],hue=lineage_label,
                   palette = hue_dict, ax=axs.flatten()[0],size=15)
axs[0].set_xlabel('Fluctuation coefficient',fontsize=24)
axs[0].set_yticklabels(['F','O','S'], size=24)
axs[0].get_legend().remove()
axs[0].tick_params(axis='x', which='major', labelsize=16)
p1.set_xlim([-0.05,1.05])
p1.set_xticks(range(2))

plot_les = np.concatenate([le for community in communities_les for le in community.values()])
community_type = np.concatenate((np.repeat('chaos',10),np.repeat('oscillations',10),np.repeat('stable',10)))
lineage_label = np.repeat(np.arange(3),10)

p2 = sns.pointplot(y=community_type,x=plot_les[np.concatenate((np.arange(10,20),np.arange(70,80),np.arange(100,110)))],
                   hue=lineage_label,errorbar='sd',markersize=15,
                   dodge=0.3,palette = hue_dict, ax=axs.flatten()[1])
#axs[1].ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
axs[1].set_xlabel('Max. lyapunov exponent',fontsize=24)
axs[1].set_yticklabels(['F','O','S'], size=24)  
axs[1].get_legend().remove()
axs[1].tick_params(axis='x', which='major', labelsize=16)

community_label = np.array(['chaos','oscillations','stable'])
lineage_label = np.arange(3)

p3 = sns.stripplot(y=community_label,x=np.array(invasibilities)[[1,7,10]],hue=lineage_label,
                   dodge=0.3,palette = hue_dict, ax=axs.flatten()[2], size = 15)
p3.set_xlim([-0.05,1.05])
axs[2].set_xlabel('Re-invadability',fontsize=24)
axs[2].set_yticklabels(['F','O','S'],size=24)
axs[2].get_legend().remove()
axs[2].tick_params(axis='x', which='major', labelsize=16)
p3.set_xticks(range(2))
     
#fig.supylabel('Ecological dynamics \n',fontsize=24)
sns.despine()

plt.savefig("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/metrics_comparison.png",
            dpi=300,bbox_inches='tight')
