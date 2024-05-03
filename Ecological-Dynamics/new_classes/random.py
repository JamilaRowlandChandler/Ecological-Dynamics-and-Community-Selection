# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:32:49 2024

@author: jamil
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import itertools

from model_classes import gLV

from utility_functions import community_object_to_df
from utility_functions import pickle_dump
from utility_functions import mean_std_deviation

###################

mu_a = 0.7
sigma_a = 0.1

gLV_dynamics = gLV(no_species = 49, growth_func = 'fixed', growth_args = None,
               interact_func = 'random',interact_args = {'mu_a':mu_a,'sigma_a':sigma_a})
gLV_dynamics.simulate_community(np.arange(5),t_end = 10000)

one_more_species_matrix = mu_a + sigma_a*np.random.randn(50,50)
np.fill_diagonal(one_more_species_matrix,1)
one_more_species_matrix[:-1,:-1] = gLV_dynamics.interaction_matrix

initial_abundances = np.vstack([ode_sol.y[:,-1] for ode_sol in gLV_dynamics.ODE_sols.values()])
#new_species_abundances = np.random.uniform(1e-8,2/50,5).reshape((5,1))
new_species_abundances = np.random.uniform(1e-8,1e-4,5).reshape((5,1))
initial_abundances = np.concatenate((initial_abundances,new_species_abundances),
                                    axis=1)

gLV_dynamics2 = gLV(no_species = 50, growth_func = 'fixed', growth_args = None,
               interact_func = None, interact_args = {'mu_a':mu_a,'sigma_a':sigma_a},
               usersupplied_interactmat = one_more_species_matrix)
gLV_dynamics2.simulate_community(np.arange(5),t_end = 10000,init_cond_func=None,
                                 usersupplied_init_conds=initial_abundances.T)

plt.plot(gLV_dynamics.ODE_sols['lineage 1'].t,gLV_dynamics.ODE_sols['lineage 1'].y.T)
plt.plot(gLV_dynamics2.ODE_sols['lineage 1'].t,gLV_dynamics2.ODE_sols['lineage 1'].y[-1,:].T)

print(gLV_dynamics2.ODE_sols['lineage 1'].y[-1,180:])

######################################

mu_a = 0.2
sigma_a = 0.1

growth_rates = np.random.binomial(1, 0.7, 49)

gLV_dynamics = gLV(no_species = 49, growth_func = None, growth_args = None,
               interact_func = 'random',interact_args = {'mu_a':mu_a,'sigma_a':sigma_a},
               usersupplied_growth = growth_rates)
gLV_dynamics.simulate_community(np.arange(5),t_end = 10000)
gLV_dynamics.calculate_community_properties(np.arange(5),from_which_time = 7000)

plt.plot(gLV_dynamics.ODE_sols['lineage 1'].t,gLV_dynamics.ODE_sols['lineage 1'].y.T)

print(gLV_dynamics.final_diversity)

############################################################

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

chaos_im = np.load('chaos_09_005.npy')
oscillations_im = np.load('oscillations_09_005.npy')

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

communities_fluctuation_coefficient = \
    [detect_fluctuations(deepcopy(community_object),lineages = np.arange(5), from_which_time = 7000) \
        for community_object in [chaos_dynamics,oscillations_dynamics]]
        
print(chaos_dynamics.final_diversity,'\n',oscillations_dynamics.final_diversity)

##########################

custom_YlGrBl = \
    mpl.colors.LinearSegmentedColormap.from_list('custom YlGBl',
                                                 ['#e9a100ff','#1fb200ff',
                                                  '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                 N=50)
sns.set_style('white')
fig, axs = plt.subplots(2,1,sharex=True,sharey=True,figsize=(7,7),layout='constrained')

for ax, simulation in zip(axs.flatten(),[chaos_dynamics.ODE_sols['lineage 3'],
                                         oscillations_dynamics.ODE_sols['lineage 0']]):
    
    for i in range(49):
    
        ax.plot(simulation.t[100:],simulation.y[i,100:].T,
                color = custom_YlGrBl(i),linewidth=2)
        
    ax.set_xlim([5000-50,10000+50])
    ax.set_ylim([-0.01,0.4])
        
fig.supxlabel('time',fontsize=32)
fig.supylabel('Species abundances',fontsize=32)
plt.xticks([], [])
plt.yticks([], [])
sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/fluctuation_coefficient_comparison.png",
            dpi=300,bbox_inches='tight')

################

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

communities_les = \
    [calculate_les(deepcopy(community_object),lineages = np.arange(5)) \
        for community_object in [chaos_dynamics,oscillations_dynamics]]
        
#######################

sns.set_style('white')
hue_dict = {**{i : '#e9a100ff' for i in range(4)},**{i : '#001256fd' for i in range(4,8)}}

fig, axs = plt.subplots(1,3,sharex=True,figsize=(20,6),layout='constrained')

community_label = np.concatenate((np.repeat('chaos',5),np.repeat('oscillations',5)))

communities_fluctuation_coefficient[0].pop(1)
communities_fluctuation_coefficient[1].pop(4)
proportion_fluctuating = list(itertools.chain.from_iterable(communities_fluctuation_coefficient))
lineage_label = np.arange(8)

p1 = sns.stripplot(x=community_label,y=proportion_fluctuating,hue=lineage_label,
                   palette = hue_dict, ax=axs.flatten()[0])
axs[0].set_ylabel('Fluctuation coefficient',fontsize=18)
axs[0].set_xticklabels(['Unstable and fluctuating','Oscillations'], size=18)
axs[0].get_legend().remove()
p1.set_ylim([-0.05,1.05])
p1.set_yticks(range(2))

plot_les = np.concatenate([le for community in communities_les for le in community.values()])
plot_les = np.delete(plot_les,[np.concatenate((np.arange(10,20),np.arange(90,100)))],None)
community_type = np.concatenate((np.repeat('chaos',40),np.repeat('oscillations',40)))
lineage_label = np.repeat(np.arange(10),8)

p2 = sns.pointplot(x=community_type,y=plot_les,hue=lineage_label,errorbar='sd',
                   dodge=0.3,palette = hue_dict, ax=axs.flatten()[1])
axs[1].set_ylabel('Max. lyapunov exponent',fontsize=18)
axs[1].set_xticklabels(['Unstable and fluctuating','Oscillations'], size=18)  
axs[1].get_legend().remove()

invasibilities = [invasibility_l 
                  for community in [chaos_dynamics,oscillations_dynamics]
                      for invasibility_l in community.invasibility.values()]
invasibilities.pop(1)
invasibilities.pop(-1)
community_label = np.concatenate((np.repeat('chaos',4),np.repeat('oscillations',4)))
lineage_label = np.arange(8)
p3 = sns.stripplot(x=community_label,y=invasibilities,hue=lineage_label,
                   dodge=0.3,palette = hue_dict, ax=axs.flatten()[2])
p3.set_ylim([-0.05,1.05])
axs[2].set_ylabel('Re-invadability',fontsize=18)
axs[2].set_xticklabels(['Unstable and fluctuating','Oscillations'], size=18)
axs[2].get_legend().remove()
p3.set_yticks(range(2))
     
fig.supxlabel('Ecological dynamics',fontsize=24)
sns.despine()

