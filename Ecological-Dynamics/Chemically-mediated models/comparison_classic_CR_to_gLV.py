# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:34:32 2024

@author: jamil
"""

# %%

# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\Chemically-mediated models

########################

# %%

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
from scipy.stats import kendalltau
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
import pandas as pd
import seaborn as sns
import sys
from copy import deepcopy

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Lotka-Volterra models/model_modules')
from model_classes import gLV
from utility_functions import pickle_dump

# %%

def dCR_dt(t, var,
           no_species,
           growth, death, consumption, influx):

    species = var[:no_species]
    resources = var[no_species:]

    dSdt = species * (np.sum(growth * resources, axis=1) - death) + 1e-8

    dRdt = (resources * (influx - resources)) - \
        (resources * np.sum(growth.T * consumption * species, axis=1)) + 1e-8

    return np.concatenate((dSdt, dRdt))

# %%

def normal_distributed_parameters(mu, sigma, dims):

    return mu + sigma*np.random.randn(*dims)

# %%

def rescaled_detect_invasability(simulation_t, simulation_y,
                        t_start, extinct_thresh=1e-3):
    '''

    Detect the proportion of extant/surviving species in a community that can "reinvade"
    the community.
    THIS IS THE MAIN METRIC FOR IDENTIFYING HIGH-DIVERSITY FLUCTUATING COMMUNITIES.

    How the function works: 
        (1) Detect extant/surviving/non-extinct species.
        (2) Detect whether extant species have "fluctuating" dynamics using scipy's 
            find_peaks function. This will assess whether there are "peaks" in
            each species population dynamics. If a community is stable/dynamics
            are a flat line, there will be no peaks. If a community is fluctuating,
            then its population dynamics should have peaks.
    (A) (3) If no species have fluctuating dynamics, invasibility is set 
            to 0 and the function terminates.
    (B) (3) If some species have fluctuating dynamics, identify whether these species,
            after t_start, go below some baseline_abundance (this is lower
            than the extinction threshold). Record this time.
        (4) Of those species that reached low abundances, assess whether they 
            reinvaded (their abundances increased above basline_abundance after
                       the recorded time).
        (5) Calculate the proportion of extant/present species with fluctuating dynamics
        and can reinvade the community from low abundances

    Parameters
    ----------
    t_start : float
        Start time to detect re-invading species.
    extinct_thresh : float, optional
        Extinction threshold. The default is 1e-4.

    Returns
    -------
    proportion_fluctuating_reinvading_species : float
            Proportion of extant/present species with fluctuating dynamics
            and can reinvade the community from low abundances.

    '''

    # find the index of the start time to detect whether species can reinvade the community.
    t_start_index = np.where(simulation_t >= t_start)[0]

    # set baseline_abundance as slightly greater than the migration rate.
    baseline_abundance = 10**-4

    # identifying extant/surviving species between t_start and the end of simulations
    extant_species = np.any(
        simulation_y[:, t_start_index] > extinct_thresh, axis=1).nonzero()[0]

    # Identify which of the extant species have "fluctuating dynamics".
    fluctuating_species = extant_species[np.logical_not(np.isnan([find_normalised_peaks(simulation_y[spec, t_start_index])[0]
                                                                  for spec in extant_species]))]  # THIS IS KINDA WRONG

    # If there are species with fluctuating dynamics present
    if fluctuating_species.size > 0:

        # # find if and where species abundances dip below baseline_abundance.
        # Tuple entry 0 = species, Tuple entry 1 = index of the timepoint where their
        #   abundances dipped below baseline_abundance.
        when_fluctuating_species_are_lost = np.nonzero(simulation_y[fluctuating_species, :]
                                                       < baseline_abundance)  # THIS IS VERY WRONG

        # If species abundances dip below baseline_abundance
        if len(when_fluctuating_species_are_lost[0]) > 0:

            # Identify the species with abundances that dip below baseline_abundance
            #   and the first entry where the unique species was identified.
            unique_species, index = \
                np.unique(
                    when_fluctuating_species_are_lost[0], return_index=True)

            reinvading_species = np.array([np.any(simulation_y[
                fluctuating_species[when_fluctuating_species_are_lost[0][i]],
                when_fluctuating_species_are_lost[1][i]:]
                > baseline_abundance) for i in index])

            # count number of reinvading species
            no_reinvading_species = np.sum(reinvading_species)

            # calculate the proportion of extant species that can reinvade the system
            proportion_fluctuating_reinvading_species = no_reinvading_species / \
                len(extant_species)

        # If no species abundances dip below baseline_abundance, the proportion
        #   of species that can reinvade the system (proportion_fluctuating_reinvading_species)
        #   is set to 0.
        else:

            proportion_fluctuating_reinvading_species = 0

    # If no species have fluctuating dynamics, the proportion of species that
    #   can reinvade the system (proportion_fluctuating_reinvading_species)
    #   is set to 0.
    else:

        proportion_fluctuating_reinvading_species = 0

    return proportion_fluctuating_reinvading_species


#%%

def find_normalised_peaks(data):
    '''

    Find peaks in data, normalised by relative peak prominence. Uses functions
    from scipy.signal

    Parameters
    ----------
    data : np.array of floats or ints
        Data to identify peaks in.

    Returns
    -------
    peak_ind or np.nan
        Indices in data where peaks are present. Returns np.nan if no peaks are present.

    '''

    # Identify indexes of peaks using scipy.signal.find_peaks
    peak_ind, _ = find_peaks(data)

    # If peaks are present
    if peak_ind.size > 0:

        # get the prominance of peaks compared to surrounding data (similar to peak amplitude).
        prominences = peak_prominences(data, peak_ind)[0]
        # get peak prominances relative to the data.
        normalised_prominences = prominences/(data[peak_ind] - prominences)
        # select peaks from normalised prominences > 0.8
        peak_ind = peak_ind[normalised_prominences > 0.8]

    # If peaks are present after normalisation
    if peak_ind.size > 0:

        return peak_ind  # return indexes of peaks

    # If peaks are not present
    else:

        return np.array([np.nan])  # return np.nan
    
# %%
    
def fluctuation_coefficient(times, dynamics, extinction_threshold = 1e-3):
     
    last_500_t = np.argmax(times > 2500)
    final_diversity = np.any(dynamics[:, last_500_t:] > extinction_threshold, axis=1)

    extant_species = dynamics[final_diversity, last_500_t:]

    return np.count_nonzero(np.std(extant_species, axis=1)/np.mean(extant_species, axis=1) > 5e-2)

# %%

def thermodynamic_threshold_dynamics(mu, sigma, no_species = 100, no_resources = 100):
    
    influx = np.ones(no_resources)
    death = np.ones(no_species)
    
    no_communities = 25
    no_lineages = 5
    
    def community_mu_sigma(i, mu, sigma):
        
        print({'mu': mu, 'sigma' : sigma, 'Community' : i}, '\n')
        
        growth = np.abs(normal_distributed_parameters(1, sigma,
                                                    dims=(no_species, no_resources)))
        
        consumption = np.abs(normal_distributed_parameters(mu, sigma,
                                                    dims=(no_resources, no_species)))
   
        def simulate_community(no_species, no_resources,
                               death, influx, growth, consumption):
            
            initial_abundances = np.random.uniform(1e-8, 2/no_species, no_species)
            initial_concentrations = np.random.uniform(1e-8, 2/no_species, no_resources)
    
            first_simulation = solve_ivp(dCR_dt, [0, 500],
                                         np.concatenate(
                                             (initial_abundances, initial_concentrations)),
                                         args=(no_species, growth, death,
                                               consumption, influx),
                                         method='RK45', rtol=1e-14, atol=1e-14,
                                         t_eval=np.linspace(0,500,50))
    
            new_initial_conditions = first_simulation.y[:, -1]
    
            if np.any(np.log(np.abs(new_initial_conditions)) > 6) \
                or np.isnan(np.log(np.abs(new_initial_conditions))).any():
    
                return None
    
            else:
    
                final_simulation = solve_ivp(dCR_dt, [0, 3000],
                                             new_initial_conditions,
                                             args=(no_species, growth,
                                                   death, consumption, influx),
                                             method='RK45', rtol=1e-14, atol=1e-14,
                                             t_eval=np.linspace(0,3000,200))
    
                if np.any(np.log(np.abs(final_simulation.y[:, -1])) > 6) \
                    or np.isnan(np.log(np.abs(final_simulation.y[:, -1]))).any():
    
                    return None
    
                else:
    
                    species_reinvadability = rescaled_detect_invasability(final_simulation.t, final_simulation.y[:no_species, :],
                                                                          2000)
                    resource_reinvadability = rescaled_detect_invasability(final_simulation.t, final_simulation.y[:no_species, :],
                                                                           2000)
                    
                    species_fluctuations = fluctuation_coefficient(final_simulation.t, final_simulation.y[:no_species, :])
                    resource_fluctuations = fluctuation_coefficient(final_simulation.t, final_simulation.y[no_species:, :])
                    
                    last_500_t = np.argmax(final_simulation.t > 2500)
                    species_diversity = \
                        np.count_nonzero(np.any(final_simulation.y[:no_species, last_500_t:] > 1e-3,
                                                axis=1))/no_species
                    resource_diversity =  \
                        np.count_nonzero(np.any(final_simulation.y[no_species:, last_500_t:] > 1e-3,
                                                axis=1))/no_resources
    
                    return [final_simulation, species_reinvadability, resource_reinvadability,
                            species_fluctuations, resource_fluctuations, species_diversity,
                            resource_diversity]
    
        messy_list = [simulate_community(no_species, no_resources, death,
                                         influx, growth, consumption)
                      for _ in range(no_lineages)]
        cleaned_messy_list = list(
            filter(lambda item: item is not None, messy_list))
        
        return cleaned_messy_list
    
    community_list = [community_mu_sigma(i, mu, sigma) for i in range(no_communities)]

    return {'Simulations': [item[0] for community in community_list for item in community],
            'Species Reinvadability': [item[1] for community in community_list for item in community],
            'Resource Reinvadability': [item[2] for community in community_list for item in community],
            'Species Fluctuation CV': [item[3] for community in community_list for item in community],
            'Resource Fluctuation CV': [item[4] for community in community_list for item in community],
            'Species diversity': [item[5] for community in community_list for item in community],
            'Resource diversity': [item[6] for community in community_list for item in community]}

# %%

def gLV_simulations_thermodynamic_limit(mu, sigma, no_species = 100):
    
    print({'mu': mu, 'sigma' : sigma}, '\n')
    
    
    no_communities = 50
    no_lineages = 5
   
    def simulate_community(no_lineages, no_species, mu, sigma):
        
        gLV_dynamics = gLV(no_species = no_species, growth_func = 'fixed', growth_args = None,
                       interact_func = 'random', interact_args = {'mu_a':mu,'sigma_a':sigma},
                       dispersal = 1e-8)
        gLV_dynamics.simulate_community(np.arange(no_lineages),t_end = 500)
        
        initial_abundances = np.vstack([ode_sol.y[:,-1] for ode_sol in gLV_dynamics.ODE_sols.values()])
        
        gLV_dynamics.simulate_community(np.arange(no_lineages), t_end = 3000, init_cond_func=None,
                                         usersupplied_init_conds = initial_abundances.T)
        gLV_dynamics.calculate_community_properties(np.arange(no_lineages), from_which_time = 2000)
        
        return deepcopy(gLV_dynamics)
    
    communities_list = [simulate_community(no_lineages, no_species, mu, sigma)
                  for i in range(no_communities)]
    
    simulations_and_properties = {'Simulations': [deepcopy(simulation) for community in communities_list for simulation in community.ODE_sols.values()],
                                  'Species Reinvadability': [deepcopy(reinvadability) for community in communities_list for reinvadability in community.reinvadability.values()],
                                  'Species Fluctuation CV': [deepcopy(fluctuation_coefficient(simulation.t, simulation.y, extinction_threshold = 1e-4)) for community in communities_list for simulation in community.ODE_sols.values()],
                                  'Species diversity': [deepcopy(diversity/no_species) for community in communities_list for diversity in community.final_diversity.values()],}

    return simulations_and_properties

# %%

def CR_dynamics_df(simulation_data, mu_consumption, sigma_consumption):
    
    closeness_to_competitive_exclusion = \
        np.array(simulation_data['Species diversity'])/np.array(simulation_data['Resource diversity'])
    
    data_length = len(simulation_data['Resource diversity'])
       
    annot_no_species = np.repeat(50, data_length)
    annot_mu = np.repeat(mu_consumption, data_length)
    annot_sigma = np.repeat(sigma_consumption, data_length)
    
    data = pd.DataFrame([annot_mu, annot_sigma, annot_no_species, simulation_data['Species Reinvadability'],
                         simulation_data['Resource Reinvadability'], simulation_data['Species Fluctuation CV'],
                         simulation_data['Resource Fluctuation CV'], simulation_data['Species diversity'],
                         simulation_data['Resource diversity'], closeness_to_competitive_exclusion], 
                        index = ['Average consumption rate', 'Consumption rate std', 'Number of species', 
                                 'Reinvadability (species)', 'Reinvadability (resources)',
                                 'Fluctuation CV (species)', 'Fluctuation CV (resources)',
                                 'Diversity (species)', 'Diversity (resources)',
                                 'Closeness to competitive exclusion']).T
    
    return data

# %%

def gLV_dynamics_df(simulation_data, mu, sigma):
     
    data_length = len(simulation_data['Species diversity'])
       
    annot_no_species = np.repeat(50, data_length)
    annot_mu = np.repeat(mu, data_length)
    annot_sigma = np.repeat(sigma, data_length)
    
    data = pd.DataFrame([annot_mu, annot_sigma, annot_no_species, simulation_data['Species Reinvadability'],
                         simulation_data['Species Fluctuation CV'], simulation_data['Species diversity']], 
                        index = ['Average interaction strength', 'Interaction strength std', 'Number of species', 
                                 'Reinvadability (species)', 'Fluctuation CV (species)',
                                 'Diversity (species)']).T
    
    return data

#%%

def prop_stable(x, instability_threshold = 1e-3):
    
    return np.count_nonzero(x < instability_threshold)/len(x)

# %%

def kendal_tau(x):
    
    corr_test = kendalltau(x.iloc[:,0], x.iloc[:,1], variant = 'b')
    
    return [corr_test.statistic, corr_test.pvalue]

#%%

def annotate_p_values(x):

    p_value_conditions = [x <= 0.001, x <= 0.01, x <= 0.05, x > 0.05]
    p_value_annotations = ['***', '**', '*', 'ns']
    
    return np.select(p_value_conditions, p_value_annotations, '')

#%%

def gLV_carrying_capacity_scaled_with_alpha(mu, sigma, no_species = 100):
    
    print({'mu': mu, 'sigma' : sigma}, '\n')
    
    
    no_communities = 25
    no_lineages = 5
   
    def simulate_community(no_lineages, no_species, mu, sigma):
        
        interaction_matrix = mu + sigma*np.random.randn(no_species, no_species)
        
        gLV_dynamics = gLV(no_species = no_species, growth_func = 'normal',
                           growth_args = {'mu_g':1,'sigma_g':sigma},
                           interact_func = None, interact_args = {'mu_a':mu,'sigma_a':sigma},
                           dispersal = 1e-8, usersupplied_interactmat = interaction_matrix)
        gLV_dynamics.simulate_community(np.arange(no_lineages),t_end = 500)
        
        initial_abundances = np.vstack([ode_sol.y[:,-1] for ode_sol in gLV_dynamics.ODE_sols.values()])
        
        gLV_dynamics.simulate_community(np.arange(no_lineages), t_end = 3000, init_cond_func=None,
                                         usersupplied_init_conds = initial_abundances.T)
        gLV_dynamics.calculate_community_properties(np.arange(no_lineages), from_which_time = 2000)
        
        return deepcopy(gLV_dynamics)
    
    communities_list = [simulate_community(no_lineages, no_species, mu, sigma)
                  for i in range(no_communities)]
    
    simulations_and_properties = {'Simulations': [deepcopy(simulation) for community in communities_list for simulation in community.ODE_sols.values()],
                                  'Species Reinvadability': [deepcopy(reinvadability) for community in communities_list for reinvadability in community.reinvadability.values()],
                                  'Species Fluctuation CV': [deepcopy(fluctuation_coefficient(simulation.t, simulation.y, extinction_threshold = 1e-4)) for community in communities_list for simulation in community.ODE_sols.values()],
                                  'Species diversity': [deepcopy(diversity/no_species) for community in communities_list for diversity in community.final_diversity.values()],}

    return simulations_and_properties

# %%

############################ generalised Lotka-Volterra dynamics ##############################

# large fixed species pool size, varying average and variance in consumption and growth.

mu_as = [0.3,0.5,0.7,0.9,1.1]
sigma_as = [0.05,0.1,0.15,0.2]

gLV_communities = [gLV_simulations_thermodynamic_limit(mu, sigma, no_species = 50) for sigma in sigma_as for mu in mu_as]
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_d_s_2.pkl",
            gLV_communities)

#%%

gLV_communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_d_s_2.pkl")

# %%

############################### Consumer resource dynamics ##############################################

# large fixed species pool size, varying average and variance in consumption and growth.

# %%
CR_communities_03_005 = thermodynamic_threshold_dynamics(0.3, sigma = 0.05, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_03_005.pkl",
            CR_communities_03_005)
del CR_communities_03_005
CR_communities_05_005 = thermodynamic_threshold_dynamics(0.5, sigma = 0.05, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_05_005.pkl",
            CR_communities_05_005)
del CR_communities_05_005
CR_communities_07_005 = thermodynamic_threshold_dynamics(0.7, sigma = 0.05, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_07_005.pkl",
            CR_communities_07_005)
del CR_communities_07_005
CR_communities_09_005 = thermodynamic_threshold_dynamics(0.9, sigma = 0.05, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_09_005.pkl",
            CR_communities_09_005)
del CR_communities_09_005
CR_communities_11_005 = thermodynamic_threshold_dynamics(1.1, sigma = 0.05, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_11_005.pkl",
            CR_communities_11_005)
del CR_communities_11_005
# %%
CR_communities_03_01 = thermodynamic_threshold_dynamics(0.3, sigma = 0.1, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_03_01.pkl",
            CR_communities_03_01)
del CR_communities_03_01
CR_communities_05_01 = thermodynamic_threshold_dynamics(0.5, sigma = 0.1, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_05_01.pkl",
            CR_communities_05_01)
del CR_communities_05_01
CR_communities_07_01 = thermodynamic_threshold_dynamics(0.7, sigma = 0.1, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_07_01.pkl",
            CR_communities_07_01)
del CR_communities_07_01
CR_communities_09_01 = thermodynamic_threshold_dynamics(0.9, sigma = 0.1, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_09_01.pkl",
            CR_communities_09_01)
del CR_communities_09_01
CR_communities_11_01 = thermodynamic_threshold_dynamics(1.1, sigma = 0.1, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_11_01.pkl",
            CR_communities_11_01)
del CR_communities_11_01
CR_communities_03_015 = thermodynamic_threshold_dynamics(0.3, sigma = 0.15, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_03_015.pkl",
            CR_communities_03_015)
del CR_communities_03_015
CR_communities_07_015 = thermodynamic_threshold_dynamics(0.7, sigma = 0.15, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_07_015.pkl",
            CR_communities_07_015)
del CR_communities_07_015
CR_communities_09_015 = thermodynamic_threshold_dynamics(0.9, sigma = 0.15, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_09_015.pkl",
            CR_communities_09_015)
del CR_communities_09_015
CR_communities_05_015 = thermodynamic_threshold_dynamics(0.5, sigma = 0.15, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_05_015.pkl",
            CR_communities_05_015)
del CR_communities_05_015
CR_communities_11_015 = thermodynamic_threshold_dynamics(1.1, sigma = 0.15, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_11_015.pkl",
            CR_communities_11_015)
del CR_communities_11_015
CR_communities_03_02 = thermodynamic_threshold_dynamics(0.3, sigma = 0.2, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_03_02.pkl",
            CR_communities_03_02)
del CR_communities_03_02
CR_communities_05_02 = thermodynamic_threshold_dynamics(0.5, sigma = 0.2, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_05_02.pkl",
            CR_communities_05_02)
del CR_communities_05_02
CR_communities_07_02 = thermodynamic_threshold_dynamics(0.7, sigma = 0.2, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_07_02.pkl",
            CR_communities_07_02)
del CR_communities_07_02
CR_communities_09_02 = thermodynamic_threshold_dynamics(0.9, sigma = 0.2, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_09_02.pkl",
            CR_communities_09_02)
del CR_communities_09_02
CR_communities_11_02 = thermodynamic_threshold_dynamics(1.1, sigma = 0.2, no_species = 50, no_resources = 50)
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_11_02.pkl",
            CR_communities_11_02)
del CR_communities_11_02

#%%

##################################### Generate Dataframes #############################

mu_string = ['03','05','07','09','11']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1]
sigma_string = ['005','01','015','02']
sigma_cs = [0.05, 0.1, 0.15, 0.2]

df_cr_list = []

for sigma, sigma_s in zip(sigma_cs, sigma_string):
    
    for mu, mu_s in zip(mu_cs, mu_string):
        
        simulation_data = \
            pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_" + str(mu_s) + "_" + str(sigma_s) + ".pkl")
            
        df = CR_dynamics_df(simulation_data, mu, sigma)
        df_cr_list.append(df)
        
        del simulation_data
        
data_mu_sigma_s = pd.concat(df_cr_list)
data_mu_sigma_s['Model'] = np.repeat('CR', data_mu_sigma_s.shape[0])

# %%

mu_as = [0.3, 0.5, 0.7, 0.9, 1.1]
sigma_as = [0.05, 0.1, 0.15, 0.2]

data_gLV_mu_sigma = pd.concat([gLV_dynamics_df(simulation_data, mu, sigma)
                           for simulation_data, mu, sigma in zip(gLV_communities, np.tile(mu_as, len(sigma_as)),
                           np.repeat(sigma_as, len(mu_as)))])


#%%

gLV_data_plotting = pd.concat([data_gLV_mu_sigma.iloc[np.where((data_gLV_mu_sigma['Average interaction strength'] == mu) & \
                                                               (data_gLV_mu_sigma['Interaction strength std'] == sigma))].iloc[:125]
                               for sigma in sigma_as for mu in mu_as])
    
gLV_data_plotting['Model'] = np.repeat('gLV', gLV_data_plotting.shape[0])

#%%

######################### Phase digrams ####################################

prop_reinvadability_cr = pd.pivot_table(data_mu_sigma_s,
                                      index = 'Consumption rate std',
                                      columns = 'Average consumption rate',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_cr = 1 - prop_reinvadability_cr

prop_reinvadability_gLV = pd.pivot_table(gLV_data_plotting,
                                      index = 'Interaction strength std',
                                      columns = 'Average interaction strength',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_gLV = 1 - prop_reinvadability_gLV

#%%

fig, axs = plt.subplots(1, 2, layout = 'constrained', figsize = (8,3.5))

sns.set_style('white')

colourmap_base = mpl.colormaps['viridis_r'](0.85)
light_dark_range = np.linspace(1,0,256)
lighten_func = lambda val, i : val + i*(1-val)
colours_list = [tuple([lighten_func(val,i) for val in colourmap_base[:-1]]) + (colourmap_base[-1],)
                for i in light_dark_range]
cmap = mpl.colors.ListedColormap(colours_list)
norm = mpl.colors.PowerNorm(0.45, vmin = 0, vmax = 1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

fig.suptitle('The phase transition out of stability qualitatively\ndiffers between the gLV and C-R model',fontsize=20,
             weight='bold', y = 1.15)

subfig0 = sns.heatmap(prop_reinvadability_cr,  ax=axs[0], vmin=0, vmax=1, cbar=False, cmap = cmap, norm = sm.norm,
                      square = True)
subfig1 = sns.heatmap(prop_reinvadability_gLV, ax=axs[1], vmin=0, vmax=1, cbar=True, cmap = cmap, norm = sm.norm,
                      square = True, cbar_kws={"ticks":[0,0.5,1]})
subfig0.invert_yaxis()
subfig1.invert_yaxis()

axs[1].set_title('gLV', fontsize = 16, weight = 'bold')
axs[0].set_title('Consumer-Resource model', fontsize = 16, weight = 'bold')

cbar = subfig1.collections[0].colorbar
cbar.set_label(label='P(chaos)',weight='bold', size='16')
cbar.ax.tick_params(labelsize=12)

subfig0.axhline(0, 0, 1.1, color = 'black', linewidth = 2)
subfig0.axhline(4, 0, 1.1, color = 'black', linewidth = 2)
subfig0.axvline(0, 0, 4, color = 'black', linewidth = 2)
subfig0.axvline(5, 0, 4, color = 'black', linewidth = 2)

subfig1.axhline(0, 0, 1.1, color = 'black', linewidth = 2)
subfig1.axhline(4, 0, 1.1, color = 'black', linewidth = 2)
subfig1.axvline(0, 0, 4, color = 'black', linewidth = 2)
subfig1.axvline(5, 0, 4, color = 'black', linewidth = 2)

subfig0.set_xlabel('Avg. consumption rate',fontsize=16, weight = 'bold')
subfig0.set_ylabel(r'$\sigma$',fontsize=16, weight = 'bold')
subfig1.set_xlabel('Avg. interaction strength',fontsize=16, weight = 'bold')
subfig1.set_ylabel('')

for ax in axs:

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/phase_diagram_gLV_CR.png",
            dpi=300,bbox_inches='tight')

#%%

###################################### Diversity - stability relationships ###################

sns.set_style('white')

fig, axs = plt.subplots(1,3,sharex=True,sharey=True,figsize=(11,3.75),layout='constrained')
fig.suptitle('Avg. species interaction or consumption rate',fontsize=28,weight='bold',y=1.2)
fig.supxlabel('Stability',fontsize=28,weight='bold')
fig.supylabel('Species\nsurvival fraction',fontsize=28,weight='bold', multialignment='center')

gLV_data_to_plot = \
    data_gLV_mu_sigma.iloc[np.arange(0,data_gLV_mu_sigma.shape[0],5)]
                                      
for ax, mu in zip(axs.flatten(), [0.3,0.7,0.9]):
    
    gLV_subdata = gLV_data_plotting.iloc[np.where((gLV_data_plotting['Average interaction strength'] == mu) & \
                                                  (gLV_data_plotting['Interaction strength std'] == 0.1))]
    cr_subdata =  data_mu_sigma_s.iloc[np.where((data_mu_sigma_s['Average consumption rate'] == mu) & \
                                   (data_mu_sigma_s['Consumption rate std'] == 0.1))]
        
    plotting_data = pd.concat([cr_subdata, gLV_subdata])
    plotting_data['Reinvadability (species)'] = 1 - plotting_data['Reinvadability (species)']
    
    subfig = sns.scatterplot(plotting_data, x = 'Reinvadability (species)',
                             y = 'Diversity (species)', hue='Model',
                             ax=ax,palette='viridis_r',s=100)
    subfig.set(xlabel=None,ylabel=None)
    subfig.set_yticks(range(2))
    subfig.set_xticks(range(2))
    ax.tick_params(axis='both', which='major', labelsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    
fig.legend(handles, ['C-R model', 'gLV'],
           loc='center right', bbox_to_anchor=(1.18, 0.75),
           fontsize = 16)

sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/diversity_stability_scatterplot_gLV_CR_2.png",
            dpi=300,bbox_inches='tight')


#%%
###################################### Diversity - stability relationships ###################

sns.set_style('white')

fig, axs = plt.subplots(1,1,sharex=True,sharey=True,figsize=(5.5,4.5),layout='constrained')
fig.suptitle('The gLV has a stronger negative diversity-stability\nrelationship in its dynamical phase than the C-R model.',
             fontsize=28,weight='bold',y=1.2)
gLV_subdata = gLV_data_plotting.iloc[np.where((gLV_data_plotting['Average interaction strength'] == 0.9) & \
                                              (gLV_data_plotting['Interaction strength std'] == 0.1))]
cr_subdata =  data_mu_sigma_s.iloc[np.where((data_mu_sigma_s['Average consumption rate'] == 0.3) & \
                               (data_mu_sigma_s['Consumption rate std'] == 0.1))]
                                      
plotting_data = pd.concat([cr_subdata, gLV_subdata])
plotting_data['Reinvadability (species)'] = 1 - plotting_data['Reinvadability (species)']

subfig = sns.scatterplot(plotting_data, x = 'Reinvadability (species)',
                         y = 'Diversity (species)', hue='Model',
                         ax=axs,
                         palette=['#349b55ff','#440154ff'],
                         s=140)
subfig.set(xlabel=None,ylabel=None)
subfig.set_yticks(range(2))
subfig.set_xticks(range(2))
axs.tick_params(axis='both', which='major', labelsize=16)
handles, labels = axs.get_legend_handles_labels()
axs.get_legend().remove()

axs.set_xlabel('Stability',fontsize=28,weight='bold')
axs.set_ylabel('Species\nsurvival fraction',fontsize=28,weight='bold', multialignment='center')
    
fig.legend(handles, ['C-R model', 'gLV'],
           loc='center right', bbox_to_anchor=(1.22, 0.75),
           fontsize = 18)

sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/diversity_stability_scatterplot_gLV_CR.png",
            dpi=300,bbox_inches='tight')
#%%

################ Correlation plots ###################################

diversity_stability_cr = data_mu_sigma_s.groupby(['Average consumption rate',
                                                  'Consumption rate std'])[['Reinvadability (species)',
                                                                            'Diversity (species)']].apply(lambda x : kendal_tau(x)).reset_index()
                                                                            
diversity_stability_cr[['Correlation', 'p value']] = pd.DataFrame(diversity_stability_cr[0].tolist(),
                                                                  index = diversity_stability_cr.index)                                                                 
                                                                          
d_s_cr = pd.pivot_table(diversity_stability_cr,
                        index = 'Consumption rate std',
                        columns = 'Average consumption rate',
                        values = 'Correlation')

d_s_cr_p_val = pd.pivot_table(diversity_stability_cr,
                        index = 'Consumption rate std',
                        columns = 'Average consumption rate',
                        values = 'p value')

d_s_cr_annotation = annotate_p_values(d_s_cr_p_val)

#%%

diversity_stability_gLV = gLV_data_plotting.groupby(['Average interaction strength',
                                                     'Interaction strength std'])[['Reinvadability (species)',
                                                                            'Diversity (species)']].apply(lambda x : kendal_tau(x)).reset_index()
                                                                            
diversity_stability_gLV[['Correlation', 'p value']] = pd.DataFrame(diversity_stability_gLV[0].tolist(),
                                                                  index = diversity_stability_gLV.index)                                                                 
                                                                          
d_s_gLV = pd.pivot_table(diversity_stability_gLV,
                        index = 'Interaction strength std',
                        columns = 'Average interaction strength',
                        values = 'Correlation')

d_s_gLV_p_val = pd.pivot_table(diversity_stability_gLV,
                        index = 'Interaction strength std',
                        columns = 'Average interaction strength',
                        values = 'p value')

d_s_gLV_annotation = annotate_p_values(d_s_gLV_p_val)

#%%

fig, axs = plt.subplots(1, 2, layout = 'constrained', figsize = (8,3.5))

sns.set_style('white')

custom_corrpal = \
    mpl.colors.LinearSegmentedColormap.from_list('custom_corrpal',
                                                 ['#440154ff','#ffffffff','#305f19ff'],
                                                 N=256)

fig.suptitle('Diversity-stability relationships in the dynamical phase of the\ngLV and C-R model',fontsize=20,
             weight='bold', y = 1.15)

subfig0 = sns.heatmap(-d_s_cr,  ax=axs[0], vmin=-1, vmax=1, cbar=False, cmap = custom_corrpal,
                      square = True, annot = d_s_cr_annotation, fmt = '')
subfig1 = sns.heatmap(-d_s_gLV, ax=axs[1], vmin=-1, vmax=1, cbar=True, cmap = custom_corrpal,
                      square = True, cbar_kws={"ticks":[-1,0,1]},
                      annot = d_s_gLV_annotation, fmt = '')
subfig0.invert_yaxis()
subfig1.invert_yaxis()

axs[1].set_title('gLV', fontsize = 16, weight = 'bold')
axs[0].set_title('Consumer-Resource model', fontsize = 16, weight = 'bold')

cbar = subfig1.collections[0].colorbar
cbar.set_label(label='corr(diversity, stability)',weight='bold', size='16')
cbar.ax.tick_params(labelsize=12)

subfig0.axhline(0, 0, 1.1, color = 'black', linewidth = 2)
subfig0.axhline(4, 0, 1.1, color = 'black', linewidth = 2)
subfig0.axvline(0, 0, 4, color = 'black', linewidth = 2)
subfig0.axvline(5, 0, 4, color = 'black', linewidth = 2)

subfig1.axhline(0, 0, 1.1, color = 'black', linewidth = 2)
subfig1.axhline(4, 0, 1.1, color = 'black', linewidth = 2)
subfig1.axvline(0, 0, 4, color = 'black', linewidth = 2)
subfig1.axvline(5, 0, 4, color = 'black', linewidth = 2)

subfig0.set_xlabel('Avg. consumption rate',fontsize=16, weight = 'bold')
subfig0.set_ylabel(r'$\sigma$',fontsize=16, weight = 'bold')
subfig1.set_xlabel('Avg. interaction strength',fontsize=16, weight = 'bold')
subfig1.set_ylabel('')

for ax in axs:

    ax.set_yticklabels([])
    ax.set_xticklabels([])

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/diversity_stability_corrplot_gLV_CR.png",
            dpi=300,bbox_inches='tight')


#%%

############################## Competitive exclusion principle ################

sns.set_style('white')

fig, axs = plt.subplots(1,3,sharex=True,sharey=True,figsize=(11,3.5),layout='constrained')
fig.suptitle('Avg. species interaction or consumption rate',fontsize=28,weight='bold',y=1.2)
fig.supxlabel('Stability',fontsize=28,weight='bold')
fig.supylabel('Closeness to\ncompetitive exclusion',fontsize=28,weight='bold', multialignment='center')
                                      
for ax, mu in zip(axs.flatten(), [0.3,0.7,0.9]):
    
    cr_subdata =  data_mu_sigma_s.iloc[np.where((data_mu_sigma_s['Average consumption rate'] == mu) & \
                                   (data_mu_sigma_s['Consumption rate std'] == 0.1))]
     
    subfig = sns.scatterplot(x = 1- cr_subdata['Reinvadability (species)'],
                             y = cr_subdata['Closeness to competitive exclusion'],
                             ax=ax,palette='viridis_r',s=100)
    subfig.set(xlabel=None,ylabel=None)
    subfig.set_yticks(range(2))
    subfig.set_xticks(range(2))
    subfig.set_ylim([0,1.05])
    ax.tick_params(axis='both', which='major', labelsize=16)

sns.despine()

#%%

ce_stability_cr = data_mu_sigma_s.groupby(['Average consumption rate',
                                           'Consumption rate std'])[['Reinvadability (species)',
                                                                     'Closeness to competitive exclusion']].apply(lambda x : kendal_tau(x)).reset_index()
                                                                            
ce_stability_cr[['Correlation', 'p value']] = pd.DataFrame(ce_stability_cr[0].tolist(),
                                                           index = ce_stability_cr.index)                                                                 
                                                                          
ce_s_cr = pd.pivot_table(ce_stability_cr,
                        index = 'Consumption rate std',
                        columns = 'Average consumption rate',
                        values = 'Correlation')

ce_s_cr_p_val = pd.pivot_table(ce_stability_cr,
                               index = 'Consumption rate std',
                               columns = 'Average consumption rate',
                               values = 'p value')

ce_s_cr_annotation = annotate_p_values(ce_s_cr_p_val)

#%%

fig, axs = plt.subplots(1, 1, layout = 'constrained', figsize = (4,3.5))

sns.set_style('white')

custom_corrpal = \
    mpl.colors.LinearSegmentedColormap.from_list('custom_corrpal',
                                                 ['#440154ff','#ffffffff','#305f19ff'],
                                                 N=256)

subfig0 = sns.heatmap(-ce_s_cr,  ax=axs, vmin=-1, vmax=1, cbar=True, cmap = custom_corrpal,
                      square = True, annot = ce_s_cr_annotation, fmt = '',
                      cbar_kws={"ticks":[-1,0,1]})
subfig0.invert_yaxis()

axs.set_title('Consumer-Resource model', fontsize = 16, weight = 'bold')

cbar = subfig0.collections[0].colorbar
cbar.set_label(label='corr(diversity, stability)',weight='bold', size='16')
cbar.ax.tick_params(labelsize=12)

subfig0.axhline(0, 0, 1.1, color = 'black', linewidth = 2)
subfig0.axhline(4, 0, 1.1, color = 'black', linewidth = 2)
subfig0.axvline(0, 0, 4, color = 'black', linewidth = 2)
subfig0.axvline(5, 0, 4, color = 'black', linewidth = 2)

subfig0.set_xlabel('Avg. consumption rate',fontsize=16, weight = 'bold')
subfig0.set_ylabel(r'$\sigma$',fontsize=16, weight = 'bold')

axs.set_yticklabels([])
axs.set_xticklabels([])

#%%

############################# gLV carrying capacities scaled ########################

mu_as = [0.3,0.5,0.7,0.9,1.1]
sigma_as = [0.05,0.1,0.15,0.2]

#%%

gLV_scaled_K_005 = [gLV_carrying_capacity_scaled_with_alpha(mu, 0.05, no_species = 50) for mu in mu_as]

#%%

gLV_scaled_K_01 = [gLV_carrying_capacity_scaled_with_alpha(mu, 0.1, no_species = 50) for mu in mu_as]

#%%

gLV_scaled_K_015 = [gLV_carrying_capacity_scaled_with_alpha(mu, 0.15, no_species = 50) for mu in mu_as]

#%%
gLV_scaled_K_02 = [gLV_carrying_capacity_scaled_with_alpha(mu, 0.2, no_species = 50) for mu in mu_as]

#%%

data_gLV_mu_sigma_2 = pd.concat([gLV_dynamics_df(simulation_data, mu, sigma)
                           for simulation_data_sigma, sigma in \
                               zip([gLV_scaled_K_005,gLV_scaled_K_01,
                                    gLV_scaled_K_015,gLV_scaled_K_02], sigma_as)
                               for simulation_data, mu in \
                                   zip(simulation_data_sigma, mu_as)])
    
#%%

data_gLV_mu_sigma_2['Model'] = 'gLV'

#%%

######################### Phase digrams ####################################

prop_reinvadability_cr = pd.pivot_table(data_mu_sigma_s,
                                      index = 'Consumption rate std',
                                      columns = 'Average consumption rate',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_cr = 1 - prop_reinvadability_cr

prop_reinvadability_gLV = pd.pivot_table(data_gLV_mu_sigma_2,
                                      index = 'Interaction strength std',
                                      columns = 'Average interaction strength',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_gLV = 1 - prop_reinvadability_gLV

#%%

fig, axs = plt.subplots(1, 2, layout = 'constrained', figsize = (8,3.5))

sns.set_style('white')

colourmap_base = mpl.colormaps['viridis_r'](0.85)
light_dark_range = np.linspace(1,0,256)
lighten_func = lambda val, i : val + i*(1-val)
colours_list = [tuple([lighten_func(val,i) for val in colourmap_base[:-1]]) + (colourmap_base[-1],)
                for i in light_dark_range]
cmap = mpl.colors.ListedColormap(colours_list)
norm = mpl.colors.PowerNorm(0.45, vmin = 0, vmax = 1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

fig.suptitle('The phase transition out of stability qualitatively\ndiffers between the gLV and C-R model',fontsize=20,
             weight='bold', y = 1.15)

subfig0 = sns.heatmap(prop_reinvadability_cr,  ax=axs[0], vmin=0, vmax=1, cbar=False, cmap = cmap, norm = sm.norm,
                      square = True)
subfig1 = sns.heatmap(prop_reinvadability_gLV, ax=axs[1], vmin=0, vmax=1, cbar=True, cmap = cmap, norm = sm.norm,
                      square = True, cbar_kws={"ticks":[0,0.5,1]})
subfig0.invert_yaxis()
subfig1.invert_yaxis()

axs[1].set_title('gLV', fontsize = 16, weight = 'bold')
axs[0].set_title('Consumer-Resource model', fontsize = 16, weight = 'bold')

cbar = subfig1.collections[0].colorbar
cbar.set_label(label='P(chaos)',weight='bold', size='16')
cbar.ax.tick_params(labelsize=12)

subfig0.axhline(0, 0, 1.1, color = 'black', linewidth = 2)
subfig0.axhline(4, 0, 1.1, color = 'black', linewidth = 2)
subfig0.axvline(0, 0, 4, color = 'black', linewidth = 2)
subfig0.axvline(5, 0, 4, color = 'black', linewidth = 2)

subfig1.axhline(0, 0, 1.1, color = 'black', linewidth = 2)
subfig1.axhline(4, 0, 1.1, color = 'black', linewidth = 2)
subfig1.axvline(0, 0, 4, color = 'black', linewidth = 2)
subfig1.axvline(5, 0, 4, color = 'black', linewidth = 2)

subfig0.set_xlabel('Avg. consumption rate',fontsize=16, weight = 'bold')
subfig0.set_ylabel(r'$\sigma$',fontsize=16, weight = 'bold')
subfig1.set_xlabel('Avg. interaction strength',fontsize=16, weight = 'bold')
subfig1.set_ylabel('')

for ax in axs:

    ax.set_yticklabels([])
    ax.set_xticklabels([])

#%%

###################################### Diversity - stability relationships ###################

sns.set_style('white')

fig, axs = plt.subplots(1,1,sharex=True,sharey=True,figsize=(5.5,4.5),layout='constrained')
fig.suptitle('The gLV has a stronger negative diversity-stability\nrelationship in its dynamical phase than the C-R model.',
             fontsize=28,weight='bold',y=1.2)
gLV_subdata = data_gLV_mu_sigma_2.iloc[np.where((data_gLV_mu_sigma_2['Average interaction strength'] == 0.7) & \
                                              (data_gLV_mu_sigma_2['Interaction strength std'] == 0.2))]
cr_subdata =  data_mu_sigma_s.iloc[np.where((data_mu_sigma_s['Average consumption rate'] == 0.7) & \
                               (data_mu_sigma_s['Consumption rate std'] == 0.2))]
                                      
plotting_data = pd.concat([cr_subdata, gLV_subdata])
plotting_data['Reinvadability (species)'] = 1 - plotting_data['Reinvadability (species)']

subfig = sns.scatterplot(plotting_data, x = 'Reinvadability (species)',
                         y = 'Diversity (species)', hue='Model',
                         ax=axs,
                         palette=['#349b55ff','#440154ff'],
                         s=140)
subfig.set(xlabel=None,ylabel=None)
subfig.set_yticks(range(2))
subfig.set_xticks(range(2))
axs.tick_params(axis='both', which='major', labelsize=16)
handles, labels = axs.get_legend_handles_labels()
axs.get_legend().remove()

axs.set_xlabel('Stability',fontsize=28,weight='bold')
axs.set_ylabel('Species\nsurvival fraction',fontsize=28,weight='bold', multialignment='center')
    
fig.legend(handles, ['C-R model', 'gLV'],
           loc='center right', bbox_to_anchor=(1.22, 0.75),
           fontsize = 18)

sns.despine()


#%%

#%%

######################### Phase digrams ####################################

diversity_cr = pd.pivot_table(data_mu_sigma_s,
                                      index = 'Consumption rate std',
                                      columns = 'Average consumption rate',
                                      values = 'Closeness to competitive exclusion',
                                      aggfunc = 'mean')

diversity_gLV = pd.pivot_table(data_gLV_mu_sigma_2,
                                      index = 'Interaction strength std',
                                      columns = 'Average interaction strength',
                                      values = 'Diversity (species)',
                                      aggfunc = 'mean')

#%%

fig, axs = plt.subplots(1, 2, layout = 'constrained', figsize = (8,3.5))

sns.set_style('white')

colourmap_base = mpl.colormaps['viridis_r'](0.85)
light_dark_range = np.linspace(1,0,256)
lighten_func = lambda val, i : val + i*(1-val)
colours_list = [tuple([lighten_func(val,i) for val in colourmap_base[:-1]]) + (colourmap_base[-1],)
                for i in light_dark_range]
cmap = mpl.colors.ListedColormap(colours_list)
norm = mpl.colors.PowerNorm(1, vmin = 0, vmax = 1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

fig.suptitle('The phase transition out of stability qualitatively\ndiffers between the gLV and C-R model',fontsize=20,
             weight='bold', y = 1.15)

subfig0 = sns.heatmap(diversity_cr,  ax=axs[0], vmin=0, vmax=1, cbar=False, cmap = cmap, norm = sm.norm,
                      square = True)
subfig1 = sns.heatmap(diversity_gLV, ax=axs[1], vmin=0, vmax=1, cbar=True, cmap = cmap, norm = sm.norm,
                      square = True, cbar_kws={"ticks":[0,0.5,1]})
subfig0.invert_yaxis()
subfig1.invert_yaxis()

axs[1].set_title('gLV', fontsize = 16, weight = 'bold')
axs[0].set_title('Consumer-Resource model', fontsize = 16, weight = 'bold')

cbar = subfig1.collections[0].colorbar
cbar.set_label(label='P(chaos)',weight='bold', size='16')
cbar.ax.tick_params(labelsize=12)

subfig0.axhline(0, 0, 1.1, color = 'black', linewidth = 2)
subfig0.axhline(4, 0, 1.1, color = 'black', linewidth = 2)
subfig0.axvline(0, 0, 4, color = 'black', linewidth = 2)
subfig0.axvline(5, 0, 4, color = 'black', linewidth = 2)

subfig1.axhline(0, 0, 1.1, color = 'black', linewidth = 2)
subfig1.axhline(4, 0, 1.1, color = 'black', linewidth = 2)
subfig1.axvline(0, 0, 4, color = 'black', linewidth = 2)
subfig1.axvline(5, 0, 4, color = 'black', linewidth = 2)

subfig0.set_xlabel('Avg. consumption rate',fontsize=16, weight = 'bold')
subfig0.set_ylabel(r'$\sigma$',fontsize=16, weight = 'bold')
subfig1.set_xlabel('Avg. interaction strength',fontsize=16, weight = 'bold')
subfig1.set_ylabel('')

for ax in axs:

    ax.set_yticklabels([])
    ax.set_xticklabels([])








