# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:07:31 2024

@author: jamil
"""

# %%

# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\Chemically-mediated models

########################

# %%

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kendalltau
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import sys
from copy import deepcopy
import colorsys
from random import shuffle

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

    #dRdt = (resources * (influx - resources)) - \
    #    (resources * np.sum(growth.T * consumption * species, axis=1)) + 1e-8
    
    dRdt = (resources * (influx - resources)) - \
        (resources * np.sum(consumption * species, axis=1)) + 1e-8

    return np.concatenate((dSdt, dRdt))

# %%

def dCR_dt_2(t, var,
             no_species,
             growth, death, consumption, influx):
    
    species = var[:no_species]
    resources = var[no_species:]

    dSdt = species * (np.sum(growth * consumption.T * resources, axis=1) - death) + 1e-8

    dRdt = (resources * (influx - resources)) - \
        (resources * np.sum(consumption * species, axis=1)) + 1e-8

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

def consumer_resource_dynamics(model,
                               mu_c, sigma_c, mu_g, sigma_g,
                               no_species = 100, no_resources = 100):
    
    influx = np.ones(no_resources)
    death = np.ones(no_species)
    
    no_communities = 25
    no_lineages = 5
    
    def community_mu_sigma(i, no_lineages,
                           mu_c, sigma_c, mu_g, sigma_g,
                           death, influx, no_species, no_resources):
        
        print({'mu': mu_c, 'sigma' : sigma_c, 'Community' : i}, '\n')
        
        growth = np.abs(normal_distributed_parameters(mu_g, sigma_g,
                                                    dims=(no_species, no_resources)))
        
        consumption = np.abs(normal_distributed_parameters(mu_c, sigma_c,
                                                    dims=(no_resources, no_species)))

        def simulate_community(no_species, no_resources,
                               death, influx, growth, consumption):
            
            initial_abundances = np.random.uniform(1e-8, 2/no_species, no_species)
            initial_concentrations = np.random.uniform(1e-8, 2/no_species, no_resources)

            first_simulation = solve_ivp(model, [0, 500],
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

                final_simulation = solve_ivp(model, [0, 3000],
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
                    resource_reinvadability = rescaled_detect_invasability(final_simulation.t, final_simulation.y[no_species:, :],
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
    
    community_list = [community_mu_sigma(i, no_lineages,
                           mu_c, sigma_c, mu_g, sigma_g,
                           death, influx, no_species, no_resources)
                      for i in range(no_communities)]

    return {'Simulations': [item[0] for community in community_list for item in community],
            'Species Reinvadability': [item[1] for community in community_list for item in community],
            'Resource Reinvadability': [item[2] for community in community_list for item in community],
            'Species Fluctuation CV': [item[3] for community in community_list for item in community],
            'Resource Fluctuation CV': [item[4] for community in community_list for item in community],
            'Species diversity': [item[5] for community in community_list for item in community],
            'Resource diversity': [item[6] for community in community_list for item in community]}

# %%

def gLV_dynamics(mu_a, sigma_a, mu_g, sigma_g, no_species = 100, scale_self_inhibition = False):
    
    print({'mu': mu_a, 'sigma' : sigma_a}, '\n')
    
    no_communities = 25
    no_lineages = 5
   
    def simulate_community(no_lineages, no_species, mu_a, sigma_a, mu_g, sigma_g):
        
        interaction_matrix = normal_distributed_parameters(mu_a, sigma_a, (no_species, no_species))
        
        if scale_self_inhibition is False:
            
            np.fill_diagonal(interaction_matrix, 1)
        
        gLV_dynamics = gLV(no_species = no_species, growth_func = 'normal',
                           growth_args = {'mu_g' : mu_g, 'sigma_g' : sigma_g},
                           interact_func = None, 
                           interact_args = {'mu_a' : mu_a, 'sigma_a' : sigma_a},
                           usersupplied_interactmat = interaction_matrix,
                           dispersal = 1e-8)
        gLV_dynamics.simulate_community(np.arange(no_lineages),t_end = 500)
        
        initial_abundances = np.vstack([ode_sol.y[:,-1] for ode_sol in gLV_dynamics.ODE_sols.values()])
        
        gLV_dynamics.simulate_community(np.arange(no_lineages), t_end = 3000, init_cond_func=None,
                                         usersupplied_init_conds = initial_abundances.T)
        gLV_dynamics.calculate_community_properties(np.arange(no_lineages), from_which_time = 2000)
        
        return deepcopy(gLV_dynamics)
    
    communities_list = [simulate_community(no_lineages, no_species, mu_a, sigma_a, mu_g, sigma_g)
                        for i in range(no_communities)]
    
    simulations_and_properties = {'Simulations': [deepcopy(simulation) for community in communities_list for simulation in community.ODE_sols.values()],
                                  'Species Reinvadability': [deepcopy(reinvadability) for community in communities_list for reinvadability in community.reinvadability.values()],
                                  'Species Fluctuation CV': [deepcopy(fluctuation_coefficient(simulation.t, simulation.y, extinction_threshold = 1e-4)) for community in communities_list for simulation in community.ODE_sols.values()],
                                  'Species diversity': [deepcopy(diversity/no_species) for community in communities_list for diversity in community.final_diversity.values()],}

    return simulations_and_properties

# %%

def CR_dynamics_df(simulation_data, mu_c, sigma_c):
    
    closeness_to_competitive_exclusion = \
        np.array(simulation_data['Species diversity'])/np.array(simulation_data['Resource diversity'])
    
    data_length = len(simulation_data['Resource diversity'])
       
    annot_no_species = np.repeat(int(simulation_data['Simulations'][0].y.shape[0]/2),
                                 data_length)
    annot_mu = np.repeat(mu_c, data_length)
    annot_sigma = np.repeat(sigma_c, data_length)
    
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
       
    annot_no_species = np.repeat(simulation_data['Simulations'][0].y.shape[0],
                                 data_length)
    annot_mu = np.repeat(mu, data_length)
    annot_sigma = np.repeat(sigma, data_length)
    
    data = pd.DataFrame([annot_mu, annot_sigma, annot_no_species, simulation_data['Species Reinvadability'],
                         simulation_data['Species Fluctuation CV'], simulation_data['Species diversity']], 
                        index = ['Average interaction strength', 'Interaction strength std', 'Number of species', 
                                 'Reinvadability (species)', 'Fluctuation CV (species)',
                                 'Diversity (species)']).T
    
    return data

#%%

def create_and_delete_CR(filename, kwargs, function_option = '1'):
    
    match function_option:
        
        case '1':
            
            model = dCR_dt
            
        case '2':
            
            model = dCR_dt_2
    
    CR_communities = consumer_resource_dynamics(model, **kwargs)
    
    pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl",
                CR_communities)
    del CR_communities
    
#%%

def create_df_and_delete_simulations(filename, mu_c, sigma_c):
    
    CR_communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl")
    
    df = CR_dynamics_df(CR_communities, mu_c, sigma_c)
    
    return df
    
# %%

def phase_diagram_avg(data_list,
                      colourmap_base = mpl.colormaps['viridis_r'](0.85)): 
    
    fig, axs = plt.subplots(1, len(data_list), layout = 'constrained', figsize = (8,2.6))

    sns.set_style('white')
    
    def cmap_norm(base, data_len = 256):
        
        colourmap_base = mpl.colors.ColorConverter.to_rgb(base)
        
        def scale_lightness(rgb, scale_l):
            
            # convert rgb to hls
            h, l, s = colorsys.rgb_to_hls(*rgb)
            # manipulate h, l, s values and return as rgb
            return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)
        
        colours_list = [scale_lightness(colourmap_base, scale) 
                        for scale in np.linspace(3.5, 0, data_len)]
        
        cmap = mpl.colors.ListedColormap(colours_list)
        norm = mpl.colors.PowerNorm(0.25, vmin = 0, vmax = 1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        return {'cmap' : cmap, 'sm' : sm}

    colourmap = cmap_norm(colourmap_base)
    
    for i, (data, ax) in enumerate(zip(data_list, axs.flatten())):
        
        if i == len(data_list) - 1:
            
            subfig = sns.heatmap(data, ax=ax, vmin=0, vmax=1, cbar=True, cmap = colourmap['cmap'],
                                 norm = colourmap['sm'].norm, cbar_kws={"ticks":[0,0.5,1]})
            
            cbar = subfig.collections[0].colorbar
            cbar.set_label(label='Avg. volatility',weight='bold', size='14')
            cbar.ax.tick_params(labelsize=12)
            
        else:
            
            subfig = sns.heatmap(data,  ax=ax, vmin=0, vmax=1, cbar=False, cmap = colourmap['cmap'],
                                 norm = colourmap['sm'].norm)
        
        subfig.invert_yaxis()
        #subfig.axhline(0, 0, 1.1, color = 'black', linewidth = 2)
        #subfig.axhline(4, 0, 1.1, color = 'black', linewidth = 2)
        #subfig.axvline(0, 0, 4, color = 'black', linewidth = 2)
        #subfig.axvline(5, 0, 4, color = 'black', linewidth = 2)

    for i, ax in enumerate(axs):
        
        ax.set_xticklabels([r'$0.3$', r'$0.5$', r'$0.7$', r'$0.9$', r'$1.1$'])
        
        if i == 0:
            
            ax.set_yticklabels([r'$0.05^2$',r'$0.1^2$',r'$0.15^2$',r'$0.2^2$'])
        
        else:
            
            ax.set_yticklabels([])
            
    return fig, axs

#%%

def binned_average(data, x, y,
                   bins = [0, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1]):
    
    def bin_data(x_vals, y_vals, lower_bound, upper_bound):
        
        index = np.logical_and(x_vals >= lower_bound, x_vals < upper_bound)
        
        return x_vals[index], y_vals[index]
    
    def bin_average(x_points, y_points):
        
        x_average = np.mean(x_points)
        y_average = np.mean(y_points)
        
        return x_average, y_average
    
    binned_x_y = [bin_data(data[x].to_numpy(), data[y].to_numpy(), bins[i], bins[i+1])
                  for i in range(len(bins) - 1)]
    
    averaged_x_y = [bin_average(xy[0], xy[1]) for xy in binned_x_y]
    
    averaged_x = np.array([val[0] for val in averaged_x_y])
    averaged_y = np.array([val[1] for val in averaged_x_y])
    
    return averaged_x, averaged_y

# %%
        
def diversity_stability_fit_plot(data_list,
                                 x, y, hue, ylabel, legend_labels,
                                 mus, sigmas,
                                 fit_lines,
                                 palette = ['#349b55ff','#ab00d5ff']):
    
    sns.set_style('white')
    
    fig, axs = plt.subplots(1,1,sharex=True,sharey=True,figsize=(5.5,5),layout='constrained')
    
    def generate_subdata(data, mu, sigma):
        
        if np.any(data.columns == 'Average interaction strength'):
            
            subdata = data.iloc[np.where((data['Average interaction strength'] == mu) & \
                                         (data['Interaction strength std'] == sigma))]
            
        elif np.any(data.columns == 'Average consumption rate'):
            
            subdata = data.iloc[np.where((data['Average consumption rate'] == mu) & \
                                         (data['Consumption rate std'] == sigma))]
        return subdata
                          
    plotting_data = pd.concat([generate_subdata(data, mu, sigma) 
                               for data, mu, sigma in zip(data_list, mus, sigmas)])

    plotting_data[x] = 1 - plotting_data[x]

    subfig = sns.scatterplot(data = plotting_data, x = x, y = y, hue = hue,
                             ax=axs, palette = palette,
                             s=75, edgecolor = 'black', alpha = 0.4)
    
    for i, line in enumerate(fit_lines):
        
        axs.plot(1 - line[0], line[1], color = 'black', linewidth = 6)
        axs.plot(1 - line[0], line[1], color = palette[i], linewidth = 5)
        
    subfig.set(xlabel=None,ylabel=None)
    subfig.set_yticks(range(2))
    subfig.set_xticks(range(2))
    axs.tick_params(axis='both', which='major', labelsize=24)
    handles, labels = axs.get_legend_handles_labels()
    axs.get_legend().remove()

    #axs.set_xlabel('Stability metric',fontsize=28,weight='bold')
    axs.set_ylabel(ylabel,fontsize=32,weight='bold', multialignment='center')
        
    #fig.legend(handles, legend_labels,
    #           loc='center right', bbox_to_anchor=(1.8, 0.75),
    #           fontsize = 18)

    sns.despine()
    
    return fig, axs

#%%

def competitive_exclusion_fit_plot(dfs, x, y, mu_c, sigma_c,
                                   fit_lines):
    
    sns.set_style('white')
    
    palette = ['#2a7c44ff','#88d7a2ff']

    fig, axs = plt.subplots(1,1,sharex=True,sharey=True,figsize=(5.5,5),layout='constrained')
    
    def generate_subdata(data, mu, sigma):
            
        subdata = data.iloc[np.where((data['Average consumption rate'] == mu) & \
                                     (data['Consumption rate std'] == sigma))]
        
        return subdata
                          
    plotting_data = pd.concat([generate_subdata(data, mu, sigma) 
                               for data, mu, sigma in zip(dfs, mu_c, sigma_c)])
       
    plotting_data[x] = 1 - plotting_data[x]
    
    axs.axhline(1, 0, 1, color = 'grey', linewidth = 4, linestyle = '--')
    
    subfig = sns.scatterplot(data = plotting_data, x = x, y = y, hue='Conditions',
                             ax=axs, palette=palette[:len(dfs)],
                             s=75, edgecolor = '#0b2213ff', alpha = 0.4)
    
    for i, line in enumerate(fit_lines):
        
        axs.plot(1 - line[0], line[1], color = 'black', linewidth = 6)
        axs.plot(1 - line[0], line[1], color = palette[i], linewidth = 5)
 
    
    subfig.set(xlabel=None,ylabel=None)
    subfig.set_yticks(range(2))
    subfig.set_xticks(range(2))
    axs.tick_params(axis='both', which='major', labelsize=24)
    handles, labels = axs.get_legend_handles_labels()
    axs.get_legend().remove()

    #axs.set_xlabel('Stability metric',fontsize=28,weight='bold')
    axs.set_ylabel(r'$\frac{\text{Species surivival fraction}}{\text{Resource surivival fraction}}$',
                   fontsize=32,weight='bold', multialignment='center')
        
    #fig.legend(handles, np.unique(plotting_data['Conditions']),
    #           loc='center right', bbox_to_anchor=(1.6, 0.75),
    #           fontsize = 18)

    sns.despine()
    
    return fig, axs

#%%

def plot_community_dynamics(simulation_cr_s, simulation_cr_c,
                            simulation_gLV_s, simulation_gLV_c):
    
    def colour_map(base, data_len = 50):
        
        colourmap_base = mpl.colors.ColorConverter.to_rgb(base)
        
        def scale_lightness(rgb, scale_l):
            
            # convert rgb to hls
            h, l, s = colorsys.rgb_to_hls(*rgb)
            # manipulate h, l, s values and return as rgb
            return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)
        
        colours_list = [scale_lightness(colourmap_base, scale) 
                        for scale in np.linspace(0, 3.5, data_len)]
        shuffle(colours_list)
        
        return mpl.colors.ListedColormap(colours_list)

    consumer_colours = colour_map('#216035ff')
    resource_colours = colour_map('#317c2aff')
    
    gLV_colours = colour_map('#5e0174ff')
    
    fig, axs = plt.subplots(2,3,figsize=(10,4),layout='constrained')

    for ax, data, indexer, cmap in zip(axs.flatten(),
                                       [simulation_gLV_s, simulation_cr_s,
                                        simulation_cr_s,
                                        simulation_gLV_c, simulation_cr_c,
                                        simulation_cr_c],
                                      [np.arange(50), np.arange(50),
                                       np.arange(50, 100),
                                       np.arange(50), np.arange(50),
                                       np.arange(50, 100)],
                                      [gLV_colours, 
                                       consumer_colours, resource_colours,
                                       gLV_colours,
                                       consumer_colours, resource_colours]):
        
        for i, index in enumerate(indexer):
            
            ax.plot(data.t, data.y[index,:].T, color = cmap(i), linewidth = 2)

    for ax in axs.flatten():
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    for ax in axs.flatten()[:3]:
        
        ax.set_ylim([-0.01, 0.25])
    
    sns.despine()
     
    return fig, axs
    
    