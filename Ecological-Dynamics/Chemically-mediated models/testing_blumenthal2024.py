# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:51:01 2024

@author: jamil
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:54:13 2024

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
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
import pickle
import pandas as pd
import seaborn as sns

# %%
def dCR_dt(t, var,
           no_species,
           growth, death, consumption, influx):

    # breakpoint()

    #var[var < 1e-9] = 0

    species = var[:no_species]
    resources = var[no_species:]

    dSdt = species * (np.sum(growth * resources, axis=1) - death) + 10**-10

    dRdt = (resources * (influx - resources)) - \
        (resources * np.sum(growth.T * consumption * species, axis=1)) + 10**-10

    # dRdt = influx - \
    #    resources * np.sum(growth.T * consumption * species, axis = 1)

    #dSdt = species * (np.sum(growth * (resources/(1 + resources)), axis = 1) - death) + 10**-8

    # dRdt = resources * (influx - resources) - \
    #    (resources/(1 + resources)) * np.sum(growth.T * consumption * species, axis = 1) + 10**-8

    # dRdt = influx - resources - \
    #    (resources/(1 + resources)) * np.sum(growth.T * consumption * species, axis = 1) + 10**-8

    return np.concatenate((dSdt, dRdt))

# %%


def normal_distributed_parameters(mu, sigma, dims):

    return mu + sigma*np.random.randn(*dims)

# %%


def closeness_to_competitive_exclusion(no_species, no_resources,
                                       death, influx,
                                       growth_stats, consumption_stats):

    def resource_species_diversity():

        growth = normal_distributed_parameters(**growth_stats,
                                               dims=(no_species, no_resources))
        consumption = np.abs(normal_distributed_parameters(**consumption_stats,
                                                    dims=(no_resources, no_species)))

        initial_abundances = np.random.uniform(0.1, 1, no_species)
        initial_concentrations = np.random.uniform(0.1, 1, no_resources)

        simulation = solve_ivp(dCR_dt, [0, 3000],
                               np.concatenate(
                                   (initial_abundances, initial_concentrations)),
                               args=(no_species, growth, death,
                                     consumption, influx),
                               method='RK45')

        if np.any(np.log(np.abs(simulation.y[:, -1])) > 6):

            return None

        else:

            species_diversity = \
                np.count_nonzero(
                    np.any(simulation.y[:no_species, -20:] > 1e-2, axis=1))

            resource_diversity =  \
                np.count_nonzero(
                    np.any(simulation.y[no_species:, -20:] > 1e-2, axis=1))

            return {'Species diversity': species_diversity,
                    'Resource diversity': resource_diversity,
                    'Simulation': simulation}

    diversity_data_unclean = [resource_species_diversity() for _ in range(100)]
    diversity_data = list(
        filter(lambda item: item is not None, diversity_data_unclean))

    satisfies_competitive_exclusion = [community for community in diversity_data
                                       if community['Species diversity'] <= community['Resource diversity']]

    violates_competitive_exclusion = [community for community in diversity_data
                                      if community['Species diversity'] > community['Resource diversity']]

    return satisfies_competitive_exclusion, violates_competitive_exclusion

# %%


def detect_invasibility(simulation_t, simulation_y,
                        t_start, extinct_thresh=1e-4):
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
    baseline_abundance = 10**-6

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

def community_simulations(average_consumptions):

    no_species_to_test = np.array(
        [4, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100])

    no_communities = 50

    def recreate_invasibility_plot(no_species, no_resources,
                                   average_consumption):

        print({'Average consumption': average_consumption, 'No. species': no_species})

        growth = normal_distributed_parameters(
            1, 0.05, (no_species, no_resources))
        death = np.ones(no_species)
        influx = np.ones(no_resources)

        def simulate_community(no_species, no_resources,
                               growth, death, influx, consumption_stats):

            consumption = normal_distributed_parameters(**consumption_stats,
                                                        dims=(no_resources, no_species))

            initial_abundances = np.random.uniform(0.1, 1, no_species)
            initial_concentrations = np.random.uniform(0.1, 1, no_resources)

            first_simulation = solve_ivp(dCR_dt, [0, 1000],
                                         np.concatenate(
                                             (initial_abundances, initial_concentrations)),
                                         args=(no_species, growth, death,
                                               consumption, influx),
                                         method='LSODA', rtol=1e-6, atol=1e-9)

            new_initial_conditions = first_simulation.y[:, -1]

            if np.any(np.log(np.abs(new_initial_conditions)) > 6) \
                or np.isnan(np.log(np.abs(new_initial_conditions))).any():

                return None

            else:

                final_simulation = solve_ivp(dCR_dt, [0, 3000],
                                             new_initial_conditions,
                                             args=(no_species, growth,
                                                   death, consumption, influx),
                                             method='LSODA', rtol=1e-5, atol=1e-8)

                if np.any(np.log(np.abs(final_simulation.y[:, -1])) > 6) \
                    or np.isnan(np.log(np.abs(final_simulation.y[:, -1]))).any():

                    return None

                else:

                    species_reinvadability = detect_invasibility(final_simulation.t, final_simulation.y[:no_species, :],
                                                                 2000)

                    resource_reinvadability = detect_invasibility(final_simulation.t, final_simulation.y[:no_species, :],
                                                                  2000)
                    
                    last_500_t = np.argmax(final_simulation.t > 2500)

                    species_diversity = \
                        np.count_nonzero(np.any(final_simulation.y[:no_species, last_500_t:] > 1e-3,
                                                axis=1))/no_species

                    resource_diversity =  \
                        np.count_nonzero(np.any(final_simulation.y[no_species:, last_500_t:] > 1e-3,
                                                axis=1))/no_resources

                    return [final_simulation, species_reinvadability, resource_reinvadability, species_diversity, resource_diversity]

        messy_list = [simulate_community(no_species, no_resources, growth, death,
                                         influx, {'mu': average_consumption, 'sigma': 0.15})
                      for _ in range(no_communities)]
        cleaned_messy_list = list(
            filter(lambda item: item is not None, messy_list))

        return {'Simulations': [item[0] for item in cleaned_messy_list],
                'Species Reinvadability': [item[1] for item in cleaned_messy_list],
                'Resource Reinvadability': [item[2] for item in cleaned_messy_list],
                'Species diversity': [item[3] for item in cleaned_messy_list],
                'Resource diversity': [item[4] for item in cleaned_messy_list]}

    return {str(average_consumption):
            {str(no_species): recreate_invasibility_plot(no_species,
                                                         no_species,
                                                         average_consumption)
             for no_species in no_species_to_test}
            for average_consumption in average_consumptions}

# %%


def pickle_dump(filename, data):

    with open(filename, 'wb') as fp:

        pickle.dump(data, fp)

# %%


simulations_09 = community_simulations(np.array([0.9]))
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_09.pkl",
            simulations_09)

# %%

simulations_06 = community_simulations(np.array([0.6]))
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_06.pkl",
            simulations_06)

# %%

simulations_11 = community_simulations(np.array([1.1]))
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_11.pkl",
            simulations_11)

#%%

simulations_06 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_06.pkl")
simulations_09 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_09.pkl")
simulations_11 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_11.pkl")

# %%

def diversity_invasability(simulation_data, average_consumption):

    resource_diversities = np.concatenate([np.array(communities_sample['Resource diversity'])
                                           for communities_sample in simulation_data.values()])
    species_diversities = np.concatenate([np.array(communities_sample['Species diversity'])
                                          for communities_sample in simulation_data.values()])
    closeness_to_competitive_exclusion = species_diversities/resource_diversities

    species_reinvadabilities = np.concatenate([np.array(communities_sample['Species Reinvadability'])
                                               for communities_sample in simulation_data.values()])
    resource_reinvadabilities = np.concatenate([np.array(communities_sample['Resource Reinvadability'])
                                               for communities_sample in simulation_data.values()])
    
    annot_no_species = np.concatenate([np.repeat(np.int64(no_species), len(communities_sample['Resource diversity']))
                                       for no_species, communities_sample in simulation_data.items()])
    annot_consumption = np.concatenate([np.repeat(average_consumption, len(communities_sample['Resource diversity']))
                                       for no_species, communities_sample in simulation_data.items()])
    
    data = pd.DataFrame([annot_consumption, annot_no_species, species_reinvadabilities,
                         resource_reinvadabilities, species_diversities, resource_diversities,
                         closeness_to_competitive_exclusion], 
                        index = ['Average consumption rate', 'Number of species', 
                                 'Reinvadability (species)', 'Reinvadability (resources)',
                                 'Diversity (species)', 'Diversity (resources)',
                                 'Closeness to competitive exclusion']).T
    
    return data

# %%

def plot_diversity_invasbility(data, x, y, xlabel, ylabel):
    
    sns.set_style('white')

    cmap = mpl.cm.viridis_r
    bounds = np.append(np.sort(np.unique(data[0]['Number of species'])),100)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1,3,sharex=True,sharey=True,figsize=(8.5,5),layout='constrained')
    fig.suptitle('Avg. interaction strength',fontsize=24,weight='bold')
    fig.supxlabel(xlabel,fontsize=24,weight='bold')
    fig.supylabel(ylabel,fontsize=24,weight='bold', multialignment='center')

    clb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs,shrink=1.6,
                       pad=0.05)
    clb.ax.set_title('Species pool \n size',fontsize=22,weight='bold',pad=7.5)
    
    plt.gcf().text(0.25, 0.8, '0.6',fontsize=18,horizontalalignment='center',
                   verticalalignment='center')
    plt.gcf().text(0.51, 0.8, '0.9',fontsize=18,horizontalalignment='center',
                   verticalalignment='center')
    plt.gcf().text(0.775, 0.8, '1.1',fontsize=18,horizontalalignment='center',
                   verticalalignment='center')
    
    for i, ax in enumerate(axs.flatten()):
        
        subfig = sns.scatterplot(data[i], x = x, y = y,
                                 hue='Number of species',
                                 ax=ax,palette='viridis_r',hue_norm=norm,s=100)
        subfig.set(xlabel=None,ylabel=None)
        subfig.set_yticks(range(2))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.get_legend().remove()

    sns.despine()
       
    return fig, axs
    

# %%

data_09 = diversity_invasability(simulations_09['0.9'], 0.9)
data_06 = diversity_invasability(simulations_06['0.6'], 0.6)
data_11 = diversity_invasability(simulations_11['1.1'], 1.1)

# %%

fig1, axs1 = plot_diversity_invasbility([data_06, data_09, data_11], 'Reinvadability (species)',
                           'Diversity (species)',
                           'Re-invadability\n(instability measure)',
                           'Surivival fraction\n(species)')

for i, ax in enumerate(axs1.flatten()):

    ax.set_xticks(range(2))

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_invasibility_survivalfraction_spec.png",
            dpi=300,bbox_inches='tight')

# %%

fig2, axs2 = plot_diversity_invasbility([data_06, data_09, data_11], 'Reinvadability (resources)',
                           'Diversity (resources)',
                           'Re-invadability\n(instability measure)',
                           'Surivival fraction\n(resources)')

for i, ax in enumerate(axs2.flatten()):

    ax.set_xticks(range(2))

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_invasibility_survivalfraction_res.png",
            dpi=300,bbox_inches='tight')

# %%

fig3, axs3 = plot_diversity_invasbility([data_06, data_09, data_11], 'Reinvadability (species)',
                           'Closeness to competitive exclusion',
                           'Re-invadability\n(instability measure)',
                           'Closeness to competitive exclusion')

axs3.flatten()[0].set_ylim([0,1.25])

for i, ax in enumerate(axs3.flatten()):

    ax.set_xticks(range(2))

for ax in axs3.flat:
    
    ax.axhline(1,color='grey',ls='--',linewidth=2)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_invasibility_survivalfraction_ce.png",
            dpi=300,bbox_inches='tight')

# %%

print(pearsonr(data_09['Reinvadability (species)'],
               data_09['Closeness to competitive exclusion']))
print(pearsonr(data_11['Reinvadability (species)'],
               data_11['Closeness to competitive exclusion']))

# %%

def fluctuation_coefficient(times, dynamics):
     
    last_500_t = np.argmax(times > 2500)
    final_diversity = np.any(dynamics[:, last_500_t:] > 1e-3, axis=1)

    extant_species = dynamics[final_diversity, last_500_t:]

    return np.count_nonzero(np.std(extant_species, axis=1)/np.mean(extant_species, axis=1) > 5e-2)

#%%

data_09['Fluctuation CV (species)'] = [fluctuation_coefficient(community.t, community.y[:np.int64(no_species), :])
                                       for no_species, communities_data in simulations_09['0.9'].items()
                                           for community in communities_data['Simulations']]
data_09['Fluctuation CV (resources)'] = [fluctuation_coefficient(community.t, community.y[np.int64(no_species):, :])
                                       for no_species, communities_data in simulations_09['0.9'].items()
                                           for community in communities_data['Simulations']]
data_09['Proportion fluctuation CV (species)'] = data_09['Fluctuation CV (species)']/(data_09['Number of species'] * data_09['Diversity (species)'])

data_11['Fluctuation CV (species)'] = [fluctuation_coefficient(community.t, community.y[:np.int64(no_species), :])
                                       for no_species, communities_data in simulations_11['1.1'].items()
                                           for community in communities_data['Simulations']]
data_11['Fluctuation CV (resources)'] = [fluctuation_coefficient(community.t, community.y[np.int64(no_species):, :])
                                       for no_species, communities_data in simulations_11['1.1'].items()
                                           for community in communities_data['Simulations']]
data_11['Proportion fluctuation CV (species)'] = data_11['Fluctuation CV (species)']/(data_11['Number of species'] * data_11['Diversity (species)'])

data_06['Fluctuation CV (species)'] = [fluctuation_coefficient(community.t, community.y[:np.int64(no_species), :])
                                       for no_species, communities_data in simulations_06['0.6'].items()
                                           for community in communities_data['Simulations']]
data_06['Fluctuation CV (resources)'] = [fluctuation_coefficient(community.t, community.y[np.int64(no_species):, :])
                                       for no_species, communities_data in simulations_06['0.6'].items()
                                           for community in communities_data['Simulations']]
data_06['Proportion fluctuation CV (species)'] = data_06['Fluctuation CV (species)']/(data_06['Number of species'] * data_06['Diversity (species)'])
    
# %%

fig1, axs1 = plot_diversity_invasbility([data_06, data_09, data_11], 'Proportion fluctuation CV (species)',
                           'Diversity (species)',
                           'Fluctuation CV',
                           'Surivival fraction\n(species)')

for i, ax in enumerate(axs1.flatten()):

    ax.set_xticks(range(2))
    
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_fluct_cv_survivalfraction_spec.png",
            dpi=300,bbox_inches='tight')
    
# %%

fig3, axs3 = plot_diversity_invasbility([data_06, data_09, data_11], 'Proportion fluctuation CV (species)',
                           'Closeness to competitive exclusion',
                           'Fluctuation CV',
                           'Closeness to\ncompetitive exclusion')

axs3.flatten()[0].set_ylim([0,1.25])

for i, ax in enumerate(axs3.flatten()):

    ax.set_xticks(range(2))

for ax in axs3.flat:
    
    ax.axhline(1,color='grey',ls='--',linewidth=2)
    
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_fluct_cv_survivalfraction_ce.png",
            dpi=300,bbox_inches='tight')

#%%

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

for data, simulations, annoying_indexer in zip([data_06, data_09, data_11],
                                               [simulations_06, simulations_09, simulations_11],
                                               ['0.6', '0.9', '1.1']):
    
    data['Reinvadability rescaled (species)'] = [rescaled_detect_invasability(community.t, community.y[:np.int64(no_species), :], 2000)
                                           for no_species, communities_data in simulations[annoying_indexer].items()
                                               for community in communities_data['Simulations']]
    data['Reinvadability rescaled (resources)'] = [rescaled_detect_invasability(community.t, community.y[np.int64(no_species):, :], 2000)
                                           for no_species, communities_data in simulations[annoying_indexer].items()
                                               for community in communities_data['Simulations']]

# %%

fig1, axs1 = plot_diversity_invasbility([data_06, data_09, data_11], 'Reinvadability rescaled (species)',
                           'Diversity (species)',
                           'Re-invadability\n(instability measure)',
                           'Surivival fraction\n(species)')

for i, ax in enumerate(axs1.flatten()):

    ax.set_xticks(range(2))

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_invasibility_survivalfraction_spec.png",
            dpi=300,bbox_inches='tight')

# %%

fig2, axs2 = plot_diversity_invasbility([data_06, data_09, data_11], 'Reinvadability rescaled (resources)',
                           'Diversity (resources)',
                           'Re-invadability\n(instability measure)',
                           'Surivival fraction\n(resources)')

for i, ax in enumerate(axs2.flatten()):

    ax.set_xticks(range(2))

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_invasibility_survivalfraction_res.png",
            dpi=300,bbox_inches='tight')

# %%

fig3, axs3 = plot_diversity_invasbility([data_06, data_09, data_11], 'Reinvadability rescaled (species)',
                           'Closeness to competitive exclusion',
                           'Re-invadability\n(instability measure)',
                           'Closeness to\ncompetitive exclusion')

axs3.flatten()[0].set_ylim([0,1.25])

for i, ax in enumerate(axs3.flatten()):

    ax.set_xticks(range(2))

for ax in axs3.flat:
    
    ax.axhline(1,color='grey',ls='--',linewidth=2)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_invasibility_survivalfraction_ce.png",
            dpi=300,bbox_inches='tight')

#%%

print(pearsonr(data_09['Reinvadability rescaled (species)'],
               data_09['Closeness to competitive exclusion']))
print(pearsonr(data_11['Reinvadability rescaled (species)'],
               data_11['Closeness to competitive exclusion']))

#%%

colourmap_base = mpl.colormaps['viridis_r'](0.99)
light_dark_range = np.linspace(0.66,0,3)
lighten_func = lambda val, i : val + i*(1-val)
colours_list = [tuple([lighten_func(val,i) for val in colourmap_base[:-1]]) + (colourmap_base[-1],)
                for i in light_dark_range]
cmap = mpl.colors.ListedColormap(colours_list)

sns.set_style('white')

fig, axs = plt.subplots(2,2, figsize=(9,4), layout='constrained', height_ratios=[1, 0.1])
fig.suptitle('Prevalance of non-equilibrium community dynamics',fontsize=24,weight='bold')
fig.supylabel('Frequency Density',fontsize=18, multialignment='center')

subfig0 = sns.histplot(pd.concat([data_06, data_09, data_11]),
                       x = 'Reinvadability rescaled (species)',
                       hue = 'Average consumption rate', bins = 30,
                       element = 'step', stat="density", common_norm=False, cumulative = True,
                       fill = False, palette = cmap, ax = axs.flatten()[0],
                       linewidth = 3)

subfig0_break = sns.histplot(pd.concat([data_06, data_09, data_11]),
                       x = 'Reinvadability rescaled (species)',
                       hue = 'Average consumption rate', bins = 30,
                       element = 'step', stat="density", common_norm=False, cumulative = True,
                       fill = False, palette = cmap, ax = axs.flatten()[2],
                       linewidth = 2)
axs.flatten()[2].axhline(0.0035,color='black',linewidth=0.8)

subfig1 = sns.histplot(pd.concat([data_06, data_09, data_11]),
                       x = 'Proportion fluctuation CV (species)',
                       hue = 'Average consumption rate', bins = 30,
                       element = 'step', stat="density", common_norm=False, cumulative = True,
                       fill = False, palette = cmap, ax = axs.flatten()[1],
                       linewidth = 3)
subfig1_break = sns.histplot(pd.concat([data_06, data_09, data_11]),
                       x = 'Proportion fluctuation CV (species)',
                       hue = 'Average consumption rate', bins = 30,
                       element = 'step', stat="density", common_norm=False, cumulative = True,
                       fill = False, palette = cmap, ax = axs.flatten()[3],
                       linewidth = 2)
axs.flatten()[3].axhline(0.0035,color='black',linewidth=0.8)

axs.flatten()[0].get_xaxis().set_visible(False)
axs.flatten()[1].get_xaxis().set_visible(False)

subfig0.set_ylim([0.49,1.01])
subfig0_break.set_ylim([0,0.1])
subfig1.set_ylim([0.49,1.01])
subfig1_break.set_ylim([0,0.1])
subfig0.set_xlim([-0.01,1.01])
subfig0_break.set_xlim([-0.01,1.01])
subfig1.set_xlim([-0.01,1.01])
subfig1_break.set_xlim([-0.01,1.01])

subfig0_break.set_xticks(range(2))
subfig1_break.set_xticks(range(2))
subfig0.set_yticks([0.5,1])
subfig1.set_yticks([0.5,1])
subfig0_break.set_yticks(range(1))
subfig1_break.set_yticks(range(1))

for ax in axs.flatten():
    
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    
for ax in axs.flatten():

    ax.set_ylabel('')

axs.flatten()[0].set_xlabel('')
axs.flatten()[1].set_xlabel('')
axs.flatten()[2].set_xlabel('Reinvadability\n(instability measure)', fontsize=18)
axs.flatten()[3].set_xlabel('Fluctuation CV', fontsize=18)

for ax in axs.flatten():

    ax.get_legend().remove()

d = 0.01  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=axs.flatten()[0].transAxes, color="k", clip_on=False)

axs.flatten()[0].plot((-d, +d), (-d, +d), **kwargs)
kwargs.update(transform=axs.flatten()[1].transAxes)  
axs.flatten()[1].plot((-d, +d), (-d, +d), **kwargs)
kwargs.update(transform=axs.flatten()[2].transAxes) 
axs.flatten()[2].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
kwargs.update(transform=axs.flatten()[3].transAxes) 
axs.flatten()[3].plot((-d, +d), (1 - d, 1 + d), **kwargs)  


sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=True, offset=None, trim=False)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_instability_freq.png",
            dpi=300,bbox_inches='tight')
