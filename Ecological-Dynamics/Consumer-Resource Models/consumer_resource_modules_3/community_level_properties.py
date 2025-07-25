# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 10:24:02 2024

@author: jamil
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
from copy import deepcopy

from matplotlib import pyplot as plt

########### Community properties #############

class CommunityPropertiesInterface:
    
    def calculate_community_properties(self,
                                       average_property = False,
                                       time_window = 500):
        
        if average_property is True:
            
            start_idx = np.abs(self.ODE_sols[0].t - (self.t_end - time_window)).argmin()
            
        else:
            
            start_idx = -1
        
        species_distributions = \
            [self.__abundance_distribution(simulation.y[:self.no_species,
                                                        start_idx:])    
             for simulation in self.ODE_sols] 
            
        self.species_survival_fraction = [dist[0] 
                                          for dist in species_distributions]
        
        self.species_avg_abundance = [dist[1] 
                                      for dist in species_distributions]

        self.species_abundance_fluctuations = [dist[2] 
                                               for dist in species_distributions]
        
        resource_distributions = \
            [self.__abundance_distribution(simulation.y[self.no_species:,
                                                        start_idx:])    
             for simulation in self.ODE_sols] 
            
        self.resource_survival_fraction = [dist[0] 
                                          for dist in resource_distributions]
        
        self.resource_avg_abundance = [dist[1] 
                                      for dist in resource_distributions]

        self.resource_abundance_fluctuations = [dist[2] 
                                               for dist in resource_distributions]
            
    def __abundance_distribution(self, abundances, extinct_thresh = 1e-4):
        
        if abundances.shape[1] == 1:
            
            zeroth_moment = \
                np.count_nonzero(abundances > extinct_thresh)/len(abundances)
                
            first_moment = np.mean(abundances)
            
            second_moment = np.mean(abundances**2)
        
        elif abundances.shape[1] > 1:
            
            zeroth_moment = \
                np.mean(np.count_nonzero(abundances > extinct_thresh,
                                         axis = 0)/abundances.shape[0])
                
            first_moment = np.mean(np.count_nonzero(abundances > extinct_thresh,
                                    axis = 0)/abundances.shape[0])
            
            second_moment = np.mean(np.mean(abundances**2, axis = 0))
        
        return zeroth_moment, first_moment, second_moment
    
###########################################################################################################

def max_le(community, T, initial_conditions, extinction_threshold,
           separation = 1e-9, dt = 1):
    
    '''
    
    Calculate the average maximum lyapunov exponent for a lineage.
    See Sprott (1997, revised 2015) 'Numerical Calculation of Largest Lyapunov Exponent' 
    for more details.
    
    Protocol:
        (1) Extract initial species abundances from a simulation of lineage dynamics.
        (2) Simulate community dynamics from aforementioned initial abundances for time = dt.
        (3) Select an extant species, and perturbate its initial species abundance by separation.
            Simulate community dynamics for time = dt.
        (4) Measure the new degree of separation between the original trajectory and the
            perturbated trajectory. This is d1:
                d1 = [(S_1-S_(1,perturbated))^2+(S_2-S_(2,perturbated))^2+...]^(1/2)
        (5) Estimate the max. lyapunov exponent = (1/dt)*ln(|d1/separation|).
        (6) Reset the perturbated trajectories species abundaces so that the 
            original and perturbated trajectory are 'separation' apart:
                x_normalised = x_end + (separation/d1)*(x_(perturbated,end)-x_end).
        (7) Repeat steps 2, 4-6 n times, then calculate the average max. lyapunov exponent.
    
    Parameters
    ----------
    dict_key : string
        Lineage.
     n : int
         The number of iterations the lyapunov exponent is calculated over. The default is 10.
     dt : float, optional
         The timestep the lyapunov exponents is calculated over. The default is 7000.
     separation : float
         The amount a community is perturbated. The default is 1e-2.
     extinct_thresh : float
         Species extinction threshold. The default is 1e-4.
    
    Returns
    -------
    max_lyapunov_exponent : float
        The average maximum lyapunov exponent.
    
    '''
    
    #breakpoint()
   
    # Initialise list of max. lyapunov exponents
    log_d1d0 = []
    
    # Set initial conditions as population abundances at the end of lineage simulations
    
    # Set initial conditions of the original and perturbated trajectory
    original_conditions = deepcopy(initial_conditions)
    
    perturbed_conditions = deepcopy(initial_conditions)
    perturbed_conditions += separation/len(perturbed_conditions)
    
    current_time = 0
    
    separation_dt = separation
    separation_min = 1e-3 * separation
    separation_max = 1e3 * separation
    
    for n in range(int(np.round(T/dt))):
        
        if separation_dt > separation_min and separation_dt < separation_max:
            
            # Simulate the original community trajectory for time = dt
            simulation1 = community.simulate_community(dt, 1, init_cond_func='user supplied',
                                                         assign = False,
                                                         user_supplied_init_cond = 
                                                         {'species' : original_conditions[:community.no_species],
                                                          'resources' : original_conditions[community.no_species:]})
            # Simulate the perturbated community trajectory for time = dt
            simulation2 = community.simulate_community(dt, 1, init_cond_func='user supplied',
                                                         assign = False,
                                                         user_supplied_init_cond = 
                                                         {'species' : perturbed_conditions[:community.no_species],
                                                          'resources' : perturbed_conditions[community.no_species:]})
            
            # Get species abundances at the end of simulation from the original and perturbed trajectories
            final_dynamics1 = simulation1[0].y[:,-1]
            final_dynamics2 = simulation2[0].y[:,-1]
            
            # Calculated the new separation between the original and perturbated trajectory (d1)
            separation_dt = np.sqrt(np.sum((final_dynamics1 - final_dynamics2)**2))
            
            # Calculate the max. lyapunov exponent
            log_d1d0.append(np.log(separation_dt/(separation)))
            
            # Reset the original trajectory's species abundances to the species abundances at dt.
            original_conditions = final_dynamics1
             
            # Reset the perturbated trajectory's species abundances so that the original
            #    and perturbated community are 'separation' apart.
            perturbed_conditions = final_dynamics1 + \
                (final_dynamics2 - final_dynamics1)*(separation/separation_dt)
                
            current_time += dt
            
        else:
            
            break
        
    # Calculate average max. lyapunov exponent
    max_lyapunov_exponent = (1/(current_time)) * np.sum(np.array(log_d1d0))
    
    return max_lyapunov_exponent



#species_index = np.arange(self.no_species)
#    resource_index = np.arange(self.no_species, self.no_species + self.no_resources)
    
    ############# Volatility ##################
    
#    self.species_volatility = [self.__rescaled_detect_invasability(simulation.t,
#                                                                 simulation.y[species_index, :],
#                                                                 from_time - 500,
#                                                                 1e-2,
#                                                                 from_time)
#                               for simulation in self.ODE_sols]
        
#    self.resource_volatility = [self.__rescaled_detect_invasability(simulation.t,
#                                                                 simulation.y[resource_index, :],
#                                                                 from_time - 500,
#                                                                 1e-2,
#                                                                 from_time)
#                               for simulation in self.ODE_sols]
            
    ################ Fluctuations ############################
    
#    self.species_fluctuations = [self.__fluctuation_coefficient(simulation.t,
#                                                              simulation.y[species_index, :],
#                                                              1e-2,
#                                                              from_time) 
#                                 for simulation in self.ODE_sols]
    
#    self.resource_fluctuations = [self.__fluctuation_coefficient(simulation.t,
#                                                              simulation.y[resource_index, :],
#                                                              1e-2,
#                                                              from_time) 
#                                 for simulation in self.ODE_sols]
            
    ####################### Diversity ############################

#    self.species_survival_fraction = [self.__diversity(simulation, species_index, 1e-2, from_time)
#                                      for simulation in self.ODE_sols]
#    self.resource_survival_fraction = [self.__diversity(simulation, resource_index, 1e-2, from_time)
#                                       for simulation in self.ODE_sols]
        
        
#def __rescaled_detect_invasability(self, simulation_t, simulation_y,
#                                 t_start, extinct_thresh=1e-3,
#                                 extant_t_start = None):
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
#    t_start_index = np.where(simulation_t >= t_start)[0]

    # set baseline_abundance as slightly greater than the migration rate.
#    baseline_abundance = extinct_thresh * 10**-1

    # identifying extant/surviving species between t_start and the end of simulations
#    if extant_t_start is None:
    
#        extant_species = np.any(
#            simulation_y[:, t_start_index] > extinct_thresh, axis=1).nonzero()[0]
        
#    else:
        
#        extant_species = np.any(
#            simulation_y[:, simulation_t >= extant_t_start] > extinct_thresh, axis=1).nonzero()[0]

    # Identify which of the extant species have "fluctuating dynamics".
#    fluctuating_species = extant_species[np.logical_not(np.isnan([self.__find_normalised_peaks(simulation_y[spec, t_start_index])[0]
#                                                                  for spec in extant_species]))]  # THIS IS KINDA WRONG

    # If there are species with fluctuating dynamics present
#    if fluctuating_species.size > 0:

        # # find if and where species abundances dip below baseline_abundance.
        # Tuple entry 0 = species, Tuple entry 1 = index of the timepoint where their
        #   abundances dipped below baseline_abundance.
#        when_fluctuating_species_are_lost = np.nonzero(simulation_y[fluctuating_species, :]
#                                                       < baseline_abundance)  # THIS IS VERY WRONG

        # If species abundances dip below baseline_abundance
#        if len(when_fluctuating_species_are_lost[0]) > 0:

            # Identify the species with abundances that dip below baseline_abundance
            #   and the first entry where the unique species was identified.
#            unique_species, index = \
#                np.unique(
#                    when_fluctuating_species_are_lost[0], return_index=True)

#            reinvading_species = np.array([np.any(simulation_y[
#                fluctuating_species[when_fluctuating_species_are_lost[0][i]],
#                when_fluctuating_species_are_lost[1][i]:]
#                > baseline_abundance) for i in index])

            # count number of reinvading species
#            no_reinvading_species = np.sum(reinvading_species)

            # calculate the proportion of extant species that can reinvade the system
#            proportion_fluctuating_reinvading_species = no_reinvading_species / \
#                len(extant_species)

        # If no species abundances dip below baseline_abundance, the proportion
        #   of species that can reinvade the system (proportion_fluctuating_reinvading_species)
        #   is set to 0.
#        else:

#            proportion_fluctuating_reinvading_species = 0

    # If no species have fluctuating dynamics, the proportion of species that
    #   can reinvade the system (proportion_fluctuating_reinvading_species)
    #   is set to 0.
#    else:

#        proportion_fluctuating_reinvading_species = 0

#    return proportion_fluctuating_reinvading_species

#def __find_normalised_peaks(self, data):
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
#    peak_ind, _ = find_peaks(data)

    # If peaks are present
#    if peak_ind.size > 0:

        # get the prominance of peaks compared to surrounding data (similar to peak amplitude).
#        prominences = peak_prominences(data, peak_ind)[0]
        # get peak prominances relative to the data.
#        normalised_prominences = prominences/(data[peak_ind] - prominences)
        # select peaks from normalised prominences > 0.8
#        peak_ind = peak_ind[normalised_prominences > 0.8]

    # If peaks are present after normalisation
#    if peak_ind.size > 0:

#        return peak_ind  # return indexes of peaks

    # If peaks are not present
#    else:

#        return np.array([np.nan])  # return np.nan
    
#def __fluctuation_coefficient(self, times, dynamics, extinction_threshold, from_time):
     
#    t_start = np.argmax(times >= from_time)
#    final_diversity = np.any(dynamics[:, t_start:] > extinction_threshold, axis=1)

#    extant_species = dynamics[final_diversity, t_start:]

#    return np.count_nonzero(np.std(extant_species, axis=1)/np.mean(extant_species, axis=1) > 5e-2)

#def __diversity(self, data, index, extinction_threshold, from_time):

#    survival_fraction = \
#            np.count_nonzero(np.any(data.y[:, data.t >= from_time][index, :] > extinction_threshold,
#                                    axis=1))/len(index)
    
#    return survival_fraction
