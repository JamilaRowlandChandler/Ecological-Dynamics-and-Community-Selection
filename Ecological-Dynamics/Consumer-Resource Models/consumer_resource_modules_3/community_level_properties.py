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
        '''
        
        Call methods that calculate the moments of the species and resource
        abundance distribution, then assign them as attributes to the 
        consumer resource model object.

        Parameters
        ----------
        average_property : Bool, optional
            Whether or not to calculate community properties at the end of simulations (False)
            or over as an average over some time period (True). The default is False.
        time_window : float, optional
            If average_property is True, this is the time window to calculate
            community properties over. The default is 500.

        Returns
        -------
        None.

        '''
        
        if average_property is True:
            
            # calculate average community property over some time window
            
            start_idx = np.abs(self.ODE_sols[0].t - (self.t_end - time_window)).argmin()
            
        else:
            
            # calculate community properties at the end of simulations 
            
            start_idx = -1
            
        # moments of the species abundance distribution 
        
        # extract species abundance distribution and calculate its 0th-2nd moment
        species_distributions = \
            [self.__abundance_distribution(simulation.y[:self.no_species,
                                                        start_idx:])    
             for simulation in self.ODE_sols] 
        
        # assign species survival fraction (phi_N, or S*/S) 
        self.species_survival_fraction = [dist[0] 
                                          for dist in species_distributions]
        
        # assign average species abundance (<N>)
        self.species_avg_abundance = [dist[1] 
                                      for dist in species_distributions]
        
        # assign 2nd moment in speices abundance distribution (<N^2>)
        self.species_abundance_fluctuations = [dist[2] 
                                               for dist in species_distributions]
        
        # moments of the resource abundance distribution (same process as with species)
        
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
        
        '''
        
        Calculate the moments of the species and resource abundance distribution

        Parameters
        ----------
        abundances : np.ndarray
            Abundances of some variable, either over a time frame or at the end of simulations.
        extinct_thresh : float, optional
            Extinction threshold for the variable. The default is 1e-4.

        Returns
        -------
        zeroth_moment : float
            0th moment of the abundance distribution.
        first_moment : float
            1st moment of the abundance distribution.
        second_moment : float
            1st moment of the abundance distribution.

        '''
        
        if abundances.shape[1] == 1: # one abundance distribution, not over time period
            
            zeroth_moment = \
                np.count_nonzero(abundances > extinct_thresh)/len(abundances)
                
            first_moment = np.mean(abundances)
            
            second_moment = np.mean(abundances**2)
        
        elif abundances.shape[1] > 1: # multiple abundance distributions over time period
            
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
