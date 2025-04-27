# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:01:21 2024

@author: jamil
"""

import numpy as np
from copy import deepcopy
from scipy.signal import find_peaks
from scipy.signal import peak_prominences

########### Community properties #############

class CommunityPropertiesInterface:
    
    def calculate_community_properties(self, from_time = None):
        
        '''
        
        Automatically calculate community properties from a given time to the 
        end of simulations. This saves you from having to call all the functions
        for calculating different properties separately.
       
        Properties calculated: 
            species diversity
            community volatility
                                
        Parameters
        ----------
        lineages : list or np.ndarray of ints
            list of lineage indexes.
        from_time : Optional, float
            time to start calculating community properties.
            The default is None. If None, properties are calculated during the 
            last 500 time points of the simulation.

        Raises
        ------
        Exception
            If from_which_time is after the end of simulations.

        Returns
        -------
        None.

        '''
        
        if from_time:
            
            from_which_time = from_time
            
        else: 
            
            from_which_time = self.t_end - 500
        
        if self.t_end < from_which_time:
            
            raise Exception("Start time must be less than the end of simulation.")
        
        ##################### Community survival fraction (final diversity)
        
        self.survival_fraction = {'lineage ' + str(i) : 
                                     self.diversity(simulation, 1e-2, from_which_time)
                                     for i, simulation in enumerate(self.ODE_sols.values())}
           
        ############## Presence of volatile/fluctuating dynamics
        
        self.volatility = \
            {'lineage ' + str(i) : self.detect_invasibility('lineage ' + str(i),
                                                                  from_which_time) \
                 for i, simulation in enumerate(self.ODE_sols.values())}
                
    def diversity(self, data, extinction_threshold, from_time):
        
        '''
        
        Calculate the species survival fraction, or final species diversity divided
        by the initial community/species pool size.
    
        Parameters
        ----------
        data : scipy.integrate.solve_ivp object
            Lineage/simulation from a single set of initial species abundances.
        extinction_threshold : float
            Threshold value for when species are deemed extinct/extinction threshold.
        from_time : float.
            The final community diversity is calculated over the time window.
            from_time = the start time of this window.

        Returns
        -------
        survival_fraction : float
            Community survival fraction.

        '''

        survival_fraction = \
                np.count_nonzero(np.any(data.y[:, data.t >= from_time] > extinction_threshold,
                                        axis=1))/self.no_species
        
        return survival_fraction
    
    def detect_invasibility(self,lineage,t_start,extinct_thresh=1e-4):
        
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
        lineage : string 
            Lineage index.
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
        t_start_index = np.where(self.ODE_sols[lineage].t >= t_start)[0]
        
        # set baseline_abundance as slightly greater than the migration rate.
        baseline_abundance = self.dispersal * 1e2
        
        # identifying extant/surviving species between t_start and the end of simulations
        extant_species = np.any(self.ODE_sols[lineage].y[:self.no_species,t_start_index] > extinct_thresh,axis = 1).nonzero()[0]
        
        # Identify which of the extant species have "fluctuating dynamics".
        fluctuating_species = extant_species[np.logical_not(np.isnan([self.find_normalised_peaks(self.ODE_sols[lineage].y[spec,t_start_index])[0] \
                                for spec in extant_species]))] # THIS IS KINDA WRONG
        
        # If there are species with fluctuating dynamics present
        if fluctuating_species.size > 0:
            
            # # find if and where species abundances dip below baseline_abundance.
            # Tuple entry 0 = species, Tuple entry 1 = index of the timepoint where their 
            #   abundances dipped below baseline_abundance.
            when_fluctuating_species_are_lost = np.nonzero(self.ODE_sols[lineage].y[fluctuating_species,:] \
                                                            < baseline_abundance) # THIS IS VERY WRONG
                
            # If species abundances dip below baseline_abundance   
            if len(when_fluctuating_species_are_lost[0]) > 0:
            
                # Identify the species with abundances that dip below baseline_abundance
                #   and the first entry where the unique species was identified.
                unique_species, index = \
                    np.unique(when_fluctuating_species_are_lost[0],return_index=True)
                    
                reinvading_species = np.array([np.any(self.ODE_sols[lineage].y[\
                                            fluctuating_species[when_fluctuating_species_are_lost[0][i]],
                                             when_fluctuating_species_are_lost[1][i]:] \
                                                      > baseline_abundance) for i in index])
                                               
                # count number of reinvading species
                no_reinvading_species = np.sum(reinvading_species)
                
                # calculate the proportion of extant species that can reinvade the system
                proportion_fluctuating_reinvading_species = no_reinvading_species/len(extant_species)
            
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
    
    def find_normalised_peaks(self,data):
        
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
            
            return peak_ind # return indexes of peaks
        
        # If peaks are not present
        else:
              
            return np.array([np.nan]) # return np.nan
         
##################################

def max_le_gLV(model_class, T, initial_conditions, extinction_threshold,
               separation = 1e-9, dt = 1):
    
    '''
    
    Numerically estimate the community's maximum lyapunov exponent (max. LE). 
        This tells us how resistant population dynamics are to small perturbations
        in species abundances, or its linear stability.
        
    The max. LE is estimated using this algorithm:
        1.  Get a lineage's species and/or resource abundances at the end of the simulation. 
            This is the original trajectory.
        2.  Perturb the abundances by distance d_0 from the original trajectory 
            (e.g. by adding d_0/S to each variable, where S is the number of species 
             in the system. 
            This is the perturbed trajectory.
        3.  Simulate the population dynamics of the original and perturbed trajectory
            for time = dt. 
        4.  Calculate the distance between the original and perturbed community 
            at the end of simulations, separation_dt. 
        5.  Re-normalise the distance between original and perturbed trajectory to d_0. 
        6.  Repeat steps 3-5 $n$ times.
        7.  Calculate the maximum Lyapunov exponent: 

    Parameters
    ----------
    model_class : gLV object
        gLV object.
    T : float
        Max. time to estimate lyapunov exponents.
    initial_conditions : array of np.floats, size self.no_species
        The initial species abundances/start of the original trajectory.
        Usually the end of previous simulations.
    extinction_threshold : float
        Threshold value for when species are deemed extinct/extinction threshold.
    separation : float, optional
        d_0/initial separation between the original and perturbated trajectory.
        The default is 1e-9.
    dt : float, optional
        The length the original and perturbated trajectories are simulated over.
        The default is 1.

    Returns
    -------
    max_lyapunov_exponent : float
        The maximum lyapunov exponent.

    '''
    
    log_d1d0 = []
     
    # Set initial conditions of the original and perturbated trajectory
    original_conditions = deepcopy(initial_conditions)
    
    perturbed_conditions = deepcopy(initial_conditions)
    perturbed_conditions += separation/len(perturbed_conditions)
    
    # track the time for which trajectories have been simulated over 
    # (should stay less than T).
    current_time = 0
    
    # Track the separation between the original and perturbated trajectory over each dt.
    separation_dt = separation
    separation_min = 1e-3 * separation
    separation_max = 1e3 * separation
    
    for n in range(int(np.round(T/dt))):
        
        # if the separation between the original and perturbated trajectories is
        # sufficiently small
        if separation_dt > separation_min and separation_dt < separation_max:
            
            # Simulate the original community trajectory for time = dt
            simulation1 = model_class.simulate_community(np.arange(1), dt, init_cond_func=None,
                                                         assign = 'False',
                                                         usersupplied_init_conds = original_conditions.reshape((len(original_conditions),1)))
            # Simulate the perturbated community trajectory for time = dt
            simulation2 = model_class.simulate_community(np.arange(1), dt, init_cond_func=None,
                                                         assign = 'False',
                                                         usersupplied_init_conds = perturbed_conditions.reshape((len(perturbed_conditions),1)))
            # Get species abundances at the end of simulation from the original and perturbed trajectories
            final_dynamics1 = simulation1['lineage 0'].y[:,-1]
            final_dynamics2 = simulation2['lineage 0'].y[:,-1]
            
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
             
            # track the final time
            current_time += dt
            
        else:
            
            break
        
    # Calculate max. lyapunov exponent
    max_lyapunov_exponent = (1/(current_time)) * np.sum(np.array(log_d1d0))
    
    return max_lyapunov_exponent