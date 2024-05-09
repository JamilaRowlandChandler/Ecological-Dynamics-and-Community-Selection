# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:01:21 2024

@author: jamil
"""

import numpy as np
from copy import deepcopy
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
import sys

from utility_functions import *

########### Community properties #############

class CommunityPropertiesInterface:
    
    def calculate_community_properties(self,lineages,from_which_time):
        
        '''
        
        Automatically calculate community properties from a given time to the 
        end of simulations. This saves you from having to call all the functions
        for calculating different properties separately.
       
        Properties calculated: 
            species diversity
            species composition
            invasibility/reinvadability
            unique species compositions 
            community function, if applicable
                                
        Parameters
        ----------
        lineages : list or np.ndarray of ints
            list of lineage indexes.
        from_which_time : float
            time to start calculating community properties.

        Raises
        ------
        Exception
            If from_which_time is after the end of simulations.

        Returns
        -------
        None.

        '''
        
        if self.t_end < from_which_time:
            
            raise Exception("Start time must be less than the end of simulation.")
        
        ###### Calculate diversity-related properties ###########
        
        self.final_diversity, self.final_composition = {}, {}
        
        for lineage in lineages:
            
            lineage_key = 'lineage ' + str(lineage)
            
            # returns species diversity and composition between from_which_time and t_end
            final_popdyn = \
                self.species_diversity(lineage_key,[from_which_time,self.t_end])
                
            self.final_composition[lineage_key] = final_popdyn[0]
            self.final_diversity[lineage_key] = final_popdyn[1]
        
        # calculates average diversity per time window between from_which_time and t_end
        self.average_diversity_over_time = \
            {'lineage ' + str(lineage) : self.average_diversity_at_time_t('lineage ' + str(lineage),
                                                                          [from_which_time,self.t_end]) \
                 for lineage in lineages}
       
        # calculate community invasibility/re-invadability between from_which_time and t_end
        self.invasibility = \
            {'lineage ' + str(lineage) : self.detect_invasibility('lineage ' + str(lineage),
                                                                  from_which_time) \
                 for lineage in lineages}
        
        # Calculate the number of unique species compositions for the species pool
        no_uniq_compositions, comps = self.unique_compositions()
        
        self.no_unique_compositions = no_uniq_compositions
        # label each lineage with its unique composition identifier
        self.unique_composition_label = {'lineage '+ str(lineage) : comp for lineage, comp in zip(lineages, comps)}
        
        # optional - calculate properties associated with community function
        if hasattr(self, 'species_contribution_community_function'):
            
            self.community_function = \
                {'lineage ' + str(lineage) : self.community_function_totalled_over_maturation(lineage) \
                     for lineage in lineages}
    
    def species_diversity(self,lineage,timeframe,extinct_thresh=1e-4):
        
        '''
        
        Calculate species diversity and composition

        Parameters
        ----------
        lineage : int
            Lineage index.
        timeframe : list or array of flaots
            start and end time/timerange to calculate diversity in.
        extinct_thresh : float, optional
            Extinction threshold The default is 1e-4.

        Returns
        -------
        list
            Species composition and diversity.

        '''
        
        simulations_copy = deepcopy(self.ODE_sols[lineage])
        
        # find the indices of the nearest time to the times supplied in timeframe
        indices = find_nearest_in_timeframe(timeframe,simulations_copy.t)    
        
        # find species that aren't extinct aka species with abundances greater than the extinction threshold.
        present_species = \
            np.any(simulations_copy.y[:self.no_species,indices[0]:indices[1]] > extinct_thresh,
                   axis=1)
        
        # calculate species diversity (aka no. of species that aren't extinct, 
        #   or the length of the array of present species abundances)
        diversity = np.count_nonzero(present_species)
         
        return [present_species.nonzero()[0],diversity]
    
    def average_diversity_at_time_t(self,lineage,timeframe,extinct_thresh=1e-4):
        
        '''
        
        Calculate average species diversity and composition across a sliding window.

        Parameters
        ----------
        lineage : int
            Lineage index.
        timeframe : list or array of flaots
            start and end time/timerange to calculate diversity in.
        extinct_thresh : float, optional
            Extinction threshold The default is 1e-4.

        Returns
        -------
        list
            Species composition and diversity.

        '''
        
        # create sliding window
        window_shape = [self.no_species,10]
        
        simulations_copy = deepcopy(self.ODE_sols[lineage])
        
        # find the indices of the nearest time to the times supplied in timeframe
        indices = find_nearest_in_timeframe(timeframe,simulations_copy.t)    
        
        abundances_sliding_window = \
            np.lib.stride_tricks.sliding_window_view(simulations_copy.y[:self.no_species,indices[0]:indices[1]],
                                                     window_shape)
        
        present_species = np.any(abundances_sliding_window > extinct_thresh,axis=3)[0]
        # calculate species diversity (aka no. of species that aren't extinct, 
        #   or the length of the array of present species abundances)
        average_diversity = np.count_nonzero(present_species,axis=1).mean()
         
        return average_diversity
    
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
    
    def unique_compositions(self):
        
        '''
        
        Calculate the number of unique final species compositions in a community.
        
        Returns
        -------
        no_uniq_comp : int
            Number of unique compositions
        
        '''
        
        final_composition_array = \
            np.vstack([np.pad(composition,(0,self.no_species-len(composition)),
                              mode='constant', constant_values=0) \
                              for composition in self.final_composition.values()])
      
        # Identify unique rows in the 2d array of species compositions.
        #    This is the same as identifying the number of unique species compositions 
        #    for the species pool.
        # Also label each lineage with whichever unique species composition it belongs to.
        uniq_comp, comps = np.unique(final_composition_array,axis=0,return_inverse=True)
        
        # Calculate the number of unique compositions.
        no_uniq_comp = len(uniq_comp)
        
        return [no_uniq_comp, comps]
    
    ############# Community function ################
      
    def community_function_totalled_over_maturation(self,lineage):
        
        '''
        
        Additive community function, calculated over maturation

        Parameters
        ----------
        lineage : int
            Lineage index.

        Returns
        -------
        community_function : float
            community function.

        '''

        summed_abundances = np.sum(self.ODE_sols[lineage].y[:self.no_species,:],axis=1)
        
        community_function = np.sum(np.multiply(self.species_contribution_community_function,
                                                summed_abundances))
        
        return community_function
        