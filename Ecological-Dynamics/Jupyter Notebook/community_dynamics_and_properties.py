# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:35:29 2024

@author: Jamila
"""

######################

# Jamila: for console - cd "Documents/PhD for github/Ecological dynamics and community selection"

#########################

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:13:48 2024

@author: Jamila
"""

################## Packages #############

import numpy as np

from scipy.integrate import solve_ivp
from scipy import stats
from scipy.signal import find_peaks
from scipy.signal import peak_prominences

from copy import deepcopy

from matplotlib import pyplot as plt

import pickle

import pandas as pd

############################### Classes ########################

class community_parameters:
    
    '''
    
    Create parameters for generalised Lotka-Volterra models. 
        The class has methods for generating growth rates and interaction matrices.
    
    '''
    
    def __init__(self,
                 no_species,
                 growth_func_name,growth_args,
                 interact_func_name,interact_args,
                 usersupplied_growth,usersupplied_interactmat,
                 dispersal):
        '''
        
        Generate or assign parameters used in a generalised Lotka-Volterra model.

        Parameters
        ----------
        no_species : int
            Number of species in species pool.
        growth_func_name : string
            Name of function used to generate growth rates.
                'fixed' - growth rates all equal 1,
                'normal' - growth rates are generated from normal(mu_g,sigma_g).
        growth_args : dict.
            Arguments for function used to generate growth rates, if required.
        interact_func_name : string
            Name of function used to generate the interaction matrix.
                'random' - random interaction matrix generated from normal(mu_a,sigma_a),
                'random normalised by K' - random interaction matrix generated from 
                    normal(mu_a,sigma_a), normalised by species carrying capacity K,
                    drawn from a normal distribution.
        interation_args : dict.
            Arguments for function used to generate the interaction matrix, if required.
        usersupplied_growth : None or np.array() of floats, size (no_species,)
            User-supplied array of growth rates.
        usersupplied_interactmat : None or np.array() of floats, size (no_species,)
            User-supplied interaction matrix.
        dispersal : float
            Species dispersal/migration rate.

        Returns
        -------
        None.

        '''
        
        self.no_species = no_species
        
        ############### Growth rates ####################
        
        # Select function to generate growth rates.
        match growth_func_name:
            
            case 'fixed':
                
                self.growth_rates = self.growth_rates_fixed()
                
            case 'normal':
                
                for key, value in growth_args.items():
                    
                    # Assign growth function arguments as class attributes.
                    #   (Growth function arguments are parameters for 
                    #   the growth rate distribution,)
                    setattr(self,key,value)
                    
                self.growth_rates = self.growth_rates_norm()
                
            case None:
                
                # Assign growth rates using the user-supplied growth rates
                self.growth_rates = usersupplied_growth
            
        ###################### Interaction Matrix #############
        
        match interact_func_name:
            
            case 'random':
                
                for key, value in interact_args.items():
                    
                    # Assign interaction matrix function arguments as class attributes.
                    setattr(self,key,value)
                    
                # Generate interaction matrix 
                self.interaction_matrix = self.random_interaction_matrix()
                
            case None:
                
                # Assign interaction matrix using the user-supplied interaction matrix, if supplied.
                self.interaction_matrix = usersupplied_interactmat    
            
        self.dispersal = dispersal
        
        
########### Class methods #################

    ###### Growth Rates ###########
    
    def growth_rates_norm(self):
        
        '''
        
        Draw growth rates for n species from normal(mu,sigma) distribution

        Parameters
        ----------
        mu_g : float
            Mean growth rate.
        sigma_g : float
            Standard deviation in growth rate.
        no_species : int
            Number of species (n).

        Returns
        -------
        growth_r : np.array of float64.
            array of growth rates for each species drawn from normal(mu_g,sigma_g).

        '''
        
        growth_r = self.mu_g + self.sigma_g*np.random.rand(self.no_species)
        
        return growth_r

    def growth_rates_fixed(self):
        
        '''
        
        Generate array of growth rates all fixed to 1.
        
        Parameters
        ----------
        no_species : int
            number of species.

        Returns
        -------
        growth_r : np.array of float64.
            array of growth rates, all entries = 1.0.

        '''
        
        growth_r = np.ones((self.no_species,))
        
        return growth_r
    
        
    ###### Interaction Matrix ######
            
    def random_interaction_matrix(self):
        
        '''
    
        Parameters
        ----------
        mu_a : float
            mean interaction strength.
        sigma_a : float
            interaction strength standard deviation.
         no_species : int
             number of species (n).
    
        Returns
        -------
        interact_mat : np.array of size (n,n)
            Interaction matrix. 
    
        '''
        
        # generate interaction matrix drawn from normal(mu_a,sigma_a)
        interact_mat = self.mu_a + self.sigma_a*np.random.randn(self.no_species,self.no_species)
        # set a_ij = -1 for i = j/self-interaction to prevent divergence
        np.fill_diagonal(interact_mat, 1)
        
        return interact_mat
    
################

class gLV:
    
    '''
    
    Run gLV simulations from initial conditions.
        Takes community_parameters class object as an argument, which contains model parameters.
        Has class methods for generating initial species abundances and running gLV ODE simulations.
    
    '''
    
    def __init__(self,
                 community_parameters_object,
                 t_end,
                 init_cond_func_name=None,
                 usersupplied_init_cond=None):
        
        '''
        Assign class attributes, generate initial conditions, and run simulations.
        
        Parameters
        ----------
        community_parameters_object : object of class community_parameters.
            ...
        t_end : float
            End of simulation.
        init_cond_func_name : string
            Name of function used to generate initial species abundances.
                'Hu' - function from Hu et al. (2022),
                'Mallmin' - function from Mallmin et al. (unpublished).
        usersupplied_init_cond : None or np.array, size (no_species,)
            User-supplied initial species abundances, default None.
        
        Returns
        -------
        None
        
        '''
        
        # Assign attributes from community_parameters_object to gLV
        self.no_species = community_parameters_object.no_species
        
        self.growth_rates = community_parameters_object.growth_rates
        self.interaction_matrix = community_parameters_object.interaction_matrix
        self.dispersal = community_parameters_object.dispersal
        
        self.t_end = t_end
        
        ######## Generate initial species abundances ###########
        
        # Select function used to generate initial species abundances.
        match init_cond_func_name:
            
            case 'Hu':
                
                self.initial_abundances = self.initial_abundances_hu(community_parameters_object.mu_a)
            
            case 'Mallmin':
                
                self.initial_abundances = self.initial_abundances_mallmin()
            
            case None:
                
                # Assign initial conditions using the user-supplied initial abundances.
                self.initial_abundances = usersupplied_init_cond
        
        self.ODE_sol = self.gLV_simulation(t_end)
      
    ########## Functions for generating initial conditions ############
      
    def initial_abundances_mallmin(self):
        
        '''
        
        Generate initial species abundances, based on the function from Mallmin et al. (2023).
        
        Parameters
        ----------
        no_species : int
            Number of species.
        dispersal : float.
            Dispersal or migration rate.
        
        Returns
        -------
        np.array of float64, size (n,). Drawn from uniform(min=dispersal,max=2/no_species)
        
        '''
        
        return np.random.uniform(self.dispersal,2/self.no_species,self.no_species)

    def initial_abundances_hu(self,mu_a):
        
        '''
        
        Generate initial species abundances, based on the function from Hu et al. (2022).
        
        Parameters
        ----------
        no_species : int
            Number of species.
         mu_a : float
             mean interaction strength.
        
        Returns
        -------
        np.array of float64, size (n,). Drawn from uniform(min=0,max=2*mu_a)
        
        '''
        
        return np.random.uniform(0,2*mu_a,self.no_species)

    ####### Simulate dynamics ###############
    
    def gLV_simulation(self,t_end):
        
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
        
        return solve_ivp(gLV_ode_with_extinction_threshold,[0,t_end],self.initial_abundances,
                         args=(self.growth_rates,self.interaction_matrix,self.dispersal),
                         method='RK45',t_eval=np.linspace(0,t_end,200))
    
    ########### Community properties #############
    
    def identify_community_properties(self):
        
        '''
        
        Identify community properties.

        Parameters
        ----------
        t_end : float
            End time for calculating community properties. Typically the end of simulation.

        Returns
        -------
        None.

        '''
        
        t_end_minus_last20percent = 0.8*self.t_end
        t_end_minus_last30percent = 0.7*self.t_end
        
        ###### Calculate diversity-related properties ###########
        
        final_popdyn = self.species_diversity([t_end_minus_last20percent,self.t_end])
        
        self.final_diversity = final_popdyn[1]
        self.final_composition = np.concatenate((np.where(final_popdyn[0] == True)[0],
                                                 np.zeros(self.ODE_sol.y.shape[0]-self.final_diversity)))
        
        ########## Determine if the community is fluctuating ###############
       
        self.invasibility = self.detect_invasibility(t_end_minus_last30percent)
        
    def species_diversity(self,timeframe,extinct_thresh=1e-4):
        
        '''
        
        Calculate species diversity at a given time.
        
        Parameters
        ----------
        extinct_thresh : float
            Species extinction threshold.
        ind : int
            Index of time point to calculate species diversity (to find species populations at the right time)
        simulations : OdeResult object of scipy.integrate.solve_ivp module
            (Deterministic) Solution to gLV ODE system.

        Returns
        -------
        Species present, species diversity (no. species), species abundances

        '''
        
        simulations_copy = deepcopy(self.ODE_sol)
        
        # find the indices of the nearest time to the times supplied in timeframe
        indices = find_nearest_in_timeframe(timeframe,simulations_copy.t)    
        
        # find species that aren't extinct aka species with abundances greater than the extinction threshold.
        present_species = np.any(simulations_copy.y[:,indices[0]:indices[1]] > extinct_thresh,
                                 axis=1)
        
        # calculate species diversity (aka no. of species that aren't extinct, 
        #   or the length of the array of present species abundances)
        diversity = np.sum(present_species)
         
        return [present_species,diversity]
    
    def detect_invasibility(self,t_start,extinct_thresh=1e-4):
        
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
        t_start_index = np.where(self.ODE_sol.t >= t_start)[0]
        
        # set baseline_abundance as slightly greater than the migration rate.
        baseline_abundance = self.dispersal * 1e2
        
        # identifying extant/surviving species between t_start and the end of simulations
        extant_species = np.any(self.ODE_sol.y[:,t_start_index] > extinct_thresh,axis = 1).nonzero()[0]

        # Identify which of the extant species have "fluctuating dynamics".
        fluctuating_species = extant_species[np.logical_not(np.isnan([find_normalised_peaks(self.ODE_sol.y[spec,t_start_index])[0] \
                                for spec in extant_species]))]
        
        # If there are species with fluctuating dynamics present
        if fluctuating_species.size > 0:
            
            # find if and where species abundances dip below baseline_abundance.
            # Column 0 = species, column 1 = index of the timepoint where their 
            #   abundances dipped below baseline_abundance.
            when_fluctuating_species_are_lost = np.argwhere(self.ODE_sol.y[fluctuating_species,t_start_index[0]:] \
                                                            < baseline_abundance)
             
            # If species abundances dip below baseline_abundance
            if when_fluctuating_species_are_lost.size > 0:
            
                # Identify the species with abundances that dip below baseline_abundance
                unique_species, index = np.unique(when_fluctuating_species_are_lost[:,0],
                                                  return_index=True)
                
                when_fluctuating_species_are_lost = when_fluctuating_species_are_lost[index,:]
                
                # Identify whether the species with abundances that dip below 
                #   baseline_abundance reinvade at a later timepoint.
                reinvading_species = np.array([np.any(self.ODE_sol.y[\
                                            fluctuating_species[when_fluctuating_species_are_lost[i,0]],
                                             (t_start_index[0] + when_fluctuating_species_are_lost[i,1]):] \
                                                      > baseline_abundance) for i in \
                                               range(when_fluctuating_species_are_lost.shape[0])])
                    
                proportion_fluctuating_reinvading_species = np.sum(reinvading_species)/len(extant_species)
                
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
        
################

class community(community_parameters):
    
   '''
   Generate a species pool (aka generate model parameters), then simulate community
       dynamics using the generalised Lotka-Volterra model from multiple initial conditions.
   Each initial condition is called a 'lineage'.
        
   This class inherits from the community_parameters class to generate model parameters,
       then class the gLV class to run simulations.
        
   This class also calculates community properties, such as species diversity,
       % of the community with fluctuating dynamics, and the number
       of unique compositions that can be produced from a single species pool.
   '''
    
   def __init__(self,
                no_species,
                growth_func_name, growth_args,
                interact_func_name, interact_args,
                usersupplied_growth=None,usersupplied_interactmat=None,
                dispersal=1e-8):
       '''
       
       Generate model parameters (by inheriting from community_parameters),
           initialise attributes that store community properties.
           
       Parameters
       ----------
       no_species : int
           Number of species in species pool.
       growth_func_name : string
           Name of function used to generate growth rates.
               'fixed' - growth rates all equal 1,
               'normal' - growth rates are generated from normal(mu_g,sigma_g).
       growth_args : dict.
           Arguments for function used to generate growth rates, if required.
       interact_func_name : string
           Name of function used to generate the interaction matrix.
               'random' - random interaction matrix generated from normal(mu_a,sigma_a),
               'random normalised by K' - random interaction matrix generated from 
                   normal(mu_a,sigma_a), normalised by species carrying capacity K,
                   drawn from a normal distribution.
       interation_args : dict.
           Arguments for function used to generate the interaction matrix, if required.
       usersupplied_growth : None or np.array() of floats, size (no_species,), optional
           User-supplied array of growth rates. The default is None.
       usersupplied_interactmat : None or np.array() of floats, size (no_species,), optional
           User-supplied interaction matrix. The default is None.
       dispersal : float, optional
           Species dispersal/migration rate. The default is 1e-8.

       Returns
       -------
       None.

       '''
       
       super().__init__(no_species,
                        growth_func_name, growth_args,
                        interact_func_name, interact_args,
                        usersupplied_growth,usersupplied_interactmat,
                        dispersal)
       
       # Initialise attributes for storing properties for each lineage
       
       # Initial species abundances of each lineage
       self.initial_abundances = {}
       
       # gLV simulation results for each lineage
       self.ODE_sols = {}
       
       ######### Emergent properties ###########
       
       # Number of unique species compositions per species pool
       #    (This initialisation is unnecessary and un-pythonic, but I like to do it for readability.)
       self.no_unique_compositions = None
       self.unique_composition_label = {}
       
       # Species composition (species presence/absence) at the end of simulation
       self.final_composition = {}
       
       # Species diversity for each lineage at the end of simulations
       self.diversity = {}
       
       # % species that fluctuate and can reinvade the community
       self.invasibilities = {}
       
   ########################
      
   def simulate_community(self,
                          lineages,t_end,func_name,
                          init_cond_func_name=None,array_of_init_conds=None):
       
       '''
       
       Simulate community dynamics and calculate community properties for each 
           lineage sampled from the species pool.
       
       Parameters
       ----------
       t_end : float
           End of simulation.
       func_name : string
           Name of function used to supply initial conditions.
               'Default' : Use a function, supplied by init_cond_func_name, to
                   generate different initial species abundances for each lineage.
               'Supply initial conditions' : The user supplies initial species
                   abundances for each lineage.
       lineages : np.array of ints
           Index/label for lineages generated from the species pool. 
           Typically generated from np.arange or np.linspace.
       init_cond_func_name : string, optional
           Name of function used to generate initial conditions, if the user selects
               'Default'. The default is None.
       array_of_init_conds : list of np.array of floats, optional
           Arrays of initial species abundances, if the user selects 'Supply 
               initial conditions'. The default is None.

       Returns
       -------
       None.

       '''
       
       ################ Generate initial species abundances and simulate lineage dynamics ##################
       
       match func_name:
            
           case 'Generate initial conditions':
                
               for lineage in lineages:
                     
                    # Call gLV class to simulate community dynamics
                    gLV_res = gLV(self,t_end,init_cond_func_name)
                     
                    # Calculate community properties, assign to class attributes
                    gLV_res.identify_community_properties()
                    self.assign_gLV_attributes(gLV_res, lineage)
          
           case 'Supply initial conditions':
              
               for count, lineage in enumerate(lineages):
                   
                   # Call gLV class to simulate community dynamics
                   gLV_res = gLV(self,t_end,usersupplied_init_cond=array_of_init_conds[:,count])
                     
                   # Calculate community properties, assign to class attributes
                   gLV_res.identify_community_properties()
                   self.assign_gLV_attributes(gLV_res, lineage)
       
       # Calculate the number of unique species compositions for the species pool
       no_uniq_compositions, comps = self.unique_compositions()
       
       self.no_unique_compositions = no_uniq_compositions
       self.unique_composition_label = {'lineage '+ str(lineage) : comp for lineage, comp in zip(lineages, comps)}
       
   def assign_gLV_attributes(self,gLV_res,lineage):
       
       '''
       
       Assign community properties to class attributes

       Parameters
       ----------
       gLV_res : object of class gLV
           gLV object/simulation results.
       lineage : int
           Lineage index/label.

       Returns
       -------
       None.

       '''
       
       dict_key = "lineage " + str(lineage)
       
       # Assign initial species abundances
       self.initial_abundances[dict_key] = gLV_res.initial_abundances
       
       # Assign simulation results
       self.ODE_sols[dict_key] = gLV_res.ODE_sol
       
       # Assign species composition at the end of simulation
       self.final_composition[dict_key] = gLV_res.final_composition
       
       # Assign species diversity at the end of simulation
       self.diversity[dict_key] = gLV_res.final_diversity
       
       # Assign community invasibility from species pool
       self.invasibilities[dict_key] = gLV_res.invasibility
            
   def unique_compositions(self):
       
       '''
       
       Calculate the number of unique final species compositions in a community.
       
       Returns
       -------
       no_uniq_comp : int
           Number of unique compositions
       
       '''
       
       # Assemble all species compositions from each lineage into a 2d numpy array/matrix. 
       all_compositions = np.vstack(list(self.final_composition.values()))
       
       # Identify unique rows in the 2d array of species compositions.
       #    This is the same as identifying the number of unique species compositions 
       #    for the species pool.
       # Also label each lineage with whichever unique species composition it belongs to.
       uniq_comp, comps = np.unique(all_compositions,axis=0,return_inverse=True)
       
       # Calculate the number of unique compositions.
       no_uniq_comp = len(uniq_comp)
       
       return [no_uniq_comp, comps]
    
####################### Functions #####################

############################ gLV simulations ##################

def gLV_ode_with_extinction_threshold(t,spec,growth_r,interact_mat,dispersal,
                                      extinct_thresh=1e-9):
    
    '''
    
    ODE system from generalised Lotka-Volterra model. 
    
    Removes species below some extinction threshold to cap abundances species can
    reinvade from and removes very small values that could cause numerical instability.
    This is useful when dispersal = 0.
    

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
    dispersal : float
        Dispersal or migration rate.
    extinct_thresh : float
        Extinction threshold.

    Returns
    -------
    dSdt : np.array of float64, size (n,)
        array of change in population dynamics at time t aka dS/dt.

    '''
    spec[spec < extinct_thresh] = 0 # set species abundances below extinction threshold to 0
    
    dSdt = np.multiply(1 - np.matmul(interact_mat,spec), growth_r*spec) + dispersal
    
    return dSdt

####################### Random Global Functions ###############

def generate_distribution(mu_maxmin,std_maxmin,dict_labels=['mu_a','sigma_a'],
                          mu_step=0.1,std_step=0.05):
    
    '''
    
    Generate parameters for the random interaction distribution.

    Parameters
    ----------
    mu_maxmin : list of floats
        Minimum and maximum mean interaction strength, mu_a.
    std_maxmin : list of floats
        Minimum and maximum standard deviation in interaction strength, sigma_a.
    mu_step : float, optional
        mu_a step size. The default is 0.1.
    std_step : float, optional
        sigma_a step size. The default is 0.05.

    Returns
    -------
    distributions : list of dicts
        Parameters for interaction distributions - [{'mu_a':mu_min,'sigma_a':std_min},
                                                    {'mu_a':mu_min,'sigma_a':std_min+std_step},...,
                                                    {'mu_a':mu_min+mu_step,'sigma_a':std_min},...,
                                                    {'mu_a':mu_max,'sigma_a':std_max}]

    '''
    
    # Extract min. mean interaction strength
    mu_min = mu_maxmin[0]
    # Extract max. mean interaction strength
    mu_max = mu_maxmin[1]
    
    # Extract min. standard deviation in interaction strength
    std_min = std_maxmin[0]
    # Extract max. standard deviation in interaction strength
    std_max = std_maxmin[1]
    
    # Generate range of mean interaction strengths
    mu_range = np.arange(mu_min,mu_max,mu_step)
    # Generate range of standard deviations in interaction strengths
    std_range = np.arange(std_min,std_max,std_step)

    # Generate dictionary of interaction distribution parameters sets
    mu_rep = np.repeat(mu_range,len(std_range))
    std_rep = np.tile(std_range,len(mu_range))
    
    distributions = [{dict_labels[0]:np.round(mu,2), dict_labels[1]:np.round(sigma,2)} \
                     for mu, sigma in zip(mu_rep,std_rep)]
     
    return distributions
   
def find_nearest_in_timeframe(timeframe,simulation_times):
    
    '''
    
    Find the index of the nearest times to those in timeframe 
        (for extracting population dynamics at a given time).
    

    Parameters
    ----------
    timeframe : list of ints or floats
        List of times.
    simulation_times : .t from OdeResult object of scipy.integrate.solve_ivp module
        Simulation times ONLY from (deterministic) solution to gLV ODE system.
    

    Returns
    -------
    indices : int
        indices of times in simulation_times with value

    '''
    
    indices = find_nearest_multivalues(timeframe,simulation_times)
    
    return indices

def find_nearest_multivalues(array_of_values,find_in):
    
    '''
    
    Find nearest value in array for multiple values. Vectorised.
    
    Parameters
    ----------
    array_of_values : np.array of floats or inds
        array of values.
    find_in : np.array of floats or inds
        array where we want to find the nearest value (from array_of_values).
    
    Returns
    -------
    fi_ind[sorted_idx-mask] : np.array of inds
        indices of elements from find_in closest in value to values in array_of_values.
    
    '''
     
    L = find_in.size # get length of find_in
    fi_ind = np.arange(0,find_in.size) # get indices of find_in
    
    sorted_idx = np.searchsorted(find_in, array_of_values)
    sorted_idx[sorted_idx == L] = L-1
    
    mask = (sorted_idx > 0) & \
    ((np.abs(array_of_values - find_in[sorted_idx-1]) < np.abs(array_of_values - find_in[sorted_idx])))
    
    return fi_ind[sorted_idx-mask]

def mean_stderror(data):
    
    '''
    
    Calculate the mean and standard error of a dataset.
    
    Parameters
    ----------
    data : np.array
        Dataset.
    Returns
    -------
    [mean, std_error] : list of floats
        The mean and standard error of data.
    
    '''
    
    mean = np.mean(data)
    
    std_error = stats.sem(data)
    
    return [mean, std_error]

def mean_std_deviation(data):
    
    '''
    
    Calculate the mean and standard error of a dataset.
    
    Parameters
    ----------
    data : np.array
        Dataset.
    Returns
    -------
    [mean, std_error] : list of floats
        The mean and standard error of data.
    
    '''
    
    mean = np.mean(data)
    
    std_deviation = np.std(data)
    
    return [mean, std_deviation]

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
        
        return peak_ind # return indexes of peaks
    
    # If peaks are not present
    else:
          
        return np.array([np.nan]) # return np.nan
    
def community_object_to_df(community_object,
                         community_attributes=['mu_a','sigma_a','no_species',
                                               'no_unique_compositions','unique_composition_label',
                                               'diversity','invasibilities'],
                         community_label=0):
    
    no_lineages = len(community_object.ODE_sols)
    
    community_col = np.repeat(community_label,no_lineages)
    
    lineage_col = list(community_object.ODE_sols.keys())
    lineage_col = [int(lineage.replace('lineage ','')) for lineage in lineage_col]
    
    ###############################################
    
    def extract_attribute_make_df_col(community_object,attribute_name,no_lineages=no_lineages):
        
        try:
        
            attribute = getattr(community_object,attribute_name)
            
        except AttributeError:
        
            raise Exception('Community object has no attribute ' + str(attribute_name))
            
            exit
        
        if isinstance(attribute,(int,float,str,np.int32,np.int64,np.float32,np.float32)):
            
            attribute_col = np.repeat(attribute,no_lineages)
            
        elif isinstance(attribute,dict):
            
            attribute_col = list(attribute.values())
        
        elif isinstance(attribute,(list,np.ndarraytuple)):
            
            attribute_col = attribute
        
        return attribute_col
    
    attribute_columns = [community_col] + [lineage_col] + \
        [extract_attribute_make_df_col(community_object, attribute_name) \
             for attribute_name in community_attributes]
        
    col_names = ['community','lineage'] + community_attributes
        
    ############# Convert lists to df ################
    
    community_df = pd.DataFrame(attribute_columns)
    community_df = community_df.T
    community_df = community_df.set_axis(col_names,axis=1)
    
    return community_df
    
def pickle_dump(filename,data):
    
    '''
    
    Pickle data.

    Parameters
    ----------
    filename : string
        Pickle file name. Should end with .pkl
    data : any
        Data to pickle.

    Returns
    -------
    None.

    '''
    
    with open(filename, 'wb') as fp:
        
        pickle.dump(data, fp)
