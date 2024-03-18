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
from scipy.fft import rfft
from scipy.fft import rfftfreq

from copy import deepcopy

import collections.abc

from matplotlib import pyplot as plt

import pickle

import itertools

import pandas as pd

########################################

################ Classes ########################

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
        
        ###############
        
        if usersupplied_growth is None:
            
            if growth_args:
            
                for key, value in growth_args.items():
                    
                    # Assign growth function arguments as class attributes.
                    #   (Growth function arguments are parameters for 
                    #   the growth rate distribution,)
                    setattr(self,key,value)
            
            # Select function to generate growth rates.
            growth_func = {"fixed": self.growth_rates_fixed,
                           "normal": self.growth_rates_norm}[growth_func_name]
            
            # Generate growth rates
            self.growth_rates = growth_func()
            
        else:
            
            # Assign growth rates using the user-supplied growth rates, if supplied.
            self.growth_rates = usersupplied_growth
            
        ############
        
        for key, value in interact_args.items():
            
            # Assign interaction matrix function arguments as class attributes.
            setattr(self,key,value)
        
        if usersupplied_interactmat is None:
            
            # Select function to generate interaction matrix.
            interaction_func = {"random": self.random_interaction_matrix,
                                "random normalised by K": 
                                    self.random_interaction_matrix_norm_by_K}[interact_func_name]
            
            # Generate interaction matrix
            self.interaction_matrix = interaction_func()
            
        else: 
            
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
    
    def random_interaction_matrix_norm_by_K(self):
        
        '''
        
        Random interaction matrix weighted by carrying capacity. From Hu et al. (2022) github
    
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
        interact_mat_norm : np.array of size (n,n)
            Interaction matrix. 
    
        '''
        
        # Generate random interaction matrix
        interact_mat = self.mu_a + self.sigma_a*np.random.randn(self.no_species,self.no_species)
        
        # Generate species-specific carrying capacities (K)
        carrying_capacity = 1 + (1/12)*np.random.randn(self.no_species,)
        # Construct n x n matrix of K_i x K_j, 
        #   where i is a row index (species index) and j is a column index (species index)
        K_matrix = np.dot((1/carrying_capacity).reshape(len(carrying_capacity),1),
                          carrying_capacity.reshape(1,len(carrying_capacity)))
        
        # Element-wise multiplication of 
        interact_mat_norm = interact_mat * K_matrix
        
        # set a_ij = -1 for i = j/self-interaction to prevent divergence
    
        np.fill_diagonal(interact_mat_norm, 1) 
        
        return interact_mat_norm
    
    ######################
    
    def generate_community_function(self,usersupplied_community_function=None):
        
        '''
        
        Generate or assign community function.
        
        Parameters
        ----------
        usersupplied_community_function : None or np.array, size (no_species,).
            User-supplied array of species contribution to community function, default None.
            
        Returns
        -------
        None
        '''
    
        if usersupplied_community_function is None:
            
            self.species_contribute_community_function = self.species_contribution_to_community_function()
        
        else: 
            
            self.species_contribute_community_function = usersupplied_community_function
        
       
    def species_contribution_to_community_function(self,
                                                   mu_contribution=0,sigma_contribution=1):
        
        '''
        
        Generate parameters for species contribution to community function, or species function.
            Inspired by Chang et al. (2021), "Engineering complex communities by directed evolution".
            All species had a fixed species function, rather than community function
            being emergent from dynamic mechanistic interactions.
            Species contribution to community function is drawn from 
            normal(mu_contribution,sigma_contribution)
            
        Parameters
        ----------
        no_species : int
            Number of species.
        mean_contribution : float
            Mean species function.
        function_std : float
            Standard deviation for species function.
        
        Returns
        -------
        species_function : np.array of floats, size (no_species,)
            Array of individual species functions, drawn from distribution normal(0,function_std).
        
        '''
        
        species_function = mu_contribution + sigma_contribution*np.random.randn(self.no_species)
        
        return species_function

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
        
        # Assign growth rates and interaction matrix as gLV class attributes.
        #   (This isn't necessary, but I like it.)
        self.no_species = community_parameters_object.no_species
        
        self.growth_rates = community_parameters_object.growth_rates
        self.interaction_matrix = community_parameters_object.interaction_matrix
        self.dispersal = community_parameters_object.dispersal
        
        if usersupplied_init_cond is None:
        
            # Select function used to generate initial species abundances.
            init_cond_func_info = {"Hu":{"func":self.initial_abundances_hu,
                                    "args":[community_parameters_object.no_species,
                                            community_parameters_object.mu_a]},
                              "Mallmin":{"func":self.initial_abundances_mallmin,
                                         "args":[community_parameters_object.no_species
                                                 ,self.dispersal]}}[init_cond_func_name]
            
            # Generate initial conditions
            self.initial_abundances = init_cond_func_info['func'](*init_cond_func_info['args'])
              
        else: 
            
            # Assign initial conditions using the user-supplied initial abundances, if supplied.
            self.initial_abundances = usersupplied_init_cond
        
        self.ODE_sol = self.gLV_simulation(t_end)
      
    ########## Functions for generating initial conditions ############
      
    def initial_abundances_mallmin(self,no_species,dispersal):
        
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
        
        return np.random.uniform(dispersal,2/no_species,no_species)

    def initial_abundances_hu(self,no_species,mu_a):
        
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
        
        return np.random.uniform(0,2*mu_a,no_species)

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
        
        return solve_ivp(gLV_ode,[0,t_end],self.initial_abundances,
                         args=(self.growth_rates,self.interaction_matrix,self.dispersal),
                         method='RK45',t_eval=np.linspace(0,t_end,200))
    
    ########### Community properties #############
    
    def identify_community_properties(self,t_end):
        
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
        
        t_end_minus_last20percent = 0.8*t_end
        t_end_minus_last30percent = 0.7*t_end
        
        ###### Calculate diversity-related properties ###########
        
        final_popdyn = self.species_diversity([t_end_minus_last20percent,t_end])
        
        self.final_diversity = final_popdyn[1]
        self.final_composition = np.concatenate((np.where(final_popdyn[0] == True)[0],
                                                 np.zeros(self.ODE_sol.y.shape[0]-self.final_diversity)))
        
        ########## Determine if the community is fluctuating ###############
       
        self.fluctuations = self.detect_fluctuations_timeframe(5e-2,[t_end_minus_last20percent,t_end])
        
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
        
        t_start_index = np.where(self.ODE_sol.t >= t_start)[0]
        
        baseline_abundance = self.dispersal * 1e2
        
        extant_species = np.any(self.ODE_sol.y[:,t_start_index] > extinct_thresh,axis = 1).nonzero()[0]

        fluctuating_species = extant_species[np.logical_not(np.isnan([find_normalised_peaks(self.ODE_sol.y[spec,t_start_index])[0] \
                                for spec in extant_species]))]
        
        if fluctuating_species.size > 0:
            
            when_fluctuating_species_are_lost = np.argwhere(self.ODE_sol.y[fluctuating_species,t_start_index[0]:] \
                                                            < baseline_abundance)
                
            if when_fluctuating_species_are_lost.size > 0:
            
                unique_species, index = np.unique(when_fluctuating_species_are_lost[:,0],
                                                  return_index=True)
                
                when_fluctuating_species_are_lost = when_fluctuating_species_are_lost[index,:]
                
                reinvading_species = np.array([np.any(self.ODE_sol.y[\
                                            fluctuating_species[when_fluctuating_species_are_lost[i,0]],
                                             (t_start_index[0] + when_fluctuating_species_are_lost[i,1]):] \
                                                      > baseline_abundance) for i in \
                                               range(when_fluctuating_species_are_lost.shape[0])])
                    
                proportion_fluctuating_reinvading_species = np.sum(reinvading_species)/len(extant_species)
                
            else:
                
                proportion_fluctuating_reinvading_species = 0 

        else:
            
            proportion_fluctuating_reinvading_species = 0 
            
        return proportion_fluctuating_reinvading_species
        
    def detect_fluctuations_timeframe(self,fluctuation_thresh,timeframe,extinct_thresh=1e-4):
        
        '''
        
        Detect whether a community exhibts fluctuating dynamics within a given timeframe.

        Parameters
        ----------
        fluctuation_thresh : float
            The threshold for the average coefficient of variation, 
                which determines whether or not a community is fluctuating.
        timeframe : list of ints, length 2.
            The start and end time to detect fluctuations in.
        simulations : OdeResult object of scipy.integrate.solve_ivp module
            (Deterministic) Solution to gLV ODE system.
        extinct_thresh : float
            Species extinction threshold.

        Returns
        -------
        fluctuations_truefalse : TYPE
            DESCRIPTION.

        '''
        
        # deepy copy object attribute so the class object doesn't get mutated
        simulations_copy = deepcopy(self.ODE_sol)
        simulations_copy.y = simulations_copy.y[np.where(simulations_copy.y[:,-1] > extinct_thresh)[0],:]
        
        # find the indices of the nearest time to the times supplied in timeframe
        indices = find_nearest_in_timeframe(timeframe,simulations_copy.t)    
        
        # detect if a community exhibits fluctuating species dynamics
        proportion_of_species_fluctuating = \
            self.detect_fluctuations(fluctuation_thresh,indices,simulations_copy.y)
        
        return proportion_of_species_fluctuating

    def detect_fluctuations(self,fluctuation_thresh,indices,pop_dyn):
        
        '''
        
        Detect whether a community is fluctuating or not.
        Based on the function from Hu et al. (2022).
        
        The average coefficient of variation for each species is calculated 
            (standard deviation in species dynamics weighted by average species 
             abundance), then the proportion of the community exhibiting fluctuating dynamics is calculated. 

        Parameters
        ----------
        fluctuation_thresh : float
            The threshold for the average coefficient of variation, 
                which determines whether or not a community is fluctuating.
        indices : int
            Index of nearest times to the timeframe supplied.
        pop_dyn : .y from OdeResult object of scipy.integrate.solve_ivp module
            Population dynamics ONLY from (deterministic) solution to gLV ODE system.
        
        Returns
        -------
        Boolean
            True = the community is fluctuating, False = the community is at a fixed point.

        '''
        
        # calculate the standard deviation in species population dynamics, 
        #   normalised by average species abundance, for all species.
        average_variation_coeff_per_spec = np.std(pop_dyn[:,indices[0]:indices[1]],axis=1)/np.mean(pop_dyn[:,indices[0]:indices[1]],axis=1)
        
        # find the species with the average CV greater than the flucutation threshold
        species_fluctutating = np.where(average_variation_coeff_per_spec > fluctuation_thresh)[0]
        
        # find the proportion of species with fluctuating dynamics in the whole community.
        fluct_prop = len(species_fluctutating)/pop_dyn.shape[0]
        
        return fluct_prop
    
    ############# Community function ################
        
    def call_community_function(self,comm_parms_attr_contribution_comm_func):
        '''
        
        Estimate community function, if required.

        Parameters
        ----------
        comm_parms_attr_contribution_comm_func : np.array, size (no_species,)
                                                species_contribute_community_function attribute from
                                                object of community_parameters class.
            Species contribution to community function.

        Returns
        -------
        None.

        '''
        
        self.species_contribute_community_function = \
            comm_parms_attr_contribution_comm_func
        
        self.community_function = self.community_function_totalled_over_maturation()
      
    def community_function_totalled_over_maturation(self):
        
        '''
        
        Parameters
        ----------
        species_function : np.array of floats, size (no_species,)
            Species contribution to community function.
        species_abundances_over_time : .y attribute from OdeResult object of scipy.integrate.solve_ivp module
            Species abundances over time.

        Returns
        -------
        community_function : TYPE
            DESCRIPTION.

        '''
        
        summed_abundances = np.sum(self.ODE_sol.y,axis=1)
        
        community_function = np.sum(np.multiply(self.species_contribute_community_function,
                                                summed_abundances))
        
        return community_function

################

class community(community_parameters):
    
   '''
   Generate a species pool (aka generate model parameters), then simulate community
       dynamics using the generalised Lotka-Volterra model from multiple initial conditions.
   Each initial condition is called a 'lineage'.
        
   This class inherits from the community_parameters class to generate model parameters,
       then class the gLV class to run simulations.
        
   This class also calculates community properties, such as species diversity,
       % of the community with fluctuating dynamics, lyapunov exponents, and the number
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
       
       # % of species with fluctuating dynamics per lineage
       self.fluctuations = {}
       
       # Species diversity for each lineage at the end of simulations
       self.diversity = {}
       
       # % species that fluctuate and can reinvade the community
       self.invasibilities = {}
       
   ########################
      
   def simulate_community(self,
                          t_end,
                          func_name,lineages,
                          init_cond_func_name=None,array_of_init_conds=None,
                          with_community_function=False):
       
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
       with_community_function : Boolean, optional
           Choose to calculate community function alongside other community properties.
               The default is False.

       Returns
       -------
       None.

       '''
       
       # Choose how to generate initial species abundances for each lineage
       repeat_simulations_info = {"Default":{"func":self.repeat_simulations,
                                             "args":[lineages,t_end,
                                                     init_cond_func_name,with_community_function]},
                                  "Supply initial conditions":{"func":self.repeat_simulations_supply_initcond,
                                                               "args":[lineages,t_end,
                                                               array_of_init_conds,with_community_function]}}[func_name]
       
       # Call function to generate initial species abundances
       repeat_simulations_info["func"](*repeat_simulations_info["args"])
            
   def repeat_simulations(self,lineages,t_end,init_cond_func_name,with_community_function):
       
       '''
       
       Simulate community dynamics/generalised Lotka-Volterra model for each lineage.
       Initial conditions are generated using a function.

       Parameters
       ----------
       lineages : np.array of ints
           Index/label for lineages generated from the species pool. 
       t_end : float
           End of simulation.
       init_cond_func_name : string
           Name of function used to generate initial conditions. The default is None.
       with_community_function : Boolean, optional
           Choose to calculate community function alongside other community properties.
               The default is False.

       Returns
       -------
       None.

       '''
       
       # If community function should be calculated
       if with_community_function:
           
           # Creat attribute to store each lineage's community function
           self.community_functions = {}
           
           for lineage in lineages:
               
               # Call gLV class to simulate community dynamics
               gLV_res = gLV(self,t_end,init_cond_func_name)
               
               # Calculate community properties, assign to class attributes
               gLV_res.identify_community_properties(t_end)
               self.assign_gLV_attributes(gLV_res, lineage)
               
               # Calculate community community function, assign to class attributes
               gLV_res.call_community_function(self.species_contribute_community_function)
               self.assign_community_function(gLV_res, lineage)
       
       # If community function shouldn't be calculated
       else:
           
           for lineage in lineages:
               
               # Call gLV class to simulate community dynamics
               gLV_res = gLV(self,t_end,init_cond_func_name)
               
               # Calculate community properties, assign to class attributes
               gLV_res.identify_community_properties(t_end)
               self.assign_gLV_attributes(gLV_res, lineage)
       
       # Calculate the number of unique species compositions for the species pool
       no_uniq_compositions, comps = self.unique_compositions()
       
       self.no_unique_compositions = no_uniq_compositions
       self.unique_composition_label = {'lineage '+ str(lineage) : comp for lineage, comp in zip(lineages, comps)}
       
   def repeat_simulations_supply_initcond(self,lineages,t_end,array_of_init_conds,
                                          with_community_function):
       
       '''
       
       Simulate community dynamics/generalised Lotka-Volterra model for each lineage.
       Initial conditions are supplied by the user.

       Parameters
       ----------
       lineages : np.array of ints
           Index/label for lineages generated from the species pool. 
       t_end : float
           End of simulation.
       array_of_init_conds : list of np.array of floats, optional
           Arrays of initial species abundances.
       with_community_function : Boolean, optional
           Choose to calculate community function alongside other community properties.
               The default is False.

       Returns
       -------
       None.

       '''
       
       # If community function should be calculated
       if with_community_function:
           
           self.community_functions = {}
      
           for count, lineage in enumerate(lineages):
               
               # Call gLV class to simulate community dynamics
               gLV_res = gLV(self,t_end,usersupplied_init_cond=array_of_init_conds[:,count])
             
               # Calculate community properties, assign to class attributes
               gLV_res.identify_community_properties(t_end)
               self.assign_gLV_attributes(gLV_res, lineage)
             
               # Calculate community community function, assign to class attributes
               gLV_res.call_community_function(self.species_contribute_community_function)
               self.assign_community_function(gLV_res, lineage)
        
       else:
           
           for count, lineage in enumerate(lineages):
           
                # Call gLV class to simulate community dynamics
                gLV_res = gLV(self,t_end,usersupplied_init_cond=array_of_init_conds[:,count])
                
                # Calculate community properties, assign to class attributes
                gLV_res.identify_community_properties(t_end)
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
           DESCRIPTION.
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
       
       # Assign % species with flucutating dynamics
       self.fluctuations[dict_key] = gLV_res.fluctuations
       
       # Assign community invasibility from species pool
       self.invasibilities[dict_key] = gLV_res.invasibility
      
   def assign_community_function(self,gLV_res,lineage):
       
       '''
       
       Assign community function to class attributes
       
       Parameters
       ----------
       gLV_res : object of class gLV
           DESCRIPTION.
       lineage : int
           Lineage index/label.

       Returns
       -------
       None.
       
       '''
       
       dict_key = "lineage " + str(lineage)
        
       self.community_functions[dict_key] = gLV_res.community_function
       
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
    
   def calculate_lyapunov_exponents(self,
                                    lineages,
                                    n=10,dt=7000,separation=1e-2,extinct_thresh=1e-4):
    '''
       
    Calculate the average lyapunov exponent for all lineages.
    This enables us to estimate ecological dynamics.
    
    Parameters
    ----------
     lineages : np.array of ints
         Index/label for lineages generated from the species pool. 
     n : int, optional
         The number of iterations the lyapunov exponent is calculated over. The default is 10.
     dt : float, optional
         The timestep the lyapunov exponents is calculated over. The default is 7000.
     separation : float, optional
         The amount a community is perturbated. The default is 1e-2.
     extinct_thresh : float, optional
         Species extinction threshold. The default is 1e-4.

    Returns
    -------
    None.

    '''
       
    self.lyapunov_exponents = {}
    
    # For each lineage
    for lineage in lineages:
        
        dict_key = "lineage " + str(lineage)
        
        # Call function to calculate lyapunov exponents, assign result to class attribute
        self.lyapunov_exponents[dict_key] = self.gLV_lyapunov_exponent(dict_key,
                                                                       n,dt,separation,
                                                                       extinct_thresh)
  
   def gLV_lyapunov_exponent(self,dict_key,n,dt,separation,extinct_thresh):
       
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
       
       # Initialise list of max. lyapunov exponents
       log_d1d0_list = []
       
       # Set initial conditions as population abundances at the end of lineage simulations
       initial_conditions = self.ODE_sols[dict_key].y[:,-1]
       
       # Set initial conditions of the original trajectory
       initial_conditions_no_sep = deepcopy(initial_conditions)
       
       # Set initial conditions of the perturbated communty as the same as the original trajectory
       initial_conditions_sep = deepcopy(initial_conditions)
       # Select an extant species
       species_to_perturbate = np.where(initial_conditions_sep > extinct_thresh)[0][0]
       # Perturbate the selected species by 'separation'. We now have a perturbated trajectory.
       initial_conditions_sep[species_to_perturbate] += separation
        
       # Repeat process n times
       for step in range(n):
            
           # Simulate the original community trajectory for time = dt
           gLV_res = gLV(self,dt,usersupplied_init_cond=initial_conditions_no_sep)
           # Simulate the perturbated community trajectory for time = dt
           gLV_res_separation = gLV(self,dt,usersupplied_init_cond=initial_conditions_sep)
           
           # Get species abundances at the end of simulation from the original trajectory
           species_abundances_end = gLV_res.ODE_sol.y[:,-1]
           # Get species abundances at the end of simulation from the perturbated trajectory
           species_abundances_end_sep = gLV_res_separation.ODE_sol.y[:,-1]
           
           # Calculated the new separation between the original and perturbated trajectory (d1)
           separation_dt = np.sqrt(np.sum((species_abundances_end - species_abundances_end_sep)**2))
           
           # Calculate the max. lyapunov exponent
           log_d1d0 = (1/dt)*np.log(np.abs(separation_dt/separation))
           # Add exponent to list
           log_d1d0_list.append(log_d1d0)
           
           # Reset the original trajectory's species abundances to the species abundances at dt.
           initial_conditions_no_sep = species_abundances_end 
           # Reset the perturbated trajectory's species abundances so that the original
           #    and perturbated community are 'separation' apart.
           initial_conditions_sep = species_abundances_end + \
               (separation/separation_dt)*(species_abundances_end_sep-species_abundances_end)
       
       # Calculate average max. lyapunov exponent
       max_lyapunov_exponent = mean_std_deviation(np.array(log_d1d0_list))
       
       return max_lyapunov_exponent
                  
####################### Functions #####################

############################ gLV simulations ##################

def gLV_ode(t,spec,growth_r,interact_mat,dispersal):
    
    '''
    
    ODE system from generalised Lotka-Volterra model

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
    dispersal : float.
        Dispersal or migration rate.

    Returns
    -------
    dSdt : np.array of float64, size (n,)
        array of change in population dynamics at time t aka dS/dt.

    '''

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

def find_period_ode_system(ode_object,t_start,extinct_thresh=1e-4,dt=10):
    
    '''
    
    Find the periodicity of all species population dynamics in a community.
    
    This function was created to idenitfy the max. period (max_period) of 'chaotic-like
    communities'. In gLV_lyapunov_exponent(), dt = max_period to identifying whether
    the chaotic-like communities were actually chaotic (max_lyapunov_exponent > 0),
    or oscillating (max_lyapunov_exponent = 0).
    
    Parameters
    ----------
    ode_object : gLV.ode_sol attribute or community.ODE_sols['lineage x']. scipy.integrate object.
        ODE simulations.
    t_start : float
        The starting time to check for periodicity. Function looks for periodicity
        in ode_object.t[np.where(ode_object.t >= t_start)[0]:]
    extinct_thresh : float, optional
        Extinction threshold. The default is 1e-4.
    dt : float, optional
        Time step, used to discretise ODE solver times. The default is 10.

    Returns
    -------
    periods : list
        List of max. periods of species dynamics. If a species' dynamics are not 
        periodic, then np.nan is returned.

    '''
    # Discretise time steps from ODE solver for desired time range. 
    #   This is necessary for the fourier transform
    t_discretised = np.arange(t_start,ode_object.t[-1],dt)
    
    # Identify extant/non-extinct species in desired time range.
    extant_species = np.any(ode_object.y[:,np.where(ode_object.t >= t_start)[0]] > extinct_thresh,
                                          axis = 1).nonzero()[0]
    
    def find_max_period_species(ode_object,t_start,t_discretised,dt,species):
        
        '''
        
        Find the periodicity of a single species population dynamics.
        
        This is achieved by fourier transforming the data to find the frequency 
        of population dynamics fluctuations, if the species has fluctuating and
        periodic dynamics. Then the function identifies peaks in the fourier spectrum,
        which correspond to frequencies in population dynamics oscillations.
        The peak corresponding to the max. period/min. frequency is identified and
        the max. period is returned.

        Parameters
        ----------
        ode_object : gLV.ode_sol attribute or community.ODE_sols['lineage x']. scipy.integrate object.
            ODE simulations.
        t_start : float
            The starting time to check for periodicity. Function looks for periodicity
            in ode_object.t[np.where(ode_object.t >= t_start)[0]:]
        t_discretised : np.array of floats
            Discretised times from t_start to end of simulations (ode_object.t[-1]).
            Time step = dt.
        dt : float
            Time step, used to discretise ODE solver times. The default is 10.
        species : int
            Index of species in ode_object.y.

        Returns
        -------
        max_period : float or np.nan
            The maximum periodicity of the species population dynamics, extracted
            from the fourier transform. If the system is not periodic, returns np.nan.

        '''
        
        # Interpolate species population dynamics, so that we have population dynamics
        #   at t_start to end of simulations with fixed time step dt.
        # Having population dynamics at regular time steps is necessary for the 
        #   fourier transform.
        interpolated_spec = np.interp(t_discretised,ode_object.t[np.where(ode_object.t >= t_start)[0]],
                                      ode_object.y[species,np.where(ode_object.t > t_start)[0]])
        
        # Generate the fourier spectrum of the real value species population dynamics
        #   (No complex values are included)
        fourier_spec = rfft(interpolated_spec)
        # Remove negative frequencies
        normalised_fourier_spec = 2*np.abs(fourier_spec)/len(interpolated_spec)
        
        # Calculate the periods of species population dynamics.
        period = 1/rfftfreq(len(interpolated_spec), d=dt)
        
        # Identify the peaks in the fourier spectrum, which correspond to frequencies
        #   of oscillations.
        # Remove first value in normalised_fourier_spectrum (this is standard).
        peak_ind = find_normalised_peaks(normalised_fourier_spec[1:]) + 1
        
        # If there are peaks in the fourier spectrum, so population dynamics are periodic
        if peak_ind.size > 0:
    
            try: # Extract the max. period population dynamics.
                
                max_period = period[peak_ind[0]]
                
            except IndexError: # If there is only one peak_ind value, which does
            #   not correspond to a real peak. 
                
                max_period = np.nan # no peaks
        
        # If there are no peaks in the fourier spectrum
        else: 
            
            max_period = np.nan # no peaks
        
        return max_period
    
    # Calculate the max. period of all extant species in the community.
    periods = [find_max_period_species(ode_object,t_start,t_discretised,dt,species) for species in extant_species]
    
    return periods

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
    
def community_object_to_df(community_object,community_label=0,interaction_strength_mean=True,
                           interaction_strength_std=True,no_species=True,no_unique_compositions=True,
                           diversity=True,fluctuations=True,invasibility=True,le=False):
    
    no_lineages = len(community_object.ODE_sols)
    
    community_col = np.repeat(community_label,no_lineages)
    
    lineage_col = list(community_object.ODE_sols.keys())
    lineage_col = [int(lineage.replace('lineage ','')) for lineage in lineage_col]
    
    ############## Repeated attributes ####################
    
    potential_rep_attributes = ['mu_a','sigma_a','no_species','no_unique_compositions']
    rep_attributes_to_get = [interaction_strength_mean,interaction_strength_std,
                             no_species,no_unique_compositions]
    rep_attribute_names = list(itertools.compress(potential_rep_attributes,rep_attributes_to_get))
    
    rep_attr_cols = [np.repeat(getattr(community_object,attribute_name),no_lineages) \
                    for attribute_name in rep_attribute_names]
    
    ############### Non-repeated attributes ###################
    
    no_rep_potential_attributes = ['unique_composition_label','diversity','fluctuations','invasibilities']
    no_rep_attributes_to_get = [no_unique_compositions,diversity,fluctuations,
                         invasibility]
    no_rep_attribute_names = list(itertools.compress(no_rep_potential_attributes,no_rep_attributes_to_get))
    
    no_rep_attr_cols = [list(getattr(community_object, attribute_name).values()) \
                        for attribute_name in no_rep_attribute_names]
    
    ################################# Combine attributes ############
    
    attribute_cols = [community_col] + [lineage_col] + rep_attr_cols + no_rep_attr_cols
    
    col_names = ['community','lineage'] + rep_attribute_names + no_rep_attribute_names
    
    ############## Lyapunov exponents #################
    
    if le is True:
        
        le_mean_col = [lyapunov_exponent[0] \
                       for lyapunov_exponent in community_object.lyapunov_exponents.values()]
        le_std_col = [lyapunov_exponent[1] \
                       for lyapunov_exponent in community_object.lyapunov_exponents.values()]
            
        attribute_cols += [le_mean_col,le_std_col]
        col_names += ['lyapunov_exponent_mean','lyapunov_exponent_std']
        
    ############# Convert lists to df ################
    
    community_df = pd.DataFrame(attribute_cols)
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




