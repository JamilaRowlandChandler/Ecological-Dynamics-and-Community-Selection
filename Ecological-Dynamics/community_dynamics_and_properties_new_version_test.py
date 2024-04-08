# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:35:29 2024

@author: Jamila
"""

######################

# Jamila: for console - cd "Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics"

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
from scipy import linalg

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
                 growth_func,growth_args,
                 interact_func,interact_args,
                 usersupplied_growth,usersupplied_interactmat,
                 dispersal):
        '''
        
        Generate or assign parameters used in a generalised Lotka-Volterra model.

        Parameters
        ----------
        no_species : int
            Number of species in species pool.
        growth_func : string
            Name of function used to generate growth rates.
                'fixed' - growth rates all equal 1,
                'normal' - growth rates are generated from normal(mu_g,sigma_g).
        growth_args : dict.
            Arguments for function used to generate growth rates, if required.
        interact_func : string
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
        match growth_func:
            
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
        
        match interact_func:
            
            case 'random':
                
                for key, value in interact_args.items():
                    
                    # Assign interaction matrix function arguments as class attributes.
                    setattr(self,key,value)
                    
                # Generate interaction matrix 
                self.interaction_matrix = self.random_interaction_matrix()
                
            case 'sparse':
                
                for key, value in interact_args.items():
                    
                    # Assign interaction matrix function arguments as class attributes.
                    setattr(self,key,value)
                    
                # Generate interaction matrix 
                self.interaction_matrix = self.sparse_interaction_matrix()
                
            case 'modular':
                
                for key, value in interact_args.items():
                    
                    # Assign interaction matrix function arguments as class attributes.
                    setattr(self,key,value)
                    
                self.interaction_matrix = self.modular_interaction_matrix()
                
            case 'nested':
                
                for key, value in interact_args.items():
                    
                    # Assign interaction matrix function arguments as class attributes.
                    setattr(self,key,value)
                    
                # Generate interaction matrix 
                self.interaction_matrix = self.nested_interaction_matrix()
            
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
    
        
    ##################### Interaction Matrix #####################
    
    ########## Sparse and dense random interaction matrices #########
            
    def random_interaction_matrix(self):
        
        '''
        
        Generate a classic, dense, random interaction matrix.
    
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
    
    def sparse_interaction_matrix(self):
        
        '''
        
        Generate a sparse random interaction matrix using a Erdős–Rnyi graph.
        
        See May (1972) for details. https://doi.org/10.1038/238413a0
    
        Returns
        -------
        interact_mat : np.array of size (n,n)
            Interaction matrix. 
    
        '''
        
        interact_mat =\
            self.interaction_matrix_with_connectance(self.no_species, self.mu_a,
                                                     self.sigma_a, self.connectance)
        
        return interact_mat
    
    def mixed_sparse_interaction_matrix(self):
        
        # generate competitive interactions to start
        interact_mat = \
            self.interaction_matrix_with_connectance(self.no_species,
                                                     self.competitive_mu_a,
                                                     self.competitive_sigma_a,
                                                     self.competitive_connectance)
        
        cooperative_interaction_indices = \
            np.random.binomial(1,self.probability_cooperative,
                               size=self.no_species*self.no_species).reshape((self.no_species,self.no_species))
        
        cooperative_interaction_matrix = \
            self.interaction_matrix_with_connectance(cooperative_interaction_indices.shape[0],
                                                     self.cooperative_mu_a,
                                                     self.cooperative_sigma_a,
                                                     self.cooperative_connectance,
                                                     self_inhibition=False)
        
        interact_mat[np.where(interact_mat == cooperative_interaction_indices)] = \
            cooperative_interaction_matrix
            
        return interact_mat
    
    ############# Non-uniform structured interaction matrices ###########
    
    def modular_interaction_matrix(self):
        
        '''
        
        Generate a modular interaction matrix using a stochastic block model (SBM).
        
        See Akjouj et al. (2024) for details on how SBMs can be applied to gLVs.
        https://doi.org/10.1098/rspa.2023.0284

        Returns
        -------
        interact_mat : np.array of size (n,n)
            Interaction matrix. 

        '''
        
        ########### Cluster species into modules ##########
        
        if self.module_probabilites:
            
            clustered_species = np.random.multinomial(self.no_species,
                                                      self.module_probabilities,
                                                      size=1)[0]

        else:
        
            clustered_species = np.random.multinomial(self.no_species,
                                                      np.repeat(1/self.no_modules,self.no_modules),
                                                      size=1)[0]
        
        # create the interaction matrices for each module
        module_interactions = \
            [self.interaction_matrix_with_connectance(nodes,self.p_mu_a,self.p_sigma_a,
                                                 self.p_connectance) \
             for nodes in clustered_species]
        
        # combine module interactions into a community interaction matrix
        interact_mat = linalg.block_diag(*module_interactions)
        
        ####### Assign interactions between members of different groups ######
        
        # get indices of interaction matrix where species from different modules interact
        non_group_interaction_indices = np.where(interact_mat == 0)
        
        # generate the interactions between species from different modules
        non_group_interactions = \
            self.interaction_matrix_with_connectance(self.no_species,
                                                self.q_mu_a,self.q_sigma_a,
                                                self.q_connectance)    
        
        # add between-module interactions to the interaction matrix
        interact_mat[non_group_interaction_indices] = \
            non_group_interactions[non_group_interaction_indices]
        
        return interact_mat

    def nested_interaction_matrix(self,beta=7):
        
        '''
        
        Created a nested, or scale-free, interaction matrix using the Chung-Lu model.
        
        See Akjouj et al. (2024) for details on how the Chung-Lu model can be
        applied to gLVs. https://doi.org/10.1098/rspa.2023.0284

        Parameters
        ----------
        beta : float, optional
            Scale parameter used to describe the probabiltiy node n has k nodes.
            The default is 7.

        Returns
        -------
        interact_mat : np.array of size (n,n)
            Interaction matrix. 

        '''
        
        # create i's
        species = np.arange(1,self.no_species+1)
        
        # calculate node weights, used to calculate the probability species i interacts with j.
        weights = \
            self.average_degree*((beta-2)/(beta-1))*((self.no_species/species)**(1/(beta-1)))
        
        # calculate the probability species i interacts with j.
        probability_of_interactions = \
            (np.outer(weights,weights)/np.sum(weights)).flatten()
        
        # set probabilities > 1 to 1.
        probability_of_interactions[probability_of_interactions > 1] = 1
        
        interact_mat = \
            self.interaction_matrix_with_connectance(self.no_species,
                                                     self.mu_a, self.sigma_a,
                                                     probability_of_interactions)
        
        return interact_mat
    
    def modular_mixed_interaction_matrix():
        
        pass
    
    def nested_mixed_interaction_matrix():
        
        pass
    
    ########### Extra functions for generating interaction matrices #####
    
    def interaction_matrix_with_connectance(self,n,mu_a,sigma_a,connectance,
                                            self_inhibition=True):
        
        '''
        
        Generate a random interaction matric with connectance c.

        Parameters
        ----------
        n : int
            Number of n. 
            (The interaction matrix describes interaction/edges between n.)
        mu_a : float
            Average interaction strength.
        sigma_a : float
            Standard deviation in interaction strength.
        connectance : float
            Probability of node i and j interacting (c).

        Returns
        -------
        interaction_matrix : np.ndarray of size (n,n).
            Interaction matrix.

        '''
        # create the connectance matrix (whether n are interacting or not)
        are_species_interacting = \
            np.random.binomial(1,connectance,size=n*n).reshape((n,n))
        
        # create the interaction strength matrix
        interaction_strengths = mu_a + sigma_a*np.random.randn(n,n)
        
        # create the interaction matrix
        interaction_matrix = interaction_strengths * are_species_interacting
        
        if self_inhibition == True:
            
            # set a_ij = -1 for i = j/self-interaction to prevent divergence
            np.fill_diagonal(interaction_matrix, 1)
        
        return interaction_matrix
    
    ############################### Community Function #######################
    
    def generate_community_function(self,func_name='Generate community function',
                                    community_func_args={'mu_contribution':0,'sigma_contribution':1},
                                    usersupplied_community_function=None):
        
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
        
        match func_name:
            
            case 'Generate community function':
                
                self.species_contribution_community_function = \
                    self.species_contribution_to_community_function(**community_func_args)
                
            case None: 
                
                self.species_contribution_community_function = usersupplied_community_function
       
    def species_contribution_to_community_function(self,
                                                   mu_contribution,sigma_contribution):
        
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
                 init_cond_func=None,
                 usersupplied_init_cond=None):
        
        '''
        Assign class attributes, generate initial conditions, and run simulations.
        
        Parameters
        ----------
        community_parameters_object : object of class community_parameters.
            community parameters.
        t_end : float
            End of simulation.
        init_cond_func : string
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
        match init_cond_func:
            
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
        
        t_end_minus_last30percent = 0.7*self.t_end
        
        ###### Calculate diversity-related properties ###########
        
        final_popdyn = self.species_diversity([t_end_minus_last30percent,self.t_end])
        
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
            
            # get final index of simulation window
            end_index = len(t_start_index)-1
            
            # # find if and where species abundances dip below baseline_abundance.
            # Tuple entry 0 = species, Tuple entry 1 = index of the timepoint where their 
            #   abundances dipped below baseline_abundance.
            when_fluctuating_species_are_lost = np.nonzero(self.ODE_sol.y[fluctuating_species,t_start_index[0]:] \
                                                            < baseline_abundance)
            
            # If species abundances dip below baseline_abundance   
            if len(when_fluctuating_species_are_lost[0]) > 0:
            
                # Identify the species with abundances that dip below baseline_abundance
                #   and the first entry where the unique species was identified.
                unique_species, index = \
                    np.unique(when_fluctuating_species_are_lost[0],return_index=True)
                
                # get the final index of each species 
                final_index = np.append(index[1:],len(when_fluctuating_species_are_lost[1])) - 1
                
                # count number of reinvading species
                # if the final index is less than end_index, the species is reinvading/increasing above baseline_abundance.
                no_reinvading_species = np.count_nonzero(when_fluctuating_species_are_lost[1][final_index] < end_index)
                
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
    
    ############# Community function ################
    
    def call_community_function(self,community_parameters_object):
        
        self.species_contribution_community_function = \
            community_parameters_object.species_contribution_community_function
            
        self.community_function_totalled_over_maturation()
      
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
        
        self.community_function = np.sum(np.multiply(self.species_contribution_community_function,
                                                summed_abundances))
        
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
                growth_func, growth_args,
                interact_func, interact_args,
                usersupplied_growth=None,usersupplied_interactmat=None,
                dispersal=1e-8):
       '''
       
       Generate model parameters (by inheriting from community_parameters),
           initialise attributes that store community properties.
           
       Parameters
       ----------
       no_species : int
           Number of species in species pool.
       growth_func : string
           Name of function used to generate growth rates.
               'fixed' - growth rates all equal 1,
               'normal' - growth rates are generated from normal(mu_g,sigma_g).
       growth_args : dict.
           Arguments for function used to generate growth rates, if required.
       interact_func : string
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
                        growth_func, growth_args,
                        interact_func, interact_args,
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
       
       self.community_functions = {}
       
   ########################
      
   def simulate_community(self,
                          lineages,t_end,func_name,
                          init_cond_func=None,array_of_init_conds=None,
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
               'Default' : Use a function, supplied by init_cond_func, to
                   generate different initial species abundances for each lineage.
               'Supply initial conditions' : The user supplies initial species
                   abundances for each lineage.
       lineages : np.array of ints
           Index/label for lineages generated from the species pool. 
           Typically generated from np.arange or np.linspace.
       init_cond_func : string, optional
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

       ################ Generate initial species abundances and simulate lineage dynamics #############
       
       match func_name:
            
           case 'Generate initial conditions':
                
               for lineage in lineages:
                     
                    # Call gLV class to simulate community dynamics
                    gLV_res = gLV(self,t_end,init_cond_func)
                     
                    # Calculate community properties, assign to class attributes
                    gLV_res.identify_community_properties()
                    self.assign_gLV_attributes(gLV_res, lineage)
                    
                    if with_community_function:
                        
                        # Calculate community community function, assign to class attributes
                        gLV_res.call_community_function(self)
                        self.assign_community_function(gLV_res, lineage)
          
           case 'Supply initial conditions':
              
               for count, lineage in enumerate(lineages):
                   
                   # Call gLV class to simulate community dynamics
                   gLV_res = gLV(self,t_end,usersupplied_init_cond=array_of_init_conds[:,count])
                     
                   # Calculate community properties, assign to class attributes
                   gLV_res.identify_community_properties()
                   self.assign_gLV_attributes(gLV_res, lineage)
                   
                   if with_community_function:
                       
                       # Calculate community community function, assign to class attributes
                       gLV_res.call_community_function(self)
                       self.assign_community_function(gLV_res, lineage)
         
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
       
   def assign_community_function(self,gLV_res,lineage):
       '''
       
       Assign community function

       Parameters
       ----------
       community_function : float
           community function.
       lineage : int
           lineage label.

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

def gLV_ode_separate_mixed_interactions(t,spec,
                                        growth_r,competitive_mat,cooperative_mat,
                                        gamma,dispersal,extinct_thresh=1e-9):
    
    spec[spec < extinct_thresh] = 0 # set species abundances below extinction threshold to 0
    
    competition = np.matmul(competitive_mat,spec)
    cooperation = np.matmul(cooperative_mat,spec/(gamma+spec))
    
    dSdt = np.multiply(1 + cooperation - competition, growth_r*spec) + dispersal
    
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
