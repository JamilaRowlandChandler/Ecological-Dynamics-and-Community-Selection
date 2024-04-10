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
from scipy import linalg

from model_parameters import *
from generate_initial_conditions import *
from community_properties import *
from utility_functions import *

############################### Classes ########################

class gLV_with_allee_parameters:
    
    '''
    
    Create parameters for Lotka-Volterra models with allee effects. These models
    have separate competition and cooperation interaction matrices.
        
    The class has methods for generating growth rates, and competitive and
    cooperatinve interaction matrices.
    
    '''
    
    def __init__(self,
                 no_species,
                 growth_func,growth_args,
                 competition_func,competition_args,
                 cooperation_func,cooperation_args,
                 usersupplied_growth,usersupplied_competition,usersupplied_cooperation,
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
                
                self.growth_rates = growth_rates_fixed(self)
                
            case 'normal':
                
                for key, value in growth_args.items():
                    
                    # Assign growth function arguments as class attributes.
                    #   (Growth function arguments are parameters for 
                    #   the growth rate distribution,)
                    setattr(self,key,value)
                    
                self.growth_rates = growth_rates_norm(self)
                
            case None:
                
                # Assign growth rates using the user-supplied growth rates
                self.growth_rates = usersupplied_growth
            
        ###################### Interaction Matrix #############
        
        if competition_args:
            
           for key, value in competition_args.items():
               
               # Assign interaction matrix function arguments as class attributes.
               setattr(self,key,value) 
        
        match competition_func:
            
            case 'random':
                    
                # Generate interaction matrix 
                self.competition_matrix = random_interaction_matrix(self.mu_comp,self.sigma_a,
                                                                    self.no_species)
                
            case 'sparse':
            
                # Generate interaction matrix 
                self.competition_matrix = sparse_interaction_matrix(self.mu_comp,self.sigma_a,
                                                                    self.connectance,self.no_species)
                
            case 'modular':
                
                if hasattr(self, 'module_probabilities'):
                    
                    self.competition_matrix = modular_interaction_matrix(self.no_species,self.no_modules,
                                                                         self.p_mu_comp,self.p_sigma_a,self.p_connectance,
                                                                         self.q_mu_comp,self.q_sigma_a,self.q_connectance,
                                                                         self.module_probabilities)
                    
                else:
                    
                    self.competition_matrix = modular_interaction_matrix(self.no_species,self.no_modules,
                                                                         self.p_mu_comp,self.p_sigma_a,self.p_connectance,
                                                                         self.q_mu_comp,self.q_sigma_a,self.q_connectance,
                                                                         np.repeat(1/self.no_modules,self.no_modules))
                
            case 'nested':

                # Generate interaction matrix 
                self.competition_matrix = nested_interaction_matrix(self.mu_comp,self.sigma_a,
                                                                    self.average_degree,self.no_species)
            
            case None:
                
                # Assign interaction matrix using the user-supplied interaction matrix, if supplied.
                self.competition_matrix = usersupplied_competition
                
        #####
        
        if cooperation_args:
            
           for key, value in cooperation_args.items():
               
               # Assign interaction matrix function arguments as class attributes.
               setattr(self,key,value) 
        
        match cooperation_func:
            
            case 'random':
                    
                # Generate interaction matrix 
                self.cooperation_matrix = random_interaction_matrix(self.mu_coop,self.sigma_a,
                                                                    self.no_species)
                
            case 'sparse':
            
                # Generate interaction matrix 
                self.cooperation_matrix = sparse_interaction_matrix(self.mu_coop,self.sigma_a,
                                                                    self.connectance,self.no_species)
                
            case 'modular':
                
                if hasattr(self, 'module_probabilities'):
                    
                    self.cooperation_matrix = modular_interaction_matrix(self.no_species,self.no_modules,
                                                                         self.p_mu_coop,self.p_sigma_a,self.p_connectance,
                                                                         self.q_mu_coop,self.q_sigma_a,self.q_connectance,
                                                                         self.module_probabilities)
                    
                else:
                    
                    self.cooperation_matrix = modular_interaction_matrix(self.no_species,self.no_modules,
                                                                         self.p_mu_coop,self.p_sigma_a,self.p_connectance,
                                                                         self.q_mu_coop,self.q_sigma_a,self.q_connectance,
                                                                         np.repeat(1/self.no_modules,self.no_modules))
                
            case 'nested':

                # Generate interaction matrix 
                self.cooperation_matrix = nested_interaction_matrix(self.mu_coop,self.sigma_a,
                                                                    self.average_degree,self.no_species)
            
            case None:
                
                # Assign interaction matrix using the user-supplied interaction matrix, if supplied.
                self.cooperation_matrix = usersupplied_cooperation
               
         ################# Dispersal #############  
          
        self.dispersal = dispersal
        
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
                    species_contribution_to_community_function(self.no_species,
                                                               **community_func_args)
                
            case None: 
                
                self.species_contribution_community_function = usersupplied_community_function
            
###################################################

class gLV_parameters:
    
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
                
                self.growth_rates = growth_rates_fixed(self)
                
            case 'normal':
                
                for key, value in growth_args.items():
                    
                    # Assign growth function arguments as class attributes.
                    #   (Growth function arguments are parameters for 
                    #   the growth rate distribution,)
                    setattr(self,key,value)
                    
                self.growth_rates = growth_rates_norm(self)
                
            case None:
                
                # Assign growth rates using the user-supplied growth rates
                self.growth_rates = usersupplied_growth
            
        ###################### Interaction Matrix #############
        
        if interact_args:
            
           for key, value in interact_args.items():
               
               # Assign interaction matrix function arguments as class attributes.
               setattr(self,key,value) 
        
        match interact_func:
            
            case 'random':
                    
                # Generate interaction matrix 
                self.interaction_matrix = random_interaction_matrix(self.mu_a,self.sigma_a,
                                                                    self.no_species)
                
            case 'sparse':
            
                # Generate interaction matrix 
                self.interaction_matrix = sparse_interaction_matrix(self.mu_a,self.sigma_a,
                                                                    self.connectance,self.no_species)
                
            case 'modular':
                
                if hasattr(self, 'module_probabilities'):
                    
                    self.interaction_matrix = modular_interaction_matrix(self.no_species,self.no_modules,
                                                                         self.p_mu_a,self.p_sigma_a,self.p_connectance,
                                                                         self.q_mu_a,self.q_sigma_a,self.q_connectance,
                                                                         self.module_probabilities)
                    
                else:
                    
                    self.interaction_matrix = modular_interaction_matrix(self.no_species,self.no_modules,
                                                                         self.p_mu_a,self.p_sigma_a,self.p_connectance,
                                                                         self.q_mu_a,self.q_sigma_a,self.q_connectance,
                                                                         np.repeat(1/self.no_modules,self.no_modules))
                
            case 'nested':
        
                # Generate interaction matrix 
                self.interaction_matrix = nested_interaction_matrix(self.mu_a,self.sigma_a,
                                                                    self.average_degree,self.no_species)
                    
            case None:
                        
                # Assign interaction matrix using the user-supplied interaction matrix, if supplied.
                self.interaction_matrix = usersupplied_interactmat    
            
        self.dispersal = dispersal
        

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
                    species_contribution_to_community_function(self.no_species,
                                                               **community_func_args)
                
            case None: 
                
                self.species_contribution_community_function = usersupplied_community_function
       
################

################

class gLV_with_allee:
    
    '''
    
    Run gLV simulations from initial conditions.
        Takes community_parameters class object as an argument, which contains model parameters.
        Has class methods for generating initial species abundances and running gLV ODE simulations.
    
    '''
    
    def __init__(self,
                 gLV_with_allee_parameters_object,
                 t_end,
                 init_cond_func=None,
                 usersupplied_init_cond=None):
        
        '''
        Assign class attributes, generate initial conditions, and run simulations.
        
        Parameters
        ----------
        gLV_parameters_object : object of class community_parameters.
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
        
        # Assign attributes from gLV_with_allee_parameters_object to gLV
        self.no_species = gLV_with_allee_parameters_object.no_species
        
        self.growth_rates = gLV_with_allee_parameters_object.growth_rates
        self.competition_matrix = gLV_with_allee_parameters_object.competition_matrix
        self.cooperation_matrix = gLV_with_allee_parameters_object.cooperation_matrix
        self.dispersal = gLV_with_allee_parameters_object.dispersal
        
        self.t_end = t_end
        
        ######## Generate initial species abundances ###########
        
        self.initial_abundances = \
            initial_conditions(self,init_cond_func,usersupplied_init_cond)
        
        self.ODE_sol = self.gLV_with_allee_simulation()
      
    ####### Simulate dynamics ###############
    
    def gLV_with_allee_simulation(self):
        
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
        
        return solve_ivp(gLV_ode_separate_mixed_interactions,[0,self.t_end],self.initial_abundances,
                         args=(self.growth_rates,self.competition_matrix,self.cooperation_matrix,
                               1,self.dispersal),
                         method='RK45',t_eval=np.linspace(0,self.t_end,200))
    
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
        
        final_popdyn = species_diversity(self,[t_end_minus_last30percent,self.t_end])
        
        self.final_diversity = final_popdyn[1]
        self.final_composition = np.concatenate((np.where(final_popdyn[0] == True)[0],
                                                 np.zeros(self.ODE_sol.y.shape[0]-self.final_diversity)))
        
        ########## Determine if the community is fluctuating ###############
       
        self.invasibility = detect_invasibility(self,t_end_minus_last30percent)
        
    
    ############# Community function ################
    
    def call_community_function(self,gLV_parameters_object):
        
        self.species_contribution_community_function = \
            gLV_parameters_object.species_contribution_community_function
            
        self.community_function = community_function_totalled_over_maturation(self)
    
################

class gLV:
    
    '''
    
    Run gLV simulations from initial conditions.
        Takes community_parameters class object as an argument, which contains model parameters.
        Has class methods for generating initial species abundances and running gLV ODE simulations.
    
    '''
    
    def __init__(self,
                 gLV_parameters_object,
                 t_end,
                 init_cond_func=None,
                 usersupplied_init_cond=None):
        
        '''
        Assign class attributes, generate initial conditions, and run simulations.
        
        Parameters
        ----------
        gLV_parameters_object : object of class community_parameters.
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
        
        # Assign attributes from gLV_parameters_object to gLV
        self.no_species = gLV_parameters_object.no_species
        
        self.growth_rates = gLV_parameters_object.growth_rates
        self.interaction_matrix = gLV_parameters_object.interaction_matrix
        self.dispersal = gLV_parameters_object.dispersal
        
        self.t_end = t_end
        
        ######## Generate initial species abundances ###########
        
        self.initial_abundances = \
            initial_conditions(self,init_cond_func,usersupplied_init_cond)
        
        self.ODE_sol = self.gLV_simulation(t_end)
      
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
        
        final_popdyn = species_diversity(self,[t_end_minus_last30percent,self.t_end])
        
        self.final_diversity = final_popdyn[1]
        self.final_composition = np.concatenate((np.where(final_popdyn[0] == True)[0],
                                                 np.zeros(self.ODE_sol.y.shape[0]-self.final_diversity)))
        
        ########## Determine if the community is fluctuating ###############
       
        self.invasibility = detect_invasibility(self,t_end_minus_last30percent)
        
    
    ############# Community function ################
    
    def call_community_function(self,gLV_parameters_object):
        
        self.species_contribution_community_function = \
            gLV_parameters_object.species_contribution_community_function
            
        self.community_function = community_function_totalled_over_maturation(self)
        
################

class gLV_community(gLV_parameters):
    
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
       no_uniq_compositions, comps = unique_compositions(self)
       
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

####################

class gLV_with_allee_community(gLV_with_allee_parameters):
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
                growth_func,growth_args,
                competition_func,competition_args,
                cooperation_func,cooperation_args,
                usersupplied_growth=None,usersupplied_competition=None,
                usersupplied_cooperation=None,dispersal=1e-8):
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
                        growth_func,growth_args,
                        competition_func,competition_args,
                        cooperation_func,cooperation_args,
                        usersupplied_growth,usersupplied_competition,usersupplied_cooperation,
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
                    gLV_allee_res = gLV_with_allee(self,t_end,init_cond_func)
                     
                    # Calculate community properties, assign to class attributes
                    gLV_allee_res.identify_community_properties()
                    self.assign_gLV_allee_attributes(gLV_allee_res, lineage)
                    
                    if with_community_function:
                        
                        # Calculate community community function, assign to class attributes
                        gLV_allee_res.call_community_function(self)
                        self.assign_community_function(gLV_allee_res, lineage)
          
           case 'Supply initial conditions':
              
               for count, lineage in enumerate(lineages):
                   
                   # Call gLV class to simulate community dynamics
                   gLV_allee_res = gLV_with_allee(self,t_end,
                                                  usersupplied_init_cond=array_of_init_conds[:,count])
                     
                   # Calculate community properties, assign to class attributes
                   gLV_allee_res.identify_community_properties()
                   self.assign_gLV_allee_attributes(gLV_allee_res, lineage)
                   
                   if with_community_function:
                       
                       # Calculate community community function, assign to class attributes
                       gLV_res.call_community_function(self)
                       self.assign_community_function(gLV_allee_res, lineage)
         
       # Calculate the number of unique species compositions for the species pool
       no_uniq_compositions, comps = unique_compositions(self)
       
       self.no_unique_compositions = no_uniq_compositions
       self.unique_composition_label = {'lineage '+ str(lineage) : comp for lineage, comp in zip(lineages, comps)}
       
   def assign_gLV_allee_attributes(self,gLV_allee_res,lineage):
       
       '''
       
       Assign community properties to class attributes

       Parameters
       ----------
       gLV_allee_res : object of class gLV
           gLV object/simulation results.
       lineage : int
           Lineage index/label.

       Returns
       -------
       None.

       '''
       
       dict_key = "lineage " + str(lineage)
       
       # Assign initial species abundances
       self.initial_abundances[dict_key] = gLV_allee_res.initial_abundances
       
       # Assign simulation results
       self.ODE_sols[dict_key] = gLV_allee_res.ODE_sol
       
       # Assign species composition at the end of simulation
       self.final_composition[dict_key] = gLV_allee_res.final_composition
       
       # Assign species diversity at the end of simulation
       self.diversity[dict_key] = gLV_allee_res.final_diversity
       
       # Assign community invasibility from species pool
       self.invasibilities[dict_key] = gLV_allee_res.invasibility
       
   def assign_community_function(self,gLV_allee_res,lineage):
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
       
       self.community_functions[dict_key] = gLV_allee_res.community_function
       
    
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

