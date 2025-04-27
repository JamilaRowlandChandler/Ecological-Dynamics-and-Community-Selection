# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:32:42 2024

@author: jamil
"""

import numpy as np
from scipy.integrate import solve_ivp

from model_parameters import ParametersInterface
from initial_conditions import InitialConditionsInterface
from community_properties import CommunityPropertiesInterface

#from utility_functions import *

class gLV(ParametersInterface, InitialConditionsInterface, CommunityPropertiesInterface):
    
    '''
    
    Generalised Lotka-Volterra model (gLV).
    
    '''
    
    def __init__(self,no_species,growth_func,growth_args,interact_func,interact_args,
                 usersupplied_growth=None,usersupplied_interactmat=None,
                 dispersal=1e-8):
        
        '''
        
        Initialise gLV class by assigning and generating model parameters.
        
        See ParametersInterface in model_parameters.py for details on functions
        that generate model parameters.

        Parameters
        ----------
        no_species : float
            Number of species, or size of species pool.
        growth_func : string
            Name of the function used to generate growth rates.
        growth_args : dict
            Growth rates function arguments.
        interact_func : TYPE
            Name of the function used to generate the interaction matrix.
        interact_args : dict
            Interaction matrix function_arguments.
        usersupplied_growth : np.ndarray of floats, size (no_species,), Optional
            User-supplied growth rates (if you do not want to use an in-built method). 
            The default is None.
        usersupplied_interactmat : np.ndarray of floats, size (no_species,no_species), Optional
            User-supplied interaction matrix (if you do not want to use an in-built method).
            The default is None.
        dispersal : float, optional
            Dispersal/migration rate. The default is 1e-8.

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
                    setattr(self,key,value)
                    
                self.growth_rates = self.growth_rates_norm()
                
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
                self.interaction_matrix = \
                    self.random_interaction_matrix(self.mu_a,self.sigma_a)
                
            case 'sparse':
            
                # Generate interaction matrix 
                self.interaction_matrix = \
                    self.sparse_interaction_matrix(self.mu_a,self.sigma_a,self.connectance)
                
            case 'modular':
                
                self.interaction_matrix = \
                    self.modular_interaction_matrix(self.p_mu_a,self.p_sigma_a,
                                                    self.p_connectance,
                                                    self.q_mu_a,self.q_sigma_a,
                                                    self.q_connectance)
                    
            case 'nested':
        
                # Generate interaction matrix 
                self.interaction_matrix = \
                    self.nested_interaction_matrix(self.mu_a,self.sigma_a,
                                                   self.average_degree)
                    
            case None:
                        
                # Assign interaction matrix using the user-supplied interaction matrix, if supplied.
                self.interaction_matrix = usersupplied_interactmat    
            
        self.dispersal = dispersal
          
    def simulate_community(self,lineages,t_end,init_cond_func='Mallmin',
                           usersupplied_init_conds=None,assign=True):
        
        '''
        
        Simulate community dynamics from different initial conditions
        
        Parameters
        ----------
        lineages : list or np.ndarray of ints
            lineages indexes. 
            len(lineages) = no. of initial conditions to simulate community dynamics from
        t_end : float
            Amount of time to run simulation
        initi_cond_func : string, Optional
            Name of function to generate initial species abundances. The default is 'Mallmin'.
        usersupplied_init_conds : np.ndarray of size (self.no_species,len(lineages))
            User-supplied initial species abundances (if the user does not want to use an in-built method).
            The default is None.
        
        Returns
        -------
        None
        
        '''
        
        # Assign the end of simulations as a class attribute 
        self.t_end = t_end
        
        # Generate n different sets of initial species abundances (n = no. lineages)
        initial_abundances = \
            self.generate_initial_conditions(lineages,init_cond_func,usersupplied_init_conds)
            
        # Simulate community dynamics from different initial conditions/lineages
        ODE_sols = {'lineage ' + str(lineage) : self.gLV_simulation(initial_abundances[:,i]) \
                         for i, lineage in enumerate(lineages)}  
        
        # Determine if simulations should be assigned as a class attribute.
        if assign is True:
            
            self.ODE_sols = ODE_sols
        
        else:
            
            return ODE_sols
        
    def gLV_simulation(self,initial_abundance):
        
        '''
        
        Simulate generalised Lotka-Volterra dynamics.

        Parameters
        ----------
        init_abundance : np.array of float64, size (n,)
            Initial species abundances.

        Returns
        -------
         OdeResult object of scipy.integrate.solve_ivp module
            (Deterministic) Solution to gLV ODE system.

        '''
        
        def gLV_ODE(t,spec,growth_r,interact_mat,dispersal,extinct_thresh=1e-9):
            
            # set species abundances below extinction threshold to 0
            spec[spec < extinct_thresh] = 0 
            
            dSdt = np.multiply(growth_r - np.matmul(interact_mat,spec), spec) + dispersal
            
            return dSdt
        
        return solve_ivp(gLV_ODE,[0,self.t_end],initial_abundance,
                         args=(self.growth_rates,self.interaction_matrix,self.dispersal),
                         method='RK45',t_eval=np.linspace(0,self.t_end,200))
    