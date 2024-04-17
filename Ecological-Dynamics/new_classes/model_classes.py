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
    
    def __init__(self,no_species,growth_func,growth_args,interact_func,interact_args,
                 usersupplied_growth=None,usersupplied_interactmat=None,
                 dispersal=1e-8):
        
        # This function creates parameters
        
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
                    self.species_contribution_to_community_function(self.no_species,
                                                                    **community_func_args)
                
            case None: 
                
                self.species_contribution_community_function = usersupplied_community_function
       
    def simulate_community(self,lineages,t_end,init_cond_func='Mallmin',
                           usersupplied_init_conds=None):
        
        self.t_end = t_end
        
        initial_abundances = \
            self.generate_initial_conditions(lineages,init_cond_func,usersupplied_init_conds)
        
        # Simulations
        self.ODE_sols = {'lineage ' + str(lineage) : self.gLV_simulation(initial_abundances[:,i]) \
                         for i, lineage in enumerate(lineages)}    
        
    def gLV_simulation(self,initial_abundance):
        
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
        
        def gLV_ODE(t,spec,growth_r,interact_mat,dispersal,extinct_thresh=1e-9):
            
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
            
            #dSdt = np.multiply(1 - np.matmul(interact_mat,spec), growth_r*spec) + dispersal
            dSdt = np.multiply(growth_r - np.matmul(interact_mat,spec), spec) + dispersal
            
            return dSdt
        
        return solve_ivp(gLV_ODE,[0,self.t_end],initial_abundance,
                         args=(self.growth_rates,self.interaction_matrix,self.dispersal),
                         method='RK45',t_eval=np.linspace(0,self.t_end,200))
    
class gLV_allee(ParametersInterface, InitialConditionsInterface, CommunityPropertiesInterface):
    
    def __init__(self,no_species,growth_func,growth_args,
                 competition_func,competition_args,cooperation_func,cooperation_args,
                 usersupplied_growth=None,usersupplied_competition=None,
                 usersupplied_cooperation=None,dispersal=1e-8):
        
        # This function creates parameters
        
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
        
        if competition_args:
            
           for key, value in competition_args.items():
               
               # Assign interaction matrix function arguments as class attributes.
               setattr(self,key,value) 
        
        match competition_func:
            
            case 'random':
                    
                # Generate interaction matrix 
                self.competition_matrix = \
                    self.random_interaction_matrix(self.mu_comp,self.sigma_comp)
                
            case 'sparse':
            
                # Generate interaction matrix 
                self.competition_matrix = \
                    self.sparse_interaction_matrix(self.mu_comp,self.sigma_comp,self.connectance_comp)
                
            case 'modular':
                
                self.interaction_matrix = \
                    self.modular_interaction_matrix(self.p_mu_comp,self.p_sigma_comp,
                                                    self.p_connectance,
                                                    self.q_mu_comp,self.q_sigma_comp,
                                                    self.q_connectance)
                
            case 'nested':

                # Generate interaction matrix 
                self.competition_matrix = \
                    self.nested_interaction_matrix(self.mu_comp,self.sigma_comp,
                                                   self.average_degree_comp)
            
            case None:
                
                # Assign interaction matrix using the user-supplied interaction matrix, if supplied.
                self.competition_matrix = usersupplied_competition
                
        ################
        
        if cooperation_args:
            
           for key, value in cooperation_args.items():
               
               # Assign interaction matrix function arguments as class attributes.
               setattr(self,key,value) 
        
        match cooperation_func:
            
            case 'random':
                    
                # Generate interaction matrix 
                self.cooperation_matrix = \
                    self.random_interaction_matrix(self.mu_coop,self.sigma_coop,self_interaction=0)
                
            case 'sparse':
            
                # Generate interaction matrix 
                self.cooperation_matrix = \
                    self.sparse_interaction_matrix(self.mu_coop,self.sigma_coop,self.connectance_coop,
                                                   self_interaction=0)
                
            case 'modular':
                
                self.interaction_matrix = \
                    self.modular_interaction_matrix(self.p_mu_coop,self.p_sigma_coop,
                                                    self.p_connectance,
                                                    self.q_mu_coop,self.q_sigma_coop,
                                                    self_interaction=0)
                
            case 'nested':

                # Generate interaction matrix 
                self.cooperation_matrix = \
                    self.nested_interaction_matrix(self.mu_coop,self.sigma_comp,
                                                   self.average_degree_coop,
                                                   self_interaction=0)
            
            case None:
                
                # Assign interaction matrix using the user-supplied interaction matrix, if supplied.
                self.cooperation_matrix = usersupplied_cooperation
            
        self.dispersal = dispersal
       
    def simulate_community(self,lineages,t_end,init_cond_func='Mallmin',
                           usersupplied_init_conds=None):
        
        self.t_end = t_end
        
        initial_abundances = \
            self.generate_initial_conditions(lineages,init_cond_func,usersupplied_init_conds)
        
        # Simulations
        self.ODE_sols = {'lineage ' + str(lineage) : self.gLV_allee_simulation(initial_abundances[:,i]) \
                         for i, lineage in enumerate(lineages)}    
        
    def gLV_allee_simulation(self,initial_abundance):
        
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
        
        def gLV_allee_ODE(t,spec,growth_r,competitive_mat,cooperative_mat,gamma,
                          dispersal,extinct_thresh=1e-9):
            
            spec[spec < extinct_thresh] = 0 # set species abundances below extinction threshold to 0
            
            competition = np.matmul(competitive_mat,spec)
            cooperation = np.matmul(cooperative_mat,spec/(gamma+spec))
            
            #dSdt = np.multiply(1 + cooperation - competition, growth_r*spec) + dispersal
            dSdt = np.multiply(growth_r + cooperation - competition, spec) + dispersal
            
            return dSdt
        
        return solve_ivp(gLV_allee_ODE,[0,self.t_end],initial_abundance,
                         args=(self.growth_rates,self.competition_matrix,
                               self.cooperation_matrix,1,self.dispersal),
                         method='RK45',t_eval=np.linspace(0,self.t_end,200))