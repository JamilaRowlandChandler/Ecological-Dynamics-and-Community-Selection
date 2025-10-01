# -*- coding: utf-8 -*-
"""
Created on Sun May  4 13:01:05 2025

@author: jamil
"""

import numpy as np
from scipy.integrate import solve_ivp

from parameters import ParametersInterface
from differential_equations import DifferentialEquationsInterface
from differential_equations import unbounded_growth
from community_level_properties import CommunityPropertiesInterface
#from initial_abundances import InitialConditionsInterface

# %%

def Consumer_Resource_Model(model, no_species, no_resources):
    
    '''
    
    Wrapper for different consumer resource model classes

    Parameters
    ----------
    model : str
        Type of consumer resource model. Options are:
            "Self-limiting resource supply" - resources grow logistically
            "Self-limiting resource supply, self-inhibition" - same as 
            "Self-limiting resource supply", but with direct consumer self-inhibition
            "Externally-supplied resources" - chemostat-style resource dynamics
            (constant influx + dilution)
    no_species : int
        species pool size
    no_resources : int
        resource pool size

    Raises
    ------
    Exception
        If a non-existent model is selected.

    Returns
    -------
    instance : object of some consumer-resource model class
        Instance of some consumer-resource model class.

    '''
    
    match model:
        
        case "Self-limiting resource supply":
            
            instance = SL_CRM(no_species, no_resources)
            
        case "Self-limiting resource supply, self-inhibition":
            
            instance = SL_SI_CRM(no_species, no_resources)
            
        #case "Externally-supplied resources":
            
            #instance = ES_CRM(no_species, no_resources)
            
        case _:
            
            raise Exception('You have not selected an exisiting model.\n' + \
                  'Please chose from either "Self-limiting resource supply"' + \
                      ' or "Externally-supplied resources"')
    return instance

# %%

class SL_CRM(ParametersInterface, DifferentialEquationsInterface, CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply
    
    '''
    
    def __init__(self, no_species, no_resources):
        
        '''
        
        Initiate model
        
        Parameters
        ----------
        no_species : int
            species pool size
        no_resources : int
            resource pool size

        Returns
        -------
        None.

        '''
        
        # assign species and resource pool size as class attributes
        self.no_species = no_species
        self.no_resources = no_resources
        
    def model_specific_rates(self,
                             death_method = 'constant',
                             death_args = {'d' : 1},
                             resource_growth_method = 'constant',
                             resource_growth_args = {'b' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args : dict
            Arguments for death_method.
            If 'normal', first argument is the mean, second is the stand deviation
            e.g., {'mu': mean, 'sigma' : mean}
            If 'constant', the key is the parameter name, argument is the fixed value
            e.g., {'d' : val}
            If 'used-supplied', argument is the array of death rates 
            e.g., {'d' : array_of_vals}
        resource_growth_method : str
            Method used to generate intrinsic resource growth rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        resource_growth_args : dict
            Arguments for resource_growth_method. Options are the same as 
            death_method args.

        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b']
        
        # dimensions for death rates and intrinsic growth rates
        dims_list = [(self.no_species, ), (self.no_resources, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, resource_growth_method],
                [death_args, resource_growth_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
    
    #####################################################################
    
    def simulation(self, t_end, initial_abundance):
        
        '''
        
        Simulate community dynamics
        
        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, y,
                  S, G, C, D, B):
            
            '''
            
            ODE for CRM with self-limiting resource supply

            Parameters
            ----------
            t : float
                time
            y : np.ndarray
                consumer and resource abundances at time t
            S : int
                species pool size (used to separate y into species and resource 
                                   abundances)
            G : np.ndarray
                matrix of consumer growth rates
            C : np.ndarray
                matrix of resource consumption rates
            D : np.ndarray
                consumer death rates
            B : np.ndarray
                intrinsic resource growth rates

            Returns
            -------
            np.ndarray
                Rate of change in species and resource abundances over time 
                (dNdt and dRdt)

            '''
            
            # separate species and resource abundances
            species, resources = y[:S], y[S:]
            
            # change in consumer abundances over time
            dNdt = species * (np.sum(G * resources, axis = 1) - D)
            
            # change in resource abundances over time
            dRdt = (resources * (B - resources)) - \
                (resources * np.sum(C * species, axis=1))

            return np.concatenate((dNdt, dRdt)) + 1e-8
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model, [0, t_end], initial_abundance, 
                         args = (self.no_species, self.growth, self.consumption, 
                                 self.d, self.b),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
    
# %%

class SL_SI_CRM(ParametersInterface, DifferentialEquationsInterface,
                CommunityPropertiesInterface):
    
    '''
    
    Consumer-resource model (CRM) class with self-limiting resource supply and
    direct consumer self-inhibition
    
    '''
    
    def __init__(self, no_species, no_resources):
        
        '''
        
        Initiate model
        
        Parameters
        ----------
        no_species : int
            species pool size
        no_resources : int
            resource pool size

        Returns
        -------
        None.

        '''
        
        self.no_species = no_species
        self.no_resources = no_resources
        
    def model_specific_rates(self,
                             death_method = 'constant',
                             death_args = {'d' : 1},
                             resource_growth_method = 'constant',
                             resource_growth_args = {'b' : 1},
                             si_method = 'constant',
                             si_args = {'si' : 1}):
        
        '''
        
        Generate parameters specific to the CRM with self-limiting resource 
        dynamics - consumer death rates and intrinsic resource growth rates


        Parameters
        ----------
        death_method : str
            Method used to generate death rates. Options are:
                'normal' : normally distributed parameters
                'constant' : death rates are fixed
                'user-supplied' : supply your own death rates
        death_args :  Arguments for death_method.
             If 'normal', first argument is the mean, second is the stand deviation
             e.g., {'mu': mean, 'sigma' : mean}
             If 'constant', the key is the parameter name, argument is the fixed value
             e.g., {'d' : val}
             If 'used-supplied', argument is the array of death rates 
             e.g., {'d' : array_of_vals}
        resource_growth_method : str
            Method used to generate intrinsic resource growth rates. Options are
            the same as death_method, but named 'b' rather than 'd'
        resource_growth_args : dict
            Arguments for resource_growth_method. Options are the same as 
            death_method args.
        si_method : str
            Method used to generate direct self-interaction coefficients between consumers.
            Options are the same as death_method
        si_args : dict
            Arguments for si_method. Options are the same as death_method args,
            but named 'si' rather than 'd'

        Returns
        -------
        None.

        '''
        
        # labels used to assign parameters as object attributes
        p_labels = ['d', 'b', 'si']
        
        # dimensions for death rates, intrinsic growth rates, and self-interaction coeffients
        dims_list = [(self.no_species, ), (self.no_resources, ),
                     (self.no_species, )]
        
        # generate parameters used the other_parameter_methods method
        for p_method, p_args, p_label, dims in \
            zip([death_method, resource_growth_method, si_method],
                [death_args, resource_growth_args, si_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
    
    #####################################################################
    
    def simulation(self, t_end, initial_abundance):
        
        '''
        
        Simulate community dynamics
        
        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray
            Initial abundances of species and resources.

        Returns
        -------
        Bunch object produced by scipy.integrate.solve_ivp
            Simulation.

        '''
        
        def model(t, y,
                  S, G, C, D, B, SI):
            
            '''
            
            ODE for CRM with self-limiting resource supply

            Parameters
            ----------
            t : float
                time
            y : np.ndarray
                consumer and resource abundances at time t
            S : int
                species pool size (used to separate y into species and resource 
                                   abundances)
            G : np.ndarray
                matrix of consumer growth rates
            C : np.ndarray
                matrix of resource consumption rates
            D : np.ndarray
                consumer death rates
            B : np.ndarray
                intrinsic resource growth rates
            SI : np.ndarray
                consumer self-interaction coefficients 

            Returns
            -------
            np.ndarray
                Rate of change in species and resource abundances over time 
                (dNdt and dRdt)

            '''
            
            # separate species and resource abundances
            species, resources = y[:S], y[S:]
            
            # change in consumer abundances over time
            dNdt = species * ((np.sum(G * resources, axis = 1) - D) - SI*species)
           
            # change in resource abundances over time
            dRdt = (resources * (B - resources)) - \
                (resources * np.sum(C * species, axis=1))

            return np.concatenate((dNdt, dRdt)) + 1e-8
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model, [0, t_end], initial_abundance, 
                         args = (self.no_species, self.growth, self.consumption, 
                                 self.d, self.b, self.si),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)

# %%

'''

class ES_CRM(ParametersInterface, CommunityPropertiesInterface, InitialConditionsInterface):
    
    def __init__(self, no_species, no_resources):
        
        self.no_species = no_species
        self.no_resources = no_resources
    
    def model_specific_rates(death, influx, outflux):
        
        pass
    
    def simulate_community(self, t_end, no_init_cond, init_cond_func='Mallmin',
                           assign = True, **kwargs):
        
        def simulation(t_end, initial_abundance):
            
            def model(t, y,
                      no_species, growth, death, consumption, influx, outflux,
                      dispersal):
             
             
             #ODEs for the consumer-resource model with externally supplied resources.
             
             
             species = y[:no_species]
             resources = y[no_species:]
             
             #breakpoint()

             dSdt = species * (np.sum(growth * resources, axis=1) - death) + dispersal

             dRdt = (influx - outflux * resources) - \
                     (resources * np.sum(consumption * species, axis=1)) + dispersal

             return np.concatenate((dSdt, dRdt))
            
            unbounded_growth.terminal = True
            
            # call the ODE solver with the unbounded growth event function
            # the ode solver stops when the event function is true (returns 0)           
            return solve_ivp(model, [0, t_end], initial_abundance, 
                             args = (self.no_species, self.growth, self.death,
                                     self.consumption, self.influx, self.dispersal),
                             method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                             t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
        
        # generate initial species and resource abundances
        initial_abundances = \
            self.generate_initial_conditions(np.arange(no_init_cond), init_cond_func, **kwargs)
        
        # simulate community dynamics for each set of initial conditions
        ODE_sols = [simulation(t_end, initial_abundance) 
                    for initial_abundance in initial_abundances]
        
        # should simulations be assigned to a class attribute
        if assign is True:
            
            self.ODE_sols = ODE_sols
        
        else:
            
            return ODE_sols
        
'''