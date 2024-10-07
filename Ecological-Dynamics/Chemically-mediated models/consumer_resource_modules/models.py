# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 10:42:17 2024

@author: jamil
"""

import numpy as np
from scipy.integrate import solve_ivp

from parameters import ParametersInterface
from differential_equations import DifferentialEquationsInterface
from community_level_properties import CommunityPropertiesInterface

from differential_equations import dCR_dt, dCR_dt_2, dCR_dt_3

###########################################################

class Consumer_Resource_Model(ParametersInterface, DifferentialEquationsInterface, CommunityPropertiesInterface):
    
    '''
    
    Consumer-Resource model (C-R).
    
    '''
    
    def __init__(self, no_species, no_resources,
                 growth_parameters  =  None, 
                 consumption_parameters = None,
                 dispersal = 1e-8):
        
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
        self.no_resources = no_resources
        
        for key, value in growth_parameters.items():
            
            # Assign growth function arguments as class attributes.
            setattr(self, key, value)
            
        for key, value in consumption_parameters.items():
            
            # Assign growth function arguments as class attributes.
            setattr(self,key,value)
        
        self.dispersal = dispersal