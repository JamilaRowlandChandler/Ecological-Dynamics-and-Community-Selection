# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 10:42:17 2024

@author: jamil
"""

from parameters import ParametersInterface
from differential_equations import DifferentialEquationsInterface
from community_level_properties import CommunityPropertiesInterface

###########################################################

class Consumer_Resource_Model(ParametersInterface, DifferentialEquationsInterface, CommunityPropertiesInterface):
    
    '''
    
    Consumer-Resource model (C-R).
    
    '''
    
    def __init__(self, no_species, no_resources, parameters,
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
        
        #def assign_parameters(self, parameter_names, attributes):
            
        #    match parameter_names:
                
        #        case 'correlation':
                    
        #            suffix = ''
                
        #        case 'growth':
                    
        #            suffix = '_g'
                
        #        case 'consumption':
                    
        #            suffix = '_c'
                
        #        case 'death':
                    
        #            suffix = '_m'
                
        #        case 'influx':
                    
        #            suffix = '_K'
                    
        #        case 'outflux':
                    
        #            suffix = '_D'
                 
        #    for attr_name, attr_val in attributes.items():
            
        #        setattr(self, attr_name + suffix, attr_val)
        
        self.no_species = no_species
        self.no_resources = no_resources
        
        for key, value in parameters.items():
            
            # Assign growth function arguments as class attributes.
            #assign_parameters(self, key, value)
            setattr(self, key, value)
            
        self.dispersal = dispersal