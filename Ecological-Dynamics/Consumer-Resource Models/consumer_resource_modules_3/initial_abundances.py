# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:50:53 2024

@author: jamil
"""

################# Initial conditions module #############

import numpy as np

####### Functions #######

class InitialConditionsInterface:
    
    def generate_initial_conditions(self, no_init_cond, 
                                    init_cond_func, **kwargs):
        
        '''
        
        Generate and assign initial abundances for species and resources 
        from multiple options/functions.
        
        '''
        
        species_abundances = self.__initial_variable_conditions('species',
                                                                no_init_cond,
                                                                self.no_species,
                                                                init_cond_func,
                                                                **kwargs)
        resource_abundances = self.__initial_variable_conditions('resources',
                                                                 no_init_cond,
                                                                 self.no_resources,
                                                                 init_cond_func,
                                                                 **kwargs)
        
        return np.vstack((species_abundances, resource_abundances))
        
    def __initial_variable_conditions(self, variable, no_init_cond, dims,
                                      init_cond_func, **kwargs):
        
        '''
        
        Generate initial abundances for one variable.
        
        '''
        
    
        match init_cond_func:
            
            case 'Mallmin':
                
                initial_abundances = self.__initial_abundances_mallmin(no_init_cond, dims)
            
            case 'user supplied':
                
                initial_abundances = kwargs.get('user_supplied_init_cond')[variable].reshape((dims, no_init_cond))
                
        return initial_abundances
        
    ########## Functions for generating initial conditions ############
      
    def __initial_abundances_mallmin(self, no_init_cond, dims):
        
        '''
        
        Sample multiple sets of initial abundances from Uniform(dispersal, 2/M)

        Parameters
        ----------
        no_init_cond : int
            Number of sets of initial abundances to sample.
        dims : int
            Number of abundances to sample per set (e.g. species or resource pool size).

        Returns
        -------
        np.ndarray with dimensions (dims, no_init_cond)
            Array of sets of initial abundances.

        '''
    
        return np.random.uniform(1e-8, 2/dims, dims * no_init_cond).reshape((dims, no_init_cond))
    