# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:50:53 2024

@author: jamil
"""

################# Initial conditions module #############

import numpy as np

####### Functions #######

class InitialConditionsInterface:
    
    def generate_initial_conditions(self, lineages, 
                                    init_cond_func, **kwargs):
        
        '''
        
        Generate and assign initial conditions from multiple options/functions.
        
        '''
        
        def initial_variable_conditions(variable, lineages, dims, **kwargs):
            
            no_lineages = len(lineages)
        
            match init_cond_func:
                
                case 'Mallmin':
                    
                    initial_abundances = self.initial_abundances_mallmin(no_lineages, dims)
                
                case 'user supplied':
                    
                    initial_abundances = kwargs.get('user_supplied_init_cond')[variable].reshape((dims, no_lineages))
                    
            return initial_abundances
        
        species_abundances = initial_variable_conditions('species', lineages, self.no_species, **kwargs)
        resource_abundances = initial_variable_conditions('resources', lineages, self.no_resources, **kwargs)
        
        return np.concatenate((species_abundances, resource_abundances))
    
    ########## Functions for generating initial conditions ############
      
    def initial_abundances_mallmin(self, no_lineages, dims):
        
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
    
        return np.random.uniform(self.dispersal, 2/dims, dims * no_lineages).reshape((dims, no_lineages))
    