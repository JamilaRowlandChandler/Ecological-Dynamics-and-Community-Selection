# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:01:20 2024

@author: jamil
"""

################# Initial conditions module #############

import numpy as np

####### Functions #######

class InitialConditionsInterface:
    
    def generate_initial_conditions(self,lineages,init_cond_func,usersupplied_init_conds):
        
        '''
        
        Generate and assign initial conditions from multiple options/functions.
        
        '''
        
        match init_cond_func:
            
            case 'Hu':
                
                initial_abundances = self.initial_abundances_hu(lineages)
            
            case 'Mallmin':
                
                initial_abundances = self.initial_abundances_mallmin(lineages)
            
            case None:
                
                # Assign initial conditions using the user-supplied initial abundances.
                initial_abundances = usersupplied_init_conds
        
        return initial_abundances
    
    ########## Functions for generating initial conditions ############
      
    def initial_abundances_mallmin(self,lineages):
        
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
        
        no_lineages = len(lineages)
        
        return np.random.uniform(self.dispersal,2/self.no_species,
                                 self.no_species*no_lineages).reshape((self.no_species,no_lineages))
    
    def initial_abundances_hu(self,lineages):
        
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
        
        no_lineages = len(lineages)
        
        return np.random.uniform(0,2*self.mu_a,
                                 self.no_species*no_lineages).reshape((self.no_species,no_lineages))

