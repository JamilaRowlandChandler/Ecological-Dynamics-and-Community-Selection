# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:01:20 2024

@author: jamil
"""

################# Initial conditions module #############

import numpy as np

####### Functions #######

def initial_conditions(model_object,init_cond_func,usersupplied_init_cond):

    match init_cond_func:
            
            case 'Hu':
                
                initial_abundances = initial_abundances_hu(model_object)
            
            case 'Mallmin':
                
                initial_abundances = initial_abundances_mallmin(model_object)
            
            case None:
                
                # Assign initial conditions using the user-supplied initial abundances.
                initial_abundances = usersupplied_init_cond
    
    return initial_abundances
  
########## Functions for generating initial conditions ############
  
def initial_abundances_mallmin(model_object):
    
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
    
    return np.random.uniform(model_object.dispersal,2/model_object.no_species,
                             model_object.no_species)

def initial_abundances_hu(model_object):
    
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
    
    return np.random.uniform(0,2*model_object.mu_a,model_object.no_species)
