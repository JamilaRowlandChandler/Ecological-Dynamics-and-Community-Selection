# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:29:00 2024

@author: jamil

=======================================================================
    This is my code for solving ordinary differential equations 
    (for the consumer-resource model).
=======================================================================
"""

# %%

import numpy as np
from scipy.integrate import solve_ivp

from initial_abundances import InitialConditionsInterface

#%%

class DifferentialEquationsInterface(InitialConditionsInterface):
    
    def simulate_community(self, t_end, no_init_cond, init_cond_func='Mallmin',
                           assign = True, **kwargs):
        
        self.t_end = t_end
        
        # generate initial species and resource abundances
        initial_abundances = \
            self.generate_initial_conditions(no_init_cond, init_cond_func, **kwargs)
            
        # simulate community dynamics for each set of initial conditions
        ODE_sols = [self.simulation(t_end, initial_abundances[:, i]) 
                    for i in range(initial_abundances.shape[1])]
        
        # should simulations be assigned to a class attribute
        if assign is True:
            
            self.ODE_sols = ODE_sols
        
        else:
            
            return ODE_sols
    

def unbounded_growth(t, var, *args):
    '''
    
    THIS IS IMPORTANT FOR YOU
    
    The function identifies whether unbounded growth (to infinity) is occuring.
    If there is unbounded growth, the function stops the ODE solver early.

    Parameters
    ----------
    t : float
        time.
    var : np.array() of floats
        Species and resource dynamics at timte t.
    *args : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        If 0 is returned, the ODE solver (solve_ivp) terminates.
        If a non-zero value is returned, the ODE solver continues to run

    '''
    
    
    # if any species or resource abundances are greater than some threshold
    # or if any species abundances are less than or equal to 0  
    if np.any(np.log(np.abs(var)) > 4) or np.isnan(np.log(np.abs(var))).any():
        
        return 0 # when the returned value of an event function is 0, the ode 
                    #solver terminates.
    
    else: 
        
        return 1 # the ode solver continues because the returned value is non-zero.
