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
    
    def simulate_community(self, lineages, t_end, init_cond_func='Mallmin',
                           model_version = 'self-limiting resource supply',
                           assign = True, **kwargs):
        '''
        
        Simulate community dynamics from different initial conditions

        Parameters
        ----------
        lineages : np.array() of ints
            no. of initial conditions e.g. if 3 initial conditions, lineages = [0,1,2]
        t_end : float
            simulation end time.
        init_cond_func : string optional
            Function used to generate initial abundances. The default is 'Mallmin'.
        model_version : string, optional
            Function describing which set of ODEs should be used.
            The default is 'self-limiting resource supply'.
        assign : Boolean, optional
            Determines whether simulation results should be assigned as a class attribute.
            The default is True.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        ODE_sols : dict
            Simulation results for each set of initial abundances/lineage.

        '''
        
        # generate initial species and resource abundances for each lineage
        initial_abundances = \
            self.generate_initial_conditions(lineages, init_cond_func, **kwargs)
        
        # simulate community dynamics for each set of initial conditions/lineage    
        ODE_sols = {'lineage ' + str(lineage) : self.CR_simulation(t_end, initial_abundances[:,i], model_version) \
                         for i, lineage in enumerate(lineages)}
        
        # should simulations be assigned to a class attribute
        if assign is True:
            
            self.ODE_sols = ODE_sols
        
        else:
            
            return ODE_sols
            
    def CR_simulation(self, t_end, initial_abundance, model_version):
        
        '''
        
        Simulate model dynamics for a single set of initial conditions.
        Calls the scipy.integrate.solve_ivp ODE solver t(o solve the ODEs).

        Parameters
        ----------
        t_end : float
            Simulation end time.
        initial_abundance : np.ndarray of floats
            Initial species and resource abundances.
        model_version : string
            Function describing which set of ODEs should be used.

        Returns
        -------
        Bunch object of class scipy.integrate
            Simulated community dynamics over time.

        '''
        
        # specify which ODE function and parameters are used based on the model 
        #   called.
        match model_version:
            
            case 'self-limiting resource supply':
                
                model = dCR_dt
                p_args = (self.no_species, self.eff_growth, self.death,
                          self.eff_consumption, self.influx, self.dispersal)
                
            case 'external resource supply':
                
                model = dCR_dt_2
                p_args = (self.no_species, self.eff_growth, self.death,
                          self.eff_consumption, self.influx, self.outflux,
                          self.dispersal)
        
        # THIS IS IMPORTANT FOR YOU - this tells solve_ivp to call the unbounded 
        #   growth event function
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model, [0, t_end], initial_abundance, args = p_args,
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         #method = 'Radau',
                         #method = 'BDF', rtol = 1e-8, atol = 1e-8,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
    
# %%

def dCR_dt(t, var,
           no_species,
           growth, death, consumption, influx,
           dispersal):
    
    '''
    
    ODEs for the consumer-resource model with self-limiting resource supply
    
    '''
    
    species = var[:no_species]
    resources = var[no_species:]

    dSdt = species * (np.sum(growth * resources, axis=1) - death) + dispersal
    
    dRdt = (resources * (influx - resources)) - \
        (resources * np.sum(consumption * species, axis=1)) + dispersal

    return np.concatenate((dSdt, dRdt))


def dCR_dt_2(t, var,
             no_species,
             growth, death, consumption, influx, outflux,
             dispersal):
    
    '''
    
    ODEs for the consumer-resource model with externally supplied resources.
    
    '''
    
    species = var[:no_species]
    resources = var[no_species:]
    
    #breakpoint()

    dSdt = species * (np.sum(growth * resources, axis=1) - death) + dispersal

    dRdt = (influx - outflux * resources) - \
            (resources * np.sum(consumption * species, axis=1)) + dispersal

    return np.concatenate((dSdt, dRdt))

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
