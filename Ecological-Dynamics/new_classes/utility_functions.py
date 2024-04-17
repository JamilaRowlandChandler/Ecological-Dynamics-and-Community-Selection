# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:12:02 2024

@author: jamil
"""

import numpy as np
from scipy import stats
import pandas as pd
import pickle

################## Utility functions #################

def generate_distribution(mu_maxmin,std_maxmin,dict_labels=['mu_a','sigma_a'],
                          mu_step=0.1,std_step=0.05):
    
    '''
    
    Generate parameters for the random interaction distribution.

    Parameters
    ----------
    mu_maxmin : list of floats
        Minimum and maximum mean interaction strength, mu_a.
    std_maxmin : list of floats
        Minimum and maximum standard deviation in interaction strength, sigma_a.
    mu_step : float, optional
        mu_a step size. The default is 0.1.
    std_step : float, optional
        sigma_a step size. The default is 0.05.

    Returns
    -------
    distributions : list of dicts
        Parameters for interaction distributions - [{'mu_a':mu_min,'sigma_a':std_min},
                                                    {'mu_a':mu_min,'sigma_a':std_min+std_step},...,
                                                    {'mu_a':mu_min+mu_step,'sigma_a':std_min},...,
                                                    {'mu_a':mu_max,'sigma_a':std_max}]

    '''
    
    # Extract min. mean interaction strength
    mu_min = mu_maxmin[0]
    # Extract max. mean interaction strength
    mu_max = mu_maxmin[1]
    
    # Extract min. standard deviation in interaction strength
    std_min = std_maxmin[0]
    # Extract max. standard deviation in interaction strength
    std_max = std_maxmin[1]
    
    # Generate range of mean interaction strengths
    mu_range = np.arange(mu_min,mu_max,mu_step)
    # Generate range of standard deviations in interaction strengths
    std_range = np.arange(std_min,std_max,std_step)

    # Generate dictionary of interaction distribution parameters sets
    mu_rep = np.repeat(mu_range,len(std_range))
    std_rep = np.tile(std_range,len(mu_range))
    
    distributions = [{dict_labels[0]:np.round(mu,2), dict_labels[1]:np.round(sigma,2)} \
                     for mu, sigma in zip(mu_rep,std_rep)]
     
    return distributions
   
def find_nearest_in_timeframe(timeframe,simulation_times):
    
    '''
    
    Find the index of the nearest times to those in timeframe 
        (for extracting population dynamics at a given time).
    

    Parameters
    ----------
    timeframe : list of ints or floats
        List of times.
    simulation_times : .t from OdeResult object of scipy.integrate.solve_ivp module
        Simulation times ONLY from (deterministic) solution to gLV ODE system.
    

    Returns
    -------
    indices : int
        indices of times in simulation_times with value

    '''
    
    indices = find_nearest_multivalues(timeframe,simulation_times)
    
    return indices

def find_nearest_multivalues(array_of_values,find_in):
    
    '''
    
    Find nearest value in array for multiple values. Vectorised.
    
    Parameters
    ----------
    array_of_values : np.array of floats or inds
        array of values.
    find_in : np.array of floats or inds
        array where we want to find the nearest value (from array_of_values).
    
    Returns
    -------
    fi_ind[sorted_idx-mask] : np.array of inds
        indices of elements from find_in closest in value to values in array_of_values.
    
    '''
     
    L = find_in.size # get length of find_in
    fi_ind = np.arange(0,find_in.size) # get indices of find_in
    
    sorted_idx = np.searchsorted(find_in, array_of_values)
    sorted_idx[sorted_idx == L] = L-1
    
    mask = (sorted_idx > 0) & \
    ((np.abs(array_of_values - find_in[sorted_idx-1]) < np.abs(array_of_values - find_in[sorted_idx])))
    
    return fi_ind[sorted_idx-mask]

def mean_stderror(data):
    
    '''
    
    Calculate the mean and standard error of a dataset.
    
    Parameters
    ----------
    data : np.array
        Dataset.
    Returns
    -------
    [mean, std_error] : list of floats
        The mean and standard error of data.
    
    '''
    
    mean = np.mean(data)
    
    std_error = stats.sem(data)
    
    return [mean, std_error]

def mean_std_deviation(data):
    
    '''
    
    Calculate the mean and standard error of a dataset.
    
    Parameters
    ----------
    data : np.array
        Dataset.
    Returns
    -------
    [mean, std_error] : list of floats
        The mean and standard error of data.
    
    '''
    
    mean = np.mean(data)
    
    std_deviation = np.std(data)
    
    return [mean, std_deviation]

def community_object_to_df(community_object,
                         community_attributes=['mu_a','sigma_a','no_species',
                                               'no_unique_compositions','unique_composition_label',
                                               'diversity','invasibility'],
                         community_label=0,
                         column_names=None):
    
    no_lineages = len(community_object.ODE_sols)
    
    community_col = np.repeat(community_label,no_lineages)
    
    lineage_col = list(community_object.ODE_sols.keys())
    lineage_col = [int(lineage.replace('lineage ','')) for lineage in lineage_col]
    
    ###############################################
    
    def extract_attribute_make_df_col(community_object,attribute_name,no_lineages=no_lineages):
        
        try:
        
            attribute = getattr(community_object,attribute_name)
            
        except AttributeError:
        
            raise Exception('Community object has no attribute ' + str(attribute_name))
            
            exit
        
        if isinstance(attribute,(int,float,str,np.int32,np.int64,np.float32,np.float32)):
            
            attribute_col = np.repeat(attribute,no_lineages)
            
        elif isinstance(attribute,dict):
            
            attribute_col = list(attribute.values())
        
        elif isinstance(attribute,(list,np.ndarraytuple)):
            
            attribute_col = attribute
        
        return attribute_col
    
    attribute_columns = [community_col] + [lineage_col] + \
        [extract_attribute_make_df_col(community_object, attribute_name) \
             for attribute_name in community_attributes]
            
    if column_names:
            
        col_names = ['community','lineage'] + column_names
        
    else:
        
        col_names = ['community','lineage'] + community_attributes
        
    ############# Convert lists to df ################
    
    community_df = pd.DataFrame(attribute_columns)
    community_df = community_df.T
    community_df = community_df.set_axis(col_names,axis=1)
    
    return community_df
    
def pickle_dump(filename,data):
    
    '''
    
    Pickle data.

    Parameters
    ----------
    filename : string
        Pickle file name. Should end with .pkl
    data : any
        Data to pickle.

    Returns
    -------
    None.

    '''
    
    with open(filename, 'wb') as fp:
        
        pickle.dump(data, fp)
