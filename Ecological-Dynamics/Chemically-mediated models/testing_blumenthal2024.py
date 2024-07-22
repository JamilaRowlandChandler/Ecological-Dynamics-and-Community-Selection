# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:51:01 2024

@author: jamil
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:54:13 2024

@author: jamil
"""

#%%

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

########################

#%%

def dCR_dt(t,var,
           no_species,
           growth, death, consumption, weight, influx):
    
    #breakpoint()
    
    species = var[:no_species]
    resources = var[no_species:]
    
    dSdt = species * (np.sum(growth * resources, axis = 1) - death)
    
    dRdt = resources * (influx - resources) - \
        resources * np.sum(growth.T * consumption * species, axis = 1)
    
    return np.concatenate((dSdt, dRdt))
    
    pass

#%%

def normal_distributed_parameters(mu, sigma, dims):
    
    return mu + sigma*np.random.randn(*dims)
    
#%%

no_species = 100
no_resources = 50

death = np.ones(no_species)
influx = \
    normal_distributed_parameters(1, 0.1, (no_resources, 1)).reshape((no_resources,))
growth = normal_distributed_parameters(1, 0.3, (no_species, no_resources))
consumption = normal_distributed_parameters(0.9, 0.2, (no_resources, no_species))

initial_abundances = np.random.uniform(0.1,1,no_species)
initial_concentrations = np.random.uniform(1,2,no_resources)

#%%

simulation = solve_ivp(dCR_dt, [0, 1500], 
                       np.concatenate((initial_abundances, initial_concentrations)),
                       args = (no_species, growth, death, consumption, weight, influx))

#%%

plt.plot(simulation.t, simulation.y[:no_species,:].T)
plt.ylim([0,0.4])

#%%
plt.plot(simulation.t, simulation.y[no_species:,:].T)
plt.ylim([0,0.4])

#%%

diversity = \
    np.count_nonzero(np.any(simulation.y[:no_species,-100:] > 1e-4, axis = 1))
print('Species diversity =', diversity, '\n')

resource_diversity =  \
    np.count_nonzero(np.any(simulation.y[no_species:,-100:] > 1e-4, axis = 1))
print('Final no. resources =', resource_diversity, '\n')


