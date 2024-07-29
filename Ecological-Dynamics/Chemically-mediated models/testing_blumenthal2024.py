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

# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\Chemically-mediated models

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr

########################

#%%

def dCR_dt(t,var,
           no_species,
           growth, death, consumption, influx):
    
    var[var < 1e-9] = 0
    
    species = var[:no_species]
    resources = var[no_species:]
    
    #dSdt = species * (np.sum(growth * resources, axis = 1) - death) + 10**-8
    
    #dRdt = resources * (influx - resources) - \
    #    resources * np.sum(growth.T * consumption * species, axis = 1) + 10**-8
    
    #dRdt = influx - \
    #    resources * np.sum(growth.T * consumption * species, axis = 1)
    
    dSdt = species * (np.sum(growth * (resources/(1 + resources)), axis = 1) - death) + 10**-8
    
    dRdt = resources * (influx - resources) - \
        (resources/(1 + resources)) * np.sum(growth.T * consumption * species, axis = 1) + 10**-8
    
    #dRdt = influx - resources - \
    #    (resources/(1 + resources)) * np.sum(growth.T * consumption * species, axis = 1) + 10**-8
    
    return np.concatenate((dSdt, dRdt))

#%%

def normal_distributed_parameters(mu, sigma, dims):
    
    return mu + sigma*np.random.randn(*dims)

#%%

def closeness_to_competitive_exclusion(no_species, no_resources,
                                       death, influx,
                                       growth_stats, consumption_stats):

    def resource_species_diversity():
        
        growth = normal_distributed_parameters(**growth_stats,
                                               dims = (no_species, no_resources))
        consumption = normal_distributed_parameters(**consumption_stats,
                                                    dims = (no_resources, no_species))
        
        initial_abundances = np.random.uniform(0.1,1,no_species)
        initial_concentrations = np.random.uniform(0.1,1,no_resources)
        
        simulation = solve_ivp(dCR_dt, [0, 3000], 
                               np.concatenate((initial_abundances, initial_concentrations)),
                               args = (no_species, growth, death, consumption, influx),
                               method = 'LSODA')
        
        if np.any(np.log(np.abs(simulation.y[:,-1])) > 6):
            
            return None
        
        else:

            species_diversity = \
                np.count_nonzero(np.any(simulation.y[:no_species,-20:] > 1e-4, axis = 1))
    
            resource_diversity =  \
                np.count_nonzero(np.any(simulation.y[no_species:,-20:] > 1e-4, axis = 1))
    
            return {'Species diversity' : species_diversity,
                    'Resource diversity' : resource_diversity,
                    'Simulation' : simulation}
    
    diversity_data_unclean = [resource_species_diversity() for _ in range(100)]
    diversity_data = list(filter(lambda item: item is not None, diversity_data_unclean))
    
    satisfies_competitive_exclusion = [community for community in diversity_data
                                        if community['Species diversity'] <= community['Resource diversity']]
    
    violates_competitive_exclusion = [community for community in diversity_data
                                        if community['Species diversity'] > community['Resource diversity']]
    
    return satisfies_competitive_exclusion, violates_competitive_exclusion

#%%

no_species = 100
no_resources = 50

death = np.ones(no_species)
influx = np.ones(no_resources)

growth_stats = {'mu' : 1, 'sigma' : 0.3}
consumption_stats = {'mu' : 0.9, 'sigma' : 0.2}

satisfies_competitive_exclusion, violates_competitive_exclusion = \
    closeness_to_competitive_exclusion(no_species, no_resources,
                                       death, influx,
                                       growth_stats, consumption_stats)
    
print(len(violates_competitive_exclusion))

#%%

fig, (ax1, ax2) = plt.subplots(1, 2, sharex = True, layout = 'constrained')

fig.supxlabel('time')

ax1.plot(violates_competitive_exclusion[1]['Simulation'].t,
         violates_competitive_exclusion[1]['Simulation'].y[:no_species,:].T)
ax1.set_ylabel('species abundance')
ax1.set_xlim([1000,2000])

ax2.plot(violates_competitive_exclusion[1]['Simulation'].t,
         violates_competitive_exclusion[1]['Simulation'].y[no_species:,:].T)
ax2.set_ylabel('resource abundance')
ax2.set_xlim([1000,2000])

#%%

fig, (ax1, ax2) = plt.subplots(1, 2, sharex = True, layout = 'constrained')

fig.supxlabel('time')

ax1.plot(satisfies_competitive_exclusion[9]['Simulation'].t,
         satisfies_competitive_exclusion[9]['Simulation'].y[:no_species,:].T)
ax1.set_ylabel('species abundance')
#ax1.set_ylim([0,0.2])

ax2.plot(satisfies_competitive_exclusion[9]['Simulation'].t,
         satisfies_competitive_exclusion[9]['Simulation'].y[no_species:,:].T)
ax2.set_ylabel('resource abundance')
#ax2.set_ylim([0,0.2])

#%%

def fluctuation_coefficient(species_dynamics):
    
    extant_species = species_dynamics[species_dynamics[:,-1] > 1e-4,-100:]
    
    #return np.mean(np.std(extant_species, axis = 1)/np.mean(extant_species, axis = 1))
    
    return np.count_nonzero(np.std(extant_species, axis = 1)/np.mean(extant_species, axis = 1) > 0.5)

f_coeff_v = [fluctuation_coefficient(community['Simulation'].y[no_species:,:])
             for community in violates_competitive_exclusion]

f_coeff_s = [fluctuation_coefficient(community['Simulation'].y[no_species:,:])
             for community in satisfies_competitive_exclusion]

#%%

print('S =', np.mean(f_coeff_s), ', V =', np.mean(f_coeff_v))

plt.hist(f_coeff_s, 20, alpha=0.5, label='S')
plt.hist(f_coeff_v, 20, alpha=0.5, label='V')
plt.legend(loc='upper right')
plt.show()

#%%

bootstrapped_s = [np.mean(np.random.choice(f_coeff_s, len(f_coeff_v)))
                  for _ in range(5000)]

plt.hist(bootstrapped_s, 100, alpha=0.5, label='S')
plt.vlines(np.mean(f_coeff_v), 0, 150, colors = 'black', linestyle = '-',
           label = 'V')
plt.vlines(np.percentile(bootstrapped_s, 97.5), 0, 150, colors = 'grey',
           linestyle = '--', label = '95 CI (S)')
#plt.xlim([0,1])
plt.legend(loc='upper right')
plt.show()

#%%

s_v_r_diversity = [community['Species diversity']/community['Resource diversity']
                   for community_list in [violates_competitive_exclusion,
                                          satisfies_competitive_exclusion]
                       for community in community_list]

plt.scatter(np.concatenate([np.array(f_coeff_v),np.array(f_coeff_s)]),
            s_v_r_diversity)

print(pearsonr(np.concatenate([np.array(f_coeff_v),np.array(f_coeff_s)]),
               s_v_r_diversity))

#%%

s_diversity = [community['Species diversity']
                   for community_list in [violates_competitive_exclusion,
                                          satisfies_competitive_exclusion]
                       for community in community_list]

plt.scatter(np.concatenate([np.array(f_coeff_v),np.array(f_coeff_s)]),
            s_diversity)
print(pearsonr(np.concatenate([np.array(f_coeff_v),np.array(f_coeff_s)]),
               s_diversity))

#%%

r_diversity = [community['Resource diversity']
                   for community_list in [violates_competitive_exclusion,
                                          satisfies_competitive_exclusion]
                       for community in community_list]

plt.scatter(np.concatenate([np.array(f_coeff_v),np.array(f_coeff_s)]),
            r_diversity)
print(pearsonr(np.concatenate([np.array(f_coeff_v),np.array(f_coeff_s)]),
               r_diversity))


#%%

def dCR_dt_noinflux(t,var,
           no_species,
           growth, death, consumption):
    
    #breakpoint()
    
    var[var < 1e-9] = 0
    
    species = var[:no_species]
    resources = var[no_species:]
    
    #dSdt = species * (np.sum(growth * resources, axis = 1) - death) + 10**-8
    
    #dRdt = resources * (influx - resources) - \
    #    resources * np.sum(growth.T * consumption * species, axis = 1) + 10**-8
    
    #dRdt = influx - \
    #    resources * np.sum(growth.T * consumption * species, axis = 1)
    
    dSdt = species * (np.sum(growth * (resources/(1 + resources)), axis = 1) - death) + 10**-8
    
    dRdt = - (resources/(1 + resources)) * np.sum(growth.T * consumption * species, axis = 1) + 10**-8
    
    #dRdt = influx - resources - \
    #    (resources/(1 + resources)) * np.sum(growth.T * consumption * species, axis = 1) + 10**-8
    
    return np.concatenate((dSdt, dRdt))

#%%

no_species = 100
no_resources = 50

death = 0.05*np.ones(no_species)
growth = normal_distributed_parameters(1, 0.3, (no_species, no_resources))
consumption = normal_distributed_parameters(0.9, 0.2, (no_resources, no_species))

initial_abundances = np.random.uniform(0.1,1,no_species)
resource_input = 0.5*np.ones(no_resources)

times = []
abundances = []

for i in range(1000):
    
    simulation = solve_ivp(dCR_dt_noinflux, [0, 25], 
                           np.concatenate((initial_abundances, resource_input)),
                           args = (no_species, growth, death, consumption),
                           method = 'LSODA')
    
    times.append(simulation.t[-1])
    abundances.append(simulation.y[:,-1])
    
    initial_abundances = simulation.y[:no_species,-1]
 
#%%

#combined_times = np.cumsum(np.concatenate(times))
combined_times = np.arange(25,25*1001,25)
combined_abundances = np.column_stack(abundances)

#%%

fig, (ax1, ax2) = plt.subplots(1, 2, sharex = True, layout = 'constrained')

fig.supxlabel('time')

ax1.plot(combined_times, combined_abundances[:no_species,:].T)
ax1.set_ylabel('species abundance')
#ax1.set_ylim([0,1.25])
#ax1.set_xlim([1000,2000])

#ax2.plot(combined_times, combined_abundances[no_species:,:].T)
#ax2.set_ylabel('resource abundance')

