# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:02:21 2024

@author: jamil
"""

# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\Chemically-mediated models

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib as mpl

########################

#%%
class two_species_model():
    
    def __init__(self,growth,consumption,production,death,
                 influx,outflux):
        
        self.growth = growth
        self.death = death
        self.consumption = consumption
        self.production = production
        self.influx = influx
        self.outflux = outflux
        
    def __call__(self,t,var):   
        
        # 2 species, n mediators
        
        S = var[:2]
        R = var[2:]
        
        #dS = S*np.sum(self.growth*R,axis=1)
        dS = S*np.sum(self.growth * self.consumption * R, axis=1) - self.death*S
        
        dR_consumed = R * np.sum(self.consumption.T*S,axis=1)
        
        def production_rate(i):
            
            production_i = self.production[:,i,:]
            
            production_rate = \
                np.sum(np.sum(production_i * self.consumption * R, axis = 1) * S)
            
            return production_rate
            
        dR_produced = np.array([production_rate(i) for i in range(len(R))])
        
        dR = dR_produced - dR_consumed + R*(self.influx - self.outflux*R)
        
        return np.concatenate([dS,dR])
    
#%%

def normally_distributed_parameters(mu, sigma, dims):
    
    return np.round(mu + sigma*np.random.randn(dims), 2)

def uniformally_distributed_parameters(min_val, max_val, dims):
    
    return np.round(np.random.uniform(min_val, max_val, dims), 2)

def bernoulli_distributed_parameters(val, p, dims):
    
    return val * np.random.binomial(1, p, size = dims)

#%%
    
def generate_parameters(production_distribution_function, production_args,
                        consumption_distribution_function, consumption_args):
    
    metabolic_matrix = np.array([[0,0],[1,0]])
    consumption_matrix = np.array([[1,0],[1,1]])
    
    energies = np.array([1,0.7])
    
    def sample_parameters(distribution_function, kwargs):
    
        match distribution_function:
            
            case 'normal':
                
                parameters = np.abs(normally_distributed_parameters(**kwargs))
                
            case 'uniform':
                
                parameters = uniformally_distributed_parameters(**kwargs)
                
            case 'bernoulli':
                
                parameters = bernoulli_distributed_parameters(**kwargs)
                
        return parameters
    
    consumption = np.array([sample_parameters(c_d_func, c_args)
                            for c_d_func, c_args in zip(consumption_distribution_function, consumption_args)]).reshape((2,2)) * consumption_matrix
    
    production = [np.array([sample_parameters(p_d_func, p_args)
                           for p_d_func, p_args in zip(production_distribution_function, production_args)]).reshape((2,2)) * metabolic_matrix,
                  np.zeros((2,2))]
    
    #growth = consumption_matrix * np.array([energies - np.sum((prod.T * energies).T, axis = 0) 
    #                                        for prod in production])
    
    growth = consumption_matrix
    production = np.array(production)
    
    if np.any(growth < 0) or np.any(consumption < 0) or np.any(production < 0):
        
        raise Exception('Parameter values cannot be negative',
                        growth, consumption, production)
    
    return {'growth' : growth, 'consumption' : consumption, 'production' : production}

#%%

def cross_feeding(model,
                  consumption_distribution_function, consumption_args,
                  production_distribution_function, production_args, 
                  **kwargs):
    
    ########## Parameters ###########
    
    growth_consumption_production = [generate_parameters(production_distribution_function, production_args,
                                                          consumption_distribution_function, consumption_args)
                                       for _ in range(1)]
    
    death = np.zeros(2)
    influx = np.zeros(2)
    outflux = np.zeros(2)
    
    ######## Simulate community dynamics ##################
    
    def initialise_and_simulate(model,
                                growth, death, consumption, production, influx, outflux,
                                initial_conditions,
                                **kwargs):
        
        t_end = 40
        
        dSR_dt = model(growth,consumption,production,death,influx,outflux,**kwargs)
        
        return solve_ivp(dSR_dt, [0, t_end], initial_conditions, max_step = 1,
                         method = 'RK45', rtol = 2.5e-14, atol = 2.5e-14, t_eval=np.linspace(0,t_end,t_end))
 
    monoculture_1 = [initialise_and_simulate(model = model, influx = influx, outflux = outflux,
                                             death = death,
                                             initial_conditions = [0.1,0,10,0],
                                             **params) 
                     for params in growth_consumption_production]
    
    monoculture_2 = [initialise_and_simulate(model = model, influx = influx, outflux = outflux,
                                             death = death,
                                             initial_conditions = [0,0.1,10,0],
                                             **params) 
                     for params in growth_consumption_production]
    
    coculture = [initialise_and_simulate(model = model, influx = influx, outflux = outflux,
                                         death = death,
                                         initial_conditions = [0.1,0.1,10,0],
                                         **params) 
                     for params in growth_consumption_production]
    
    simulations = {'Monoculture 1' : monoculture_1,
                   'Monoculture 2': monoculture_2,
                   'Coculture': coculture}
    
    species_interaction_df = species_interactions(simulations, 10)
    
    return {'Simulations' : simulations, 'Dataframe' : species_interaction_df,
            'Parameters' : growth_consumption_production}

#%%

def species_interactions(simulations,time):
    
    monoculture_1_yields = np.array([simulation.y[0,np.abs(simulation.t-time).argmin()] 
                                     for simulation in simulations['Monoculture 1']])
    monoculture_2_yields = np.array([simulation.y[1,np.abs(simulation.t-time).argmin()] 
                                     for simulation in simulations['Monoculture 2']])
    
    coculture_1_yields = np.array([simulation.y[0,np.abs(simulation.t-time).argmin()] 
                                     for simulation in simulations['Coculture']])
    coculture_2_yields = np.array([simulation.y[1,np.abs(simulation.t-time).argmin()] 
                                     for simulation in simulations['Coculture']])
    
    coculture_effect_on_1 = np.log2((coculture_1_yields + 1e-10)/(monoculture_1_yields + 1e-10))
    coculture_effect_on_2 = np.log2((coculture_2_yields + 1e-10)/(monoculture_2_yields + 1e-10))
    
    df = pd.DataFrame([monoculture_1_yields,
                       monoculture_2_yields,
                       coculture_1_yields,coculture_2_yields,
                       coculture_effect_on_1,coculture_effect_on_2],
                      index=['Monoculture yield (1)',
                             'Monoculture yield (2)',
                             'Coculture yield (1)',
                             'Coculture yield (2)',
                             'Coculture effect (1)',
                             'Coculture effect (2)',]).T
    
    return df

#%%

def pairwise_interaction(x, y):
    
    interactions = ['0/0', '0/-', '0/+', '+/+', '-/-', '+/-']
    
    conditions = [(np.abs(x) <= 0.1) & (np.abs(y) <= 0.1),
                  ((np.abs(x) <= 0.1) & (y < -0.1)) | ((np.abs(y) <= 0.1) & (x < -0.1)),
                  ((np.abs(x) <= 0.1) & (y > 0.1)) | ((np.abs(y) <= 0.1) & (x > 0.1)),
                  (x > 0.1) & (y > 0.1),
                  (x < -0.1) & (y < -0.1),
                  ((x < -0.1) & (y > 0.1)) | ((y < -0.1) & (x > 0.1))]

    return np.select(conditions, interactions, np.nan)

#%%

def proportion_interaction_per_bin(data):
    
    unique, counts = np.unique(data, return_counts=True)
    
    return counts/len(data)
    
#%%

n = 25

variable_consumption = np.linspace(0.01,0.5,n)

consumption_distribution_function = ['normal', 'normal', 'normal', 'uniform']
production_distribution_function = ['uniform', 'uniform', 'normal', 'uniform']

production_args = [{'min_val' : 0, 'max_val' : 0, 'dims' : 1},
                   {'min_val' : 0, 'max_val' : 0, 'dims' : 1},
                   {'mu' : 0.8, 'sigma' : 0, 'dims' : 1},
                   {'min_val' : 0, 'max_val' : 0, 'dims' : 1}]

simulations_binned_production_consumption = {str(c_val) : cross_feeding(two_species_model,
                                                                        consumption_distribution_function,
                                                                        [{'mu' : c_val, 'sigma' : 0,  'dims' : 1}, {'mu' : 0, 'sigma' : 0,  'dims' : 1},
                                                                         {'mu' : 0.03, 'sigma' : 0,  'dims' : 1}, {'min_val' : 0.5, 'max_val' : 0.2, 'dims' : 1}],
                                                                        production_distribution_function,
                                                                        production_args)
                                             for c_val in variable_consumption}

#%%

for key in simulations_binned_production_consumption.keys():
    
    simulations_binned_production_consumption[key]['Dataframe']['label'] = float(key)
    
    simulations_binned_production_consumption[key]['Dataframe']['Interaction'] = \
        pairwise_interaction(simulations_binned_production_consumption[key]['Dataframe']['Coculture effect (1)'],
                             simulations_binned_production_consumption[key]['Dataframe']['Coculture effect (2)'])
        
#%%

plot_data = pd.concat([simulations_binned_production_consumption[key]['Dataframe']
                      for key in simulations_binned_production_consumption.keys()])

#%%

plot_data['Monoculture yield (1)'] = np.round(plot_data['Monoculture yield (1)'], 2)

#%%

cmap = mpl.colors.LinearSegmentedColormap.from_list('yb_2',['#3f4bf5ff','#e9a100ff'],N=2)
mpl.colormaps.register(cmap)

#%%

cpal = sns.color_palette("yb_2",n_colors=2)

sns.set_style('white')

cmap_rect_w_key = {'+/-' : '#9a77b766',
                   '-/-' : '#e21a1c66',
                   '0/+' : '#aed3e5ff',
                   '0/-' : '#fc9a9966',
                   '0/0' : '#ccccccff'}

fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(5,4.5))

df = pd.DataFrame([np.tile(plot_data['Monoculture yield (1)'], 2),
                              np.repeat([1,2],plot_data.shape[0]),
                              np.concatenate((plot_data['Coculture effect (1)'],
                                              plot_data['Coculture effect (2)']))],
                             index = ['Monoculture yield (1)', 'Species', 'Coculture effect']).T

df['Monoculture yield (1)'] = np.round(df['Monoculture yield (1)'], 1)

x = 'Monoculture yield (1)'
y = 'Coculture effect'

ax.add_patch(mpl.patches.Rectangle((0,-0.6), 0.6, 3,
                                   edgecolor='none',
                                   facecolor= cmap_rect_w_key['0/0']))
ax.add_patch(mpl.patches.Rectangle((0.6,-0.6), 9, 3,
                                   edgecolor='none',
                                   facecolor= cmap_rect_w_key['+/-']))
ax.add_patch(mpl.patches.Rectangle((9.6,-0.6), 0.6, 3,
                                   edgecolor='none',
                                   facecolor= cmap_rect_w_key['0/+']))

ax.hlines(0,xmin=df[x].min(),xmax=df[x].max(),
          color='gray',linestyle='--')

background_lines = sns.lineplot(data = df, x = x, y = y, hue = 'Species',
                                palette = cpal, marker = 'o', linewidth = 5,
                                markersize = 13, ax = ax, markeredgecolor = 'none',
                                err_style = 'bars')
lines = background_lines.get_lines()
[l.set_color('black') for l in lines]

sns.lineplot(data = df, x = x, y = y, hue = 'Species',
             palette = cpal, marker = 'o', linewidth = 4,
             markersize = 12, ax = ax, markeredgecolor = 'none',
             err_style = 'bars')

ax.set(xlabel=None)
ax.set(ylabel=None)
ax.legend_.remove()

ax.tick_params(axis='both', which='major', labelsize=13)

fig.suptitle('Facilitation emerges in cross-feeding coculture \nwith large differences in monoculture yield',
             fontsize=20, fontweight = 'bold', y = 1.15)
ax.set_xlabel('Monoculture abundance of the\n cross-feeder', fontsize=18, weight='bold', 
              color = '#3f4bf5ff')
fig.supylabel(r'$log_2\left(\frac{\text{Coculture abundance of species } i}{\text{Monoculture abundance of species } i}\right)$',
              fontsize=18)

sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/speciesinteractions.png",
            dpi=300,bbox_inches='tight')

#%%

def initialise_and_simulate(model,
                            growth, death, consumption, production, influx, outflux,
                            initial_conditions,
                            **kwargs):
    
    t_end = 100
    
    dSR_dt = model(growth,consumption,production,death,influx,outflux,**kwargs)
    
    return solve_ivp(dSR_dt, [0, t_end], initial_conditions, max_step = 1,
                     method = 'RK45', rtol = 2.5e-14, atol = 2.5e-14, t_eval=np.linspace(0,t_end,t_end))

#%%

growth = np.array([[1,0],[1,1]])
consumption = np.array([[0.1, 0], [0.05, 0.5]])
production = np.array([np.array([[0,0],[0.8,0]]), np.zeros((2,2))])
death = np.zeros(2)

influx = np.zeros(2)
outflux = np.zeros(2)

monoculture_1 = initialise_and_simulate(model = two_species_model, influx = influx, outflux = outflux,
                                        growth = growth, consumption = consumption, production = production,
                
                                        death = death,
                                        initial_conditions = [0.1,0,5,0]) 

monoculture_2 = initialise_and_simulate(model = two_species_model, influx = influx, outflux = outflux,
                                        growth = growth, consumption = consumption, production = production,
                                        death = death,
                                        initial_conditions = [0,0.1,5,0]) 

coculture = initialise_and_simulate(model = two_species_model, influx = influx, outflux = outflux,
                                        growth = growth, consumption = consumption, production = production,
                                        death = death,
                                        initial_conditions = [0.1,0.1,5,0]) 

species_interactions_1 = np.log2(coculture.y[0,:]/monoculture_1.y[0,:])
species_interactions_2 = np.log2(coculture.y[1,:]/monoculture_2.y[1,:])
species_interactions_time = np.vstack([species_interactions_1,species_interactions_2])

#%%

interaction_label = pairwise_interaction(species_interactions_time[0,:],
                                         species_interactions_time[1,:])

unique, when_interactions_occur = np.unique(interaction_label, return_inverse = True)

#%%

sns.set_style('white')

fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(5,4.5))

cmap = mpl.colors.LinearSegmentedColormap.from_list('yb',['#3f4bf5ff','#e9a100ff'],N=2)

cmap_rect = ['#9a77b766','#e21a1c66','#aed3e5ff','#fc9a9966','#ccccccff']

ax.set_xlim([0,40])
ax.set_ylim([-0.6,2])

for i in range(len(monoculture_1.t)-1):
    
   ax.add_patch(mpl.patches.Rectangle((monoculture_1.t[i],-0.6),
                                      monoculture_1.t[i+1] - monoculture_1.t[i],
                                      2.6,
                                      edgecolor='none',
                                      facecolor= cmap_rect[when_interactions_occur[i]]))

ax.axhline(0, xmin = 0, xmax = 100, linestyle='--', color = 'black')
ax.tick_params(axis='both', which='major', labelsize=13)

for i in range(2):
    
    ax.plot(monoculture_1.t,species_interactions_time[i,:].T,
            color = 'black',linewidth=6)
    ax.plot(monoculture_1.t,species_interactions_time[i,:].T,
            color = cmap(i),linewidth=5)
    
fig.suptitle('Species interactions are time-dependent',
                    fontsize=20, fontweight = 'bold', y = 1.075)
ax.set_xlabel('time', fontsize=18, weight='bold')
fig.supylabel(r'$log_2\left(\frac{\text{Coculture abundance of species } i}{\text{Monoculture abundance of species } i}\right)$',
              fontsize=18)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/interactions_change_over_time.png",
            dpi=300,bbox_inches='tight')

