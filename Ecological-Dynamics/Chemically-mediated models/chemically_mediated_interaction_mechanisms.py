# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:54:13 2024

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

cmap = mpl.colors.LinearSegmentedColormap.from_list('yb',['#e9a100ff','#3f4bf5ff'],N=2)
mpl.colormaps.register(cmap)

#%%
class two_species_model():
    
    def __init__(self,growth,consumption,production,
                 influx,outflux):
        
        self.growth = growth
        self.consumption = consumption
        self.production = production
        self.influx = influx
        self.outflux = outflux
        
    def __call__(self,t,var):   
    
        # 2 species, n mediators
         
        S = var[:2]
        R = var[2:]
        
        dS = S*np.sum(self.growth*R,axis=1)
        
        dR_consumed = np.matmul((self.consumption * self.growth * R).T,
                                S.reshape(len(S),1))[:,0]
        
        def production_rate(i):
            
            production_i = self.production[:,i,:]
            production_rate = \
                np.sum(np.sum(production_i * self.growth * self.consumption * R, axis = 1) * S)
            
            return production_rate
            
        dR_produced = np.array([production_rate(i) for i in range(len(R))])
        
        dR = dR_produced - dR_consumed + R*(self.influx - self.outflux*R)
        
        return np.concatenate([dS,dR])
    
#%%
class two_species_model_saturating_effects():
    
    def __init__(self,growth,consumption,production,
                 influx,outflux,K):
        
        self.growth = growth
        self.consumption = consumption
        self.production = production
        self.influx = influx
        self.outflux = outflux
        self.K = K
        
    def __call__(self,t,var):
        
        # 2 species, n mediators
         
        S = var[:2]
        R = var[2:]
        
        dS = S*np.sum(self.growth*(R/(self.K+R)),axis=1)
        
        dR_consumed = \
            np.matmul((self.consumption * self.growth * (R/(self.K+R))).T,
                      S.reshape(len(S),1))[:,0]
        
        def production_rate(i):
            
            production_i = self.production[:,i,:]
            production_rate = \
                np.sum(np.sum(production_i * self.growth * self.consumption * (R/(self.K+R)), axis = 1) * S)
            
            return production_rate
            
        dR_produced = np.array([production_rate(i) for i in range(len(R))])
        
        dR = dR_produced - dR_consumed + R*(self.influx - self.outflux*R)
        
        return np.concatenate([dS,dR])
    
#%%

def simulate_dynamics(model,
                      fixed_growth,variable_growth,consumption,production,
                      influx,outflux,
                      t_end,
                      **kwargs):
        
    n = variable_growth.shape[1]
    
    def initialise_and_simulate(growth,initial_conditions):
        
        dSR_dt = model(growth,consumption,production,influx,outflux,**kwargs)
        
        return solve_ivp(dSR_dt, [0, t_end], initial_conditions, max_step = 0.25)

    monoculture_fixed = initialise_and_simulate(np.array([fixed_growth,variable_growth[:,0]]),
                                                         [0.1,0,10,0])
    
    monoculture_variable = [initialise_and_simulate(np.array([fixed_growth,variable_growth[:,i]]),
                            [0,0.1,10,0]) for i in range(n)]
    
    coculture = [initialise_and_simulate(np.array([fixed_growth,variable_growth[:,i]]),
                [0.1,0.1,10,0]) for i in range(n)]
    
    simulations = {'Monoculture 1' : monoculture_fixed,
                   'Monoculture 2': monoculture_variable,
                   'Coculture': coculture}
    
    return simulations

#%%

def species_interactions(simulations,time):
    
    monoculture_1_yield = simulations['Monoculture 1'].y[0,np.abs(simulations['Monoculture 1'].t-time).argmin()]
    monoculture_2_yields = np.array([simulation.y[1,np.abs(simulation.t-time).argmin()] 
                                     for simulation in simulations['Monoculture 2']])
    
    coculture_1_yields = np.array([simulation.y[0,np.abs(simulation.t-time).argmin()] 
                                     for simulation in simulations['Coculture']])
    coculture_2_yields = np.array([simulation.y[1,np.abs(simulation.t-time).argmin()] 
                                     for simulation in simulations['Coculture']])
    
    coculture_effect_on_1 = np.log2(coculture_1_yields/monoculture_1_yield)
    coculture_effect_on_2 = np.log2(coculture_2_yields/monoculture_2_yields)
    
    df = pd.DataFrame([np.repeat(monoculture_1_yield,len(monoculture_2_yields)),
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

def plot_df(species_interaction_df):
    
    df = pd.DataFrame([np.repeat([1,2],species_interaction_df.shape[0]),
                       np.tile(species_interaction_df['Variable growth rate (2)'],2),
                       np.tile(species_interaction_df['Monoculture yield (2)'],2),
                       np.concatenate((species_interaction_df['Monoculture yield (1)'],
                                       species_interaction_df['Monoculture yield (2)'])),
                       np.concatenate((species_interaction_df['Coculture yield (1)'],
                                       species_interaction_df['Coculture yield (2)'])),
                       np.concatenate((species_interaction_df['Coculture effect (1)'],
                                       species_interaction_df['Coculture effect (2)']))],
                      index=['Species','Variable growth rate (2)','Variable monoculture yield',
                             'Monoculture yield','Coculture yield','Coculture effect']).T
    
    return df

#%%

def cross_feeding():
    
    ########## Parameters ###########
    
    # Growth rates
    fixed_growth = np.array([0.01,0.1])
    n = 25
    variable_growth = np.vstack([np.linspace(0,0.1,n),np.zeros(n)])
    
    # Consumption rate
    consumption = np.array([[1,1],[1,0]])
    
    # Production rate
    metabolic_matrix = np.array([[0,0],[1,0]])
    production = np.array([np.zeros((2,2)),0.8*metabolic_matrix])
    
    # Influx and outflux
    influx = np.zeros(2)
    outflux = np.zeros(2)
    
    ######## Simulate community dynamics ##################
    
    # Non-resource-saturating effects
    
    yields = simulate_dynamics(two_species_model,
                               fixed_growth,variable_growth,consumption,production,
                               influx,outflux,
                               30)

    species_interaction_df10 = species_interactions(yields, 10)
    species_interaction_df30 = species_interactions(yields, 30)

    species_interaction_df10['Variable growth rate (2)'] = variable_growth[0,:]
    species_interaction_df30['Variable growth rate (2)'] = variable_growth[0,:]
    
    return species_interaction_df10, species_interaction_df30

#%%

def cross_feeding_saturation():

    ########## Parameters ###########
    
    # Growth rates
    fixed_growth = np.array([0.01,0.2])
    n = 25
    variable_growth = np.vstack([np.linspace(0,0.2,n),np.zeros(n)])
    
    # Consumption rate
    consumption = np.array([[1,1],[1,0]])
    
    # Production rate
    metabolic_matrix = np.array([[0,0],[1,0]])
    production = np.array([np.zeros((2,2)),0.8*metabolic_matrix])
    
    # Influx and outflux
    influx = np.zeros(2)
    outflux = np.zeros(2)
    
    ######## Simulate community dynamics ##################
    
    # Non-resource-saturating effects
    
    yields = simulate_dynamics(two_species_model_saturating_effects,
                               fixed_growth,variable_growth,consumption,production,
                               influx,outflux,
                               30,
                               K=1)

    species_interaction_df10 = species_interactions(yields, 10)
    species_interaction_df30 = species_interactions(yields, 30)

    species_interaction_df10['Variable growth rate (2)'] = variable_growth[0,:]
    species_interaction_df30['Variable growth rate (2)'] = variable_growth[0,:]
    
    return species_interaction_df10, species_interaction_df30

#%%

def cross_feeding_with_inhibition():
    
    ########## Parameters ###########
    
    # Growth rates
    fixed_growth = np.array([0.2,-0.05])
    n = 25
    variable_growth = np.vstack([np.linspace(0,0.2,n),np.ones(n)])
    
    # Consumption rate
    consumption = np.array([[1,0],[1,1]])
    
    # Production rate
    metabolic_matrix = np.array([[0,0],[1,0]])
    production = np.array([0.8*metabolic_matrix,np.zeros((2,2))])
    
    # Influx and outflux
    influx = np.zeros(2)
    outflux = np.zeros(2)
    
    ######## Simulate community dynamics ##################
    
    # Non-resource-saturating effects
    
    yields = simulate_dynamics(two_species_model,
                               fixed_growth,variable_growth,consumption,production,
                               influx,outflux,
                               30)

    species_interaction_df10 = species_interactions(yields, 10)
    species_interaction_df30 = species_interactions(yields, 30)

    species_interaction_df10['Variable growth rate (2)'] = variable_growth[0,:]
    species_interaction_df30['Variable growth rate (2)'] = variable_growth[0,:]
    
    return species_interaction_df10, species_interaction_df30

#%%

def cross_feeding_with_inhibition_saturation():
    
    ########## Parameters ###########
    
    # Growth rates
    fixed_growth = np.array([0.2,-0.05])
    n = 25
    variable_growth = np.vstack([np.linspace(0,0.2,n),np.ones(n)])
    
    # Consumption rate
    consumption = np.array([[1,0],[1,1]])
    
    # Production rate
    metabolic_matrix = np.array([[0,0],[1,0]])
    production = np.array([0.8*metabolic_matrix,np.zeros((2,2))])
    
    # Influx and outflux
    influx = np.zeros(2)
    outflux = np.zeros(2)
    
    ######## Simulate community dynamics ##################
    
    # Non-resource-saturating effects
    
    yields = simulate_dynamics(two_species_model_saturating_effects,
                               fixed_growth,variable_growth,consumption,production,
                               influx,outflux,
                               30,K=1)

    species_interaction_df10 = species_interactions(yields, 10)
    species_interaction_df30 = species_interactions(yields, 30)

    species_interaction_df10['Variable growth rate (2)'] = variable_growth[0,:]
    species_interaction_df30['Variable growth rate (2)'] = variable_growth[0,:]
    
    return species_interaction_df10, species_interaction_df30

#%%

############# Plotting ##################

def main():
    
    crossfeeding10, crossfeeding30 = cross_feeding()
    crossfeeding_saturation10, crossfeeding_saturation30 = cross_feeding_saturation()
    inhibition10, inhibition30 = cross_feeding_with_inhibition()
    inhibition_saturation10,inhibition_saturation30 = cross_feeding_with_inhibition_saturation() 
    
    # Non-saturating
    
    plot_crossfeeding10 = plot_df(crossfeeding10)
    plot_crossfeeding30 = plot_df(crossfeeding30)
    plot_inhibition10 = plot_df(inhibition10)
    plot_inhibition30 = plot_df(inhibition30)
    
    plot_inhibition10['Species'] = np.repeat([2,1],inhibition10.shape[0])
    plot_inhibition30['Species'] = np.repeat([2,1],inhibition30.shape[0])
    
    # Saturating
    
    plot_crossfeeding_sat10 = plot_df(crossfeeding_saturation10)
    plot_crossfeeding_sat30 = plot_df(crossfeeding_saturation30)
    plot_inhibition_sat10 = plot_df(inhibition_saturation10)
    plot_inhibition_sat30 = plot_df(inhibition_saturation30)
    
    plot_inhibition_sat10['Species'] = np.repeat([2,1],inhibition_saturation10.shape[0])
    plot_inhibition_sat30['Species'] = np.repeat([2,1],inhibition_saturation30.shape[0])
    
    ############ Plot  ###############
    
    x = 'Variable monoculture yield'
    y = 'Coculture effect'
    
    cpal = sns.color_palette("yb",n_colors=2)
    
    # Non-saturating
    
    sns.set_style('white')
    
    fig = plt.figure(layout='constrained', figsize=(7, 6.25))
    subfigs = fig.subfigures(2, 1, wspace=0.1)
    
    axes_top = subfigs[0].subplots(1, 2, sharex=True, sharey=True)

    for df, ax in zip([plot_crossfeeding10, plot_crossfeeding30],
                      axes_top.flatten()):
        
        ax.hlines(0,xmin=df[x].min(),xmax=df[x].max(),
                  color='gray',linestyle='--')

        sns.lineplot(data = df, x = x, y = y, hue = 'Species',
                     palette = cpal, marker = 'o', linewidth = 2.5,
                     markersize = 8, ax = ax)
        
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.legend_.remove()
    
    subfigs[0].suptitle('Non-reciprocal cross-feeding drives commensalist and parasitic \n species interactions.',
                        fontsize=14, fontweight = 'bold')
    subfigs[0].supxlabel('Monoculture yield of species 1', fontsize=14)
    axes_top[0].set_title('time = 10', fontsize=13)
    axes_top[1].set_title('time = 30', fontsize=13)
    
    ####
    
    axes_bottom = subfigs[1].subplots(1, 2, sharex=True, sharey=True)

    for df, ax in zip([plot_inhibition10, plot_inhibition30],
                      axes_bottom.flatten()):
        
        ax.hlines(0,xmin=df[x].min(),xmax=df[x].max(),
                  color='gray',linestyle='--')

        sns.lineplot(data = df, x = x, y = y, hue = 'Species',
                     palette = cpal, marker = 'o', linewidth = 2.5,
                     markersize = 8, ax = ax)
        
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.legend_.remove()
        
    subfigs[1].suptitle('Cross-feeding on an inhibiting waste product drives \n mutualistic and commensalist species interactions.',
                        fontsize=14, fontweight='bold')
    subfigs[1].supxlabel('Monoculture yield of species 2', fontsize=14)
    axes_bottom[0].set_title('time = 10', fontsize=13)
    axes_bottom[1].set_title('time = 30', fontsize=13)
    
    fig.supylabel(r'$log_2(\frac{\text{coculture yield}}{\text{monoculture yield}})$',multialignment='center',
                  fontsize=18)
    
    sns.despine()
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/speciesinteractions.png",
                dpi=300,bbox_inches='tight')
    
    ####################
    
    # Saturating 
    
    sns.set_style('white')
    
    fig = plt.figure(layout='constrained', figsize=(7, 6.25))
    subfigs = fig.subfigures(2, 1, wspace=0.1)
    
    axes_top = subfigs[0].subplots(1, 2, sharey=True)

    for df, ax in zip([plot_crossfeeding_sat10, plot_crossfeeding_sat30],
                      axes_top.flatten()):
        
        ax.hlines(0,xmin=df[x].min(),xmax=df[x].max(),
                  color='gray',linestyle='--')

        sns.lineplot(data = df, x = x, y = y, hue = 'Species',
                     palette = cpal, marker = 'o', linewidth = 2.5,
                     markersize = 8, ax = ax)
        
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.legend_.remove()
    
    subfigs[0].suptitle('Non-reciprocal cross-feeding drives commensalist and parasitic \n species interactions.',
                        fontsize=14, fontweight = 'bold')
    subfigs[0].supxlabel('Monoculture yield of species 1', fontsize=14)
    axes_top[0].set_title('time = 10', fontsize=13)
    axes_top[1].set_title('time = 30', fontsize=13)
    
    ####
    
    axes_bottom = subfigs[1].subplots(1, 2, sharey=True)

    for df, ax in zip([plot_inhibition_sat10, plot_inhibition_sat30],
                      axes_bottom.flatten()):
        
        ax.hlines(0,xmin=df[x].min(),xmax=df[x].max(),
                  color='gray',linestyle='--')

        sns.lineplot(data = df, x = x, y = y, hue = 'Species',
                     palette = cpal, marker = 'o', linewidth = 2.5,
                     markersize = 8, ax = ax)
        
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.legend_.remove()
        
    subfigs[1].suptitle('Cross-feeding on an inhibiting waste product drives \n mutualistic and commensalist species interactions.',
                        fontsize=14, fontweight='bold')
    subfigs[1].supxlabel('Monoculture yield of species 2', fontsize=14)
    axes_bottom[0].set_title('time = 10', fontsize=13)
    axes_bottom[1].set_title('time = 30', fontsize=13)
    
    fig.supylabel(r'$log_2(\frac{\text{coculture yield}}{\text{monoculture yield}})$',multialignment='center',
                  fontsize=18)
    
    sns.despine()
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/speciesinteractions_saturating.png",
                dpi=300,bbox_inches='tight')
    
    #######################################

if __name__ == '__main__':
    
    main()
    