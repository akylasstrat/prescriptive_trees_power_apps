# -*- coding: utf-8 -*-
"""
Results, graphs

@author: a.stratigakos
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import pickle
from sklearn.ensemble import RandomForestRegressor
from scipy import interpolate, stats
import cvxpy as cp
import matplotlib.patches as patches

# Add current path to directory
cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

from EnsemblePrescriptiveTree import EnsemblePrescriptiveTree
from forecast_utility_functions import *
from optimization_utility_functions import *


plt.rcParams['figure.dpi'] = 600

def evaluate_realized_costs(solutions, Node_demand_actual, col_names, grid, config, plot = True):
    'Function that takes as input the obtained decisions and returns dataframe with realize costs'
    da_costs = pd.DataFrame() 
    rt_costs = pd.DataFrame()
    total_costs = pd.DataFrame()
    
    horizon = config['horizon']
    stats_out = pd.DataFrame(data=np.zeros(( 4,len(col_names) )) , columns = col_names, 
                     index = ['r_up', 'r_down', 'L_shed', 'G_shed'])

    for j, set_solutions in enumerate(solutions):
        ndays = len(set_solutions)
        oos_total_cost = []
        oos_da_cost = []
        oos_rt_cost = []
        
        print('Evaluation: ', col_names[j])
        for k, i in enumerate(range(ndays)):
            if (i == 238) or (i == 239) or (i == 240): continue
            if (i+1)%25==0:print('Out of sample day ', i+1)
            start = i*horizon
            stop = (i+1)*horizon
            
            da_dispatch = set_solutions[k]
            #DA Variables
            p_G = cp.Variable((grid['n_unit'], horizon))
            R_up = cp.Variable((grid['n_unit'], horizon))
            R_down = cp.Variable((grid['n_unit'], horizon))
            flow_da = cp.Variable((grid['n_lines'], horizon))
            theta_da = cp.Variable((grid['n_nodes'], horizon))
        
            #RT Variables
            r_up = cp.Variable((grid['n_unit'], horizon))
            r_down = cp.Variable((grid['n_unit'], horizon))
            L_shed = cp.Variable((grid['n_loads'],horizon))
            G_shed = cp.Variable((grid['n_unit'],horizon))  #Shedding Supply, in case of extremely low demand
            flow_rt = cp.Variable((grid['n_lines'],horizon))
            theta_rt = cp.Variable((grid['n_nodes'], horizon))
        
            Constraints = []
            
            ###### Fix DA decisions
            Constraints += [p_G == da_dispatch['p'], flow_da == da_dispatch['flow'], theta_da == da_dispatch['theta'], 
                            R_up == da_dispatch['R_up'], R_down == da_dispatch['R_down']]
                
            #####RT constraints
            Constraints += [ r_up <= -p_G + grid['Pmax'].repeat(24,axis=1), r_up <= grid['R_up_max'].repeat(24,axis=1),
                            r_down <= p_G, r_down <= grid['R_down_max'].repeat(24,axis=1),
                            L_shed <= Node_demand_actual[:,start:stop],
                            G_shed <= p_G,
                            r_up >= 0, r_down >= 0, L_shed >= 0, G_shed >= 0]
        
            #RT Network flow
            Constraints += [flow_rt == grid['b_diag']@grid['A']@theta_rt, 
                            flow_rt <= grid['Line_Capacity'].repeat(24,axis=1), 
                            flow_rt >= -grid['Line_Capacity'].repeat(24,axis=1),
                            theta_rt[0,:] == 0]
            
            #!!!!! Node injections (evaluation is not done properly)
            Constraints += [ grid['node_G']@(p_G + r_up-r_down-G_shed) \
                + grid['node_L']@(L_shed-Node_demand_actual[:,start:stop]) == grid['B']@(theta_rt)]

            #Constraints += [ grid['node_G']@(r_up-r_down-G_shed) \
            #                + grid['node_L']@(L_shed-Node_demand_actual[:,start:stop]+Node_demand_expected[:,start:stop]) == \
            #                    grid['B']@(theta_rt-theta_da)]
            
            realized_DA_cost = cp.sum(grid['Cost']@p_G)
            realized_RT_cost = cp.sum( grid['Cost_reg_up']@r_up - grid['Cost_reg_down']@r_down + grid['VOLL']*cp.sum(L_shed,axis=0) \
                                      + grid['gshed']*cp.sum(G_shed,axis=0))
        
            prob = cp.Problem(cp.Minimize( realized_DA_cost + realized_RT_cost ) , Constraints)
            prob.solve( solver = 'GUROBI', verbose = False)
            oos_total_cost.append(prob.objective.value)
            oos_da_cost.append(realized_DA_cost.value)
            oos_rt_cost.append(realized_RT_cost.value)
            
            if prob.objective.value ==  None:
                print('Infeasible or unbound')
        
            
            if (plot==True) and (i%25==0):
                plt.plot(da_dispatch['p'].sum(axis=0), label='Production')
                plt.plot(Node_demand_actual[:,start:stop].sum(axis=0), label='Actual Demand')
                #plt.plot(Node_demand_expected[:,start:stop].sum(axis=0), label='Expected Demand')
                plt.plot(G_shed.value.sum(axis=0), '-o',label='G_shed')
                plt.plot(r_down.value.sum(axis=0), '*', label='Regulation-Down')
                plt.plot(r_up.value.sum(axis=0), 'd',label='Regulation-Up')
                plt.legend()
                plt.show()

            stats_out[col_names[j]][0] = stats_out[col_names[j]][0] + r_up.value.sum()
            stats_out[col_names[j]][1] = stats_out[col_names[j]][1] + r_down.value.sum()
            stats_out[col_names[j]][2] = stats_out[col_names[j]][2] + L_shed.value.sum()
            stats_out[col_names[j]][3] = stats_out[col_names[j]][3] + G_shed.value.sum()
        da_costs[col_names[j]] = np.array(oos_da_cost)
        rt_costs[col_names[j]] = np.array(oos_rt_cost)
        total_costs[col_names[j]] = np.array(oos_total_cost)
        
    print(stats_out)
    return da_costs, rt_costs, total_costs, stats_out

def evaluate_single_day(day, solutions, Node_demand_actual, col_names, grid, config, plot = True):
    '''Function that takes as input the DA dispatch actions, 
        solves the RT market with actual load, returns dataframe with realize costs'''
    horizon = config['horizon']
    
    for j, set_solutions in enumerate(solutions):
        print('Out of sample day ', day)
        start = day*horizon
        stop = (day+1)*horizon
        
        da_dispatch = set_solutions[day]
        #DA Variables
        p_G = cp.Variable((grid['n_unit'], horizon))
        R_up = cp.Variable((grid['n_unit'], horizon))
        R_down = cp.Variable((grid['n_unit'], horizon))
        flow_da = cp.Variable((grid['n_lines'], horizon))
        theta_da = cp.Variable((grid['n_nodes'], horizon))
    
        #RT Variables
        r_up = cp.Variable((grid['n_unit'], horizon))
        r_down = cp.Variable((grid['n_unit'], horizon))
        L_shed = cp.Variable((grid['n_loads'],horizon))
        G_shed = cp.Variable((grid['n_unit'],horizon))  #Shedding Supply, in case of extremely low demand
        flow_rt = cp.Variable((grid['n_lines'],horizon))
        theta_rt = cp.Variable((grid['n_nodes'], horizon))
    
        Constraints = []
        
        ###### Fix DA decisions
        Constraints += [p_G == da_dispatch['p'], flow_da == da_dispatch['flow'], theta_da == da_dispatch['theta'], 
                        R_up == da_dispatch['R_up'], R_down == da_dispatch['R_down']]
            
        #####RT constraints
        Constraints += [ r_up <= -p_G + grid['Pmax'].repeat(24,axis=1), r_up <= grid['R_up_max'].repeat(24,axis=1),
                        r_down <= p_G, r_down <= grid['R_down_max'].repeat(24,axis=1),
                        L_shed <= Node_demand_actual[:,start:stop],
                        G_shed <= p_G,
                        r_up >= 0, r_down >= 0, L_shed >= 0, G_shed >= 0]
    
        #RT Network flow
        Constraints += [flow_rt == grid['b_diag']@grid['A']@theta_rt, 
                        flow_rt <= grid['Line_Capacity'].repeat(24,axis=1), 
                        flow_rt >= -grid['Line_Capacity'].repeat(24,axis=1),
                        theta_rt[0,:] == 0]
        
        #Node injections    
        #!!!!! Node injections (evaluation is not done properly)
        Constraints += [ grid['node_G']@(p_G + r_up-r_down-G_shed) \
                + grid['node_L']@(L_shed-Node_demand_actual[:,start:stop]) == grid['B']@(theta_rt)]

#        Constraints += [ grid['node_G']@(r_up-r_down-G_shed) \
#                        + grid['node_L']@(L_shed-Node_demand_actual[:,start:stop]+Node_demand_expected[:,start:stop]) == \
#                            grid['B']@(theta_rt-theta_da)]
        
        realized_DA_cost = cp.sum(grid['Cost']@p_G)
        realized_RT_cost = cp.sum( grid['Cost_reg_up']@r_up - grid['Cost_reg_down']@r_down + grid['VOLL']*cp.sum(L_shed,axis=0) \
                                  + grid['gshed']*cp.sum(G_shed,axis=0))
    
        prob = cp.Problem(cp.Minimize( realized_DA_cost + realized_RT_cost ) , Constraints)
        prob.solve( solver = 'GUROBI', verbose = False)
        
        if prob.objective.value ==  None:
            print('Infeasible or unbound')    
        if plot==True:
            plt.plot(da_dispatch['p'].sum(axis=0), label='Production')
            plt.plot(Node_demand_actual[:,start:stop].sum(axis=0), label='Actual Demand')
            #plt.plot(Node_demand_expected[:,start:stop].sum(axis=0), label='Expected Demand')
            plt.plot(G_shed.value.sum(axis=0), '-o',label='G_shed')
            plt.plot(r_down.value.sum(axis=0), '*', label='Regulation-Down')
            plt.plot(r_up.value.sum(axis=0), 'd',label='Regulation-Up')
            plt.plot(L_shed.value.sum(axis=0), 's',label='L-shed')
            plt.title(col_names[j])
            plt.legend()
            plt.show()
            print(col_names[j]+' RT Cost: ', realized_RT_cost.value)
    return

#%% Problem parameters
def problem_parameters():
    parameters = {} 
    # Script parameters
    parameters['train'] = False # Trains models (forecasting and optimization), else loads results
    parameters['save_train'] = False # Save trained models (for trees etc.)
    parameters['save_results'] = False # Save DA dispatch decisions and results

    # Optimization Parameters
    parameters['n_scen'] = 200    #Number of scenarios
    parameters['horizon'] = 24 #Optimization horizon (DO NOT CHANGE)
    parameters['peak_load'] = 2700  #Peak hourly demand
    parameters['wind_capacity'] = 200  #(not used)
    
    # Forecasting parameters (only for the forecasting_module.py)
    parameters['split'] = 0.75 #split percentage
    parameters['quant'] = np.arange(.01, 1, .01)    #For probabilistic forecasts
    # Starting dates create training samples of size 6months, 1y, 1.5y and 2y
    #parameters['start_date'] = '2010-06-01' # Controls for sample size
    #parameters['start_date'] = '2010-01-01'
    #parameters['start_date'] = '2009-06-01' 
    #arameters['start_date'] = '2009-01-01' 
    
    parameters['split_date'] = '2011-01-01' # Validation split
        
    return parameters

#%% Import data, create supervised learning set for prescriptive trees

config = problem_parameters()

# Load IEEE data
grid = load_ieee24(cd+'\\data\\IEEE24Wind_Data.xlsx')

results_folder = cd+'\\results\\aggregated_results\\'

results_dir = [sub[0] for sub in os.walk(results_folder)][1:]

#%% Load results for all sample sizes

da_costs = []
rt_costs = []
total_costs = []
rt_actions = []
Prescription = []
Det_solutions = []
Stoch_solutions = []
cost_oriented_Pred = []
expected_load = []
for i, directory in enumerate(results_dir):
    print(directory)
    # Load solutions
    Prescription.append(pickle.load(open(directory+'\\Predictive_Prescriptions.pickle', 'rb')))
    Det_solutions.append(pickle.load(open(directory+'\\Deterministic_DA_decisions.pickle', 'rb')))
    Stoch_solutions.append(pickle.load(open(directory+'\\Stochastic_DA_decisions.pickle', 'rb')))
    cost_oriented_Pred.append(pickle.load(open(directory+'\\Cost_Oriented_Pred.pickle', 'rb')))
    expected_load.append(pd.read_csv(directory+'\\load_scenarios.csv', index_col=0)['Expected'].values)
    #results = pd.read_excel(cd+'\\EconomicDispatch_Results.xlsx', index_col = 0)
    # Actual Demand per node
    if i==2:              
        load_forecast = pd.read_csv(directory+'\\load_scenarios.csv', index_col=0)
        Node_demand_actual = np.outer(grid['node_demand_percentage'], load_forecast['Target'].values*config['peak_load'])

    
#%%
# Out-of-sample evaluation of prescriptive performance (solves the redispatch optimization problem for realized uncertainty)
eval_results = False
if eval_results:
    for i, directory in enumerate(results_dir):
        # Solve redispatch for each day, for each method considered
        temp_da_costs, temp_rt_costs, temp_total_costs, temp_rt_actions =  evaluate_realized_costs([Prescription[i], Det_solutions[i], Stoch_solutions[i]], Node_demand_actual, 
                                               ['Prescriptive Trees', 'Deterministic', 'Stochastic'], 
                                               grid, config, plot = False)
        da_costs.append(temp_da_costs)
        rt_costs.append(temp_rt_costs)
        total_costs.append(temp_total_costs)
        rt_actions.append(temp_rt_actions)
    # Save aggregated results
    pickle.dump(da_costs, open(results_folder+'\\aggr_da_costs.pickle', 'wb'))
    pickle.dump(rt_costs, open(results_folder+'\\aggr_rt_costs.pickle', 'wb'))
    pickle.dump(total_costs, open(results_folder+'\\aggr_total_costs.pickle', 'wb'))
    pickle.dump(rt_actions, open(results_folder+'\\aggr_rt_actions.pickle', 'wb'))

else:    
    da_costs = pickle.load(open(results_folder+'\\aggr_da_costs.pickle', 'rb'))
    rt_costs = pickle.load(open(results_folder+'\\aggr_rt_costs.pickle', 'rb'))
    total_costs = pickle.load(open(results_folder+'\\aggr_total_costs.pickle', 'rb'))
    rt_actions = pickle.load(open(results_folder+'\\aggr_rt_actions.pickle', 'rb'))


models = ['FO-Det', 'FO-Stoch', 'PF-Stoch']
colors = ['tab:blue', 'tab:brown', 'tab:green']
ticks = ['6 months', '1 year', '1.5 years', '2 years'][:-1]
col_order = ['Deterministic', 'Stochastic', 'Prescriptive Trees']


#%% Forecast accuracy plot
plt.figure(figsize=(6,3))
plt.plot([config['peak_load']*np.abs(pred-load_forecast['Target'].values).mean() for pred in cost_oriented_Pred], '-o', color='tab:green')
plt.plot([config['peak_load']*np.abs(pred-load_forecast['Target'].values).mean() for pred in expected_load], '-d', color='tab:blue')
plt.plot([config['peak_load']*np.abs(load_forecast['Target'].values[168:]-load_forecast['Target'].values[:-168]).mean() for pred in expected_load][::-1], '--', color='black')
plt.legend(['PF-Stoch', 'FO-Det', 'Seasonal Naive'])
plt.xticks(range(len(da_costs)), ticks)
plt.ylabel('Mean Absolute Error (MW)')
plt.xlabel('Sample size $n$')
plt.title('Forecast Accuracy')
plt.tight_layout()
plt.show()

#%% Expected Cost-Aggregated barplot
fig, ax = plt.subplots(constrained_layout=True)
for i in range(len(total_costs)):
    x_pos = [i-0.25, i, i+0.25]
    
    plt.bar(x_pos[0], da_costs[i]['Deterministic'].mean()/10e2, color = 'tab:blue', alpha = 0.5, width=0.2)
    plt.bar(x_pos[1], da_costs[i]['Stochastic'].mean()/10e2, color = 'tab:brown', alpha = 0.5, width=0.2)
    plt.bar(x_pos[2], da_costs[i]['Prescriptive Trees'].mean()/10e2, color = 'tab:green', alpha = 0.5, width=0.2)
        
    bar1=plt.bar(x_pos[0], rt_costs[i]['Deterministic'].mean()/10e2, 
            bottom =  da_costs[i]['Deterministic'].mean()/10e2, color = 'tab:blue', alpha = 1, width=0.2, yerr = rt_costs[i]['Deterministic'].std()/10e2)
    bar2=plt.bar(x_pos[1], rt_costs[i]['Stochastic'].mean()/10e2, 
            bottom =  da_costs[i]['Stochastic'].mean()/10e2, color = 'tab:brown', alpha = 1, width=0.2, yerr = rt_costs[i]['Stochastic'].std()/10e2)
    bar3=plt.bar(x_pos[2], rt_costs[i]['Prescriptive Trees'].mean()/10e2, 
            bottom =  da_costs[i]['Prescriptive Trees'].mean()/10e2, color = 'tab:green', alpha = 1, width=0.2, yerr = rt_costs[i]['Prescriptive Trees'].std()/10e2)
    
plt.legend(bar1+bar2+bar3, ['FO-Det', 'FO-Stoch', 'PF-Stoch'], loc = [0.3, 0.9], ncol=3)
plt.xticks(range(len(da_costs)), ticks)
plt.ylabel(r'Expected Cost ($10^3$\$)')
plt.xlabel('Sample Size $n$')
n1 = plt.bar(.5, 0, color = 'grey', alpha = 0.5, width=0.2, label='DA Costs')
n2 = plt.bar(.5, 0, color = 'grey', alpha = 1, width=0.2, label='RT Costs')
l1 = fig.legend(n1+n2, ['DA Costs', 'RT Costs'], ncol=2, loc = [0.48, 0.8])
fig.add_artist(l1)
ax.set_axisbelow(True)
plt.savefig(cd+'\\figures\\Cost_Results.pdf')
plt.show()    

#%%%%%%%%%%%% Examine single day for further insight
# Select day
day = 59
sample = -1
evaluate_single_day(day, [Prescription[sample], Det_solutions[sample], Stoch_solutions[sample]], Node_demand_actual, 
                    ['Prescriptive Trees', 'Deterministic', 'Stochastic'], grid, config, plot = True)

# Rank all generators from cheapest to most expensive
cost_ind = np.argsort(grid['Cost'])
c = 'tab:red'
num_flex_gen = sum(grid['R_up_max']>0)[0]
flex_mask = (grid['R_up_max'][cost_ind]>0).reshape(-1)

#%%
# Select specific hour
hour = [7, 15, 23]

# Loop through generators (consider only flexible generators)
color_grad = 1/num_flex_gen

fig,ax=plt.subplots(figsize=(6,3), constrained_layout=True)
ax.plot(config['peak_load']*load_forecast['Target'][(day)*24:(day+1)*24].values, color='black', linewidth=2, label='Actual')
ax.plot(config['peak_load']*load_forecast['Expected'][(day)*24:(day+1)*24].values, '-o', color='tab:blue', linewidth=2, label='FO-Det/Stoch')
ax.plot(config['peak_load']*cost_oriented_Pred[-1][(day)*24:(day+1)*24], '-d', color='tab:green', linewidth=2, label='PF-Stoch')
ax.set_xlabel('Hour')
ax.set_ylabel('Total Scheduled Production (MW)')

# Create a Rectangle patch
for h in hour:
    print(h)
    lower = config['peak_load']*load_forecast['Target'][(day)*24:(day+1)*24].values[h] - 200
    rect = patches.Rectangle((h-.5, lower), 1, 400, linewidth=2, edgecolor='red', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

plt.legend()
plt.savefig(cd+'\\figures\\Aggregated_Schedule.pdf')
plt.show()

fig,ax=plt.subplots(figsize=(6,3), constrained_layout=True)
for j, h in enumerate(hour):
    for i, gen in enumerate(cost_ind[flex_mask]):
    
        print('Unit: ', gen)
        print('Dispatch level per method')
        print('PT:', Prescription[sample][day]['p'][gen,h])
        print('Det:', Det_solutions[sample][day]['p'][gen,h])
        print('Stoch:', Stoch_solutions[sample][day]['p'][gen,h])
        print('\n')
        
        b1=plt.bar(j*3-.7, Det_solutions[sample][day]['p'][gen,h], bottom = Det_solutions[sample][day]['p'][cost_ind[flex_mask][:i],h].sum(),
                color = 'tab:blue', alpha = color_grad*(i+1), edgecolor = 'white', width=0.6)
        b2=plt.bar(j*3, Stoch_solutions[sample][day]['p'][gen,h], bottom = Stoch_solutions[sample][day]['p'][cost_ind[flex_mask][:i],h].sum(), 
                color = 'tab:brown', alpha = color_grad*(i+1),  edgecolor = 'white', width=0.6)
        b3=plt.bar(j*3+.7, Prescription[sample][day]['p'][gen,h], bottom = Prescription[sample][day]['p'][cost_ind[flex_mask][:i],h].sum(),
                color = 'tab:green', alpha = color_grad*(i+1), edgecolor = 'white', width=0.6)
plt.legend(b1+b2+b3, ['FO-Det', 'FO-Stoch', 'PF-Stoch'])
plt.ylim([0, 800])    
plt.xticks(np.arange(0,(j+1)*3,3), [str(h)+':00' for h in hour])
plt.xlabel('Hour')
plt.ylabel('Flexible Gen. Schedule (MW)')
plt.savefig(cd+'\\figures\\Hourly_Schedule.pdf')
plt.show()