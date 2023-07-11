# -*- coding: utf-8 -*-
"""
BESS for price arbitrage: schedule BESS for grid exchanges for DA horizon

@author: akylas.stratigakos@mines-paristech.fr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import gurobipy as gp
import pickle

# Add current directory to path
cd = os.path.dirname(__file__)
sys.path.append(cd)

#Import utility functions
from EnsemblePrescriptiveTree import EnsemblePrescriptiveTree
from Utility_functions import *

######################## Forecasting functions

def generate_scenarios(Target, prob_pred, Quantiles, horizon = 24, n_scen = 100, plot = True):
    K = horizon
    q = Quantiles

    #Transform to U
    split = 50*K
    u = pit_eval(Target[:split], prob_pred[:split], quantiles = Quantiles, nbins = 50, plot = plot).reshape(-1,K)
    u[u==1] = 0.99

    #Transform to standard normal/ Estimate multivariate Gaussian Copula
    norm = stats.distributions.norm()
    x_trans = norm.ppf(u)
    cov = np.cov(x_trans.T) #Covariance
    #Generate Scenarios
    
    #Step 1: Draw from multivariate Gausian
    x = np.random.multivariate_normal(np.zeros(x_trans.shape[1]), cov, n_scen).T
    
    #Step 2: Transform to uniform using the inverse probit function 
    u_scen = np.round(norm.cdf(x), 2)
    u_scen[u_scen < q.min()] = q.min()
    u_scen[u_scen> q.max()] = q.max()
    
    #Stp 3: Transform based on predictive densities
    scenarios = np.zeros((len(prob_pred), n_scen))
    
    for i in range(0, len(prob_pred), K):   
        for h in range(K):
            temp_u = u_scen[h,:]
            ind = [np.where(Quantiles == i_scen )[0][0] for i_scen in temp_u]
            scenarios[i+h,:] = prob_pred[i+h, ind]

    t = K
    start = 0
    stop = t
    if plot:
        plt.figure(dpi = 600)
    
        for i in range(10):
            if i==0:
                plt.plot(scenarios[:t, i], 'black', alpha = 0.5, label = 'Scenarios')
            else:
                plt.plot(scenarios[:t, i], 'black', alpha = 0.5)
        plt.plot(scenarios[:t, :5], 'black', alpha = 0.5)
        plt.plot(Target[start:stop], 'o', color = 'red', linewidth = 2, label = 'Actual')        
        plt.fill_between( range(len(Target[start:stop])),  prob_pred[start:stop, np.where(Quantiles == .05)[0][0]],\
                         prob_pred[start:stop, np.where(Quantiles == .95)[0][0]], facecolor='blue', alpha = 0.75, label = '90% PI' )
        plt.fill_between( range(len(Target[start:stop])),  prob_pred[start:stop, np.where(Quantiles == .01)[0][0]], \
                         prob_pred[start:stop, np.where(Quantiles == .99)[0][0]], facecolor='blue', alpha = 0.25, label = '98% PI' )
        plt.legend()
        plt.title('Day-ahead Capacity Prices')
        plt.ylabel('EUR/MWh')
        plt.show()
    
    return scenarios

def forecasting_module(config, trainY, testY, trainX, testX, Actual_Price, plot=False):
    'Forecasting prices, returns predictions (point, prob, scenarios), evaluates results'

    horizon = config['horizon']
    
    nTrees = 100
    reg_model = RandomForestRegressor(n_estimators = nTrees, min_samples_leaf = 5, random_state=0)
    reg_model.fit(trainX, trainY)

    #Retrieve Probabilistic Predictions
    tree_pred =  [reg_model.estimators_[tree].predict(testX) for tree in range(nTrees)]  #Individual tree prediction
    Prob_Pred = np.quantile(np.array(tree_pred).T , config['quant'], axis = 1).T #Find quantile from ensemble predictions

    Point_Pred = reg_model.predict(testX).reshape(-1,1)
    #Evaluate point predictions
    print('Forecast Accuracy')
    print('\t MAPE \t RMSE \t MAE')
    print(eval_point_pred(Point_Pred, Actual_Price, digits=2))
    
    #Reliability plot (probabilistic predictions) 
    h = 1  #Starting point
    step = horizon   #Hourly predictions
    #reliability_plot(testY[h::step].values, Prob_Pred[h::step], config['quant'], boot = 100, label = None)
    
    # Scenario Generation
    Scenarios = generate_scenarios(Actual_Price.values, Prob_Pred, config['quant'].round(2),
                                   horizon = horizon, n_scen = config['n_scen'], plot = plot)
    if plot:
        # Plot forecasts
        start = 24*12
        stop = start+24
        for i in range(10):
            if i==0:
                plt.plot(Scenarios[start:stop, i], 'black', alpha = 0.5, label = 'Scenarios')
            else:
                plt.plot(Scenarios[start:stop, i], 'black', alpha = 0.5)
        
        plt.plot(Actual_Price.values[start:stop], '-o',color = 'red', linewidth = 2, label = 'Actual')        
        plt.plot(Point_Pred[start:stop], color = 'y', linewidth = 2, label = 'Point Forecast')        
        
        plt.fill_between( range(len(Actual_Price[start:stop])),  Prob_Pred[start:stop, np.where(config['quant'].round(2) == .05)[0][0]],\
                         Prob_Pred[start:stop, np.where(config['quant'].round(2) == .95)[0][0]], facecolor='blue', alpha = 0.5, label = '90% PI' )
        plt.fill_between( range(len(Actual_Price[start:stop])),  Prob_Pred[start:stop, np.where(config['quant'].round(2) == .01)[0][0]], \
                         Prob_Pred[start:stop, np.where(config['quant'].round(2) == .99)[0][0]], facecolor='blue', alpha = 0.25, label = '98% PI' )
        plt.legend()
        plt.xlabel('Hour')
        plt.ylabel('Day-ahead Prices (EUR/MWh)')
        plt.tight_layout()
        plt.savefig(cd+'\\figures\\Price_Forecasts.pdf')
        plt.show()
    
    return Point_Pred, Prob_Pred, Scenarios


################ Optimization Modules
    
def det_opt(config, bess, forecasts):
    'Deterministic Optimization: Returns actions based on point forecasts'
    
    horizon = config['horizon']
    in_eff = bess['in_eff']
    out_eff = bess['out_eff']
    c_in = bess['c_in']
    c_out = bess['c_out']
    z0 = bess['z0']
    B_min = bess['B_min']
    B_max = bess['B_max']
    num_days = int(len(forecasts)/horizon)
    #Output
    Z_state = np.zeros(len(forecasts))
    Z_in = np.zeros(len(forecasts)) 
    Z_out = np.zeros(len(forecasts))
    
    m = gp.Model()
    
    m.setParam('OutputFlag', 0)
    
    # Decision variables
    z_state = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = B_min, ub = B_max, name = 'state')
    z_charge = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'charge')
    z_discharge = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'discharge')
    
    # Auxiliary variables for vector operations
    net_diff = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'net diff')
    deviation = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0)
    t = m.addMVar(1 , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph')
    
    # Technical constraints
    m.addConstr( z_charge <= c_in)  #Charge limits
    m.addConstr( z_discharge <= c_out)  #Discharge limit
    m.addConstr( net_diff == z_discharge - z_charge)   #Net difference
    m.addConstr( deviation == z_state-z0)   #Deviation from starting point
    
    # Transition Function
    m.addConstr( z_state[1:] == z_state[0:-1] + in_eff*z_charge[0:-1] - out_eff*z_discharge[0:-1]) #Transition function
    m.addConstr( z_state[0] == z0)
    m.addConstr( z_state[-1] + in_eff*z_charge[-1] - out_eff*z_discharge[-1] == 0.5)
    
    # Constraints that depend on time-varying parameters
    for day in range(num_days):
        
        #DA price forecasts
        y_hat = forecasts[day*horizon:(day+1)*horizon].reshape(-1)
        # Update objective and solve for DA
        # Epigraph formulation
        a1 = m.addConstr( t >= -y_hat.T@net_diff + bess['gamma']*(deviation@deviation) \
                         + bess['epsilon']*(z_charge@z_charge) + bess['epsilon']*(z_discharge@z_discharge))
        
        m.setObjective( t.sum(), gp.GRB.MINIMIZE)
        m.optimize()    
        # Store results
        Z_state[day*horizon:(day+1)*horizon] = z_state.X
        Z_in[day*horizon:(day+1)*horizon] = z_charge.X
        Z_out[day*horizon:(day+1)*horizon] = z_discharge.X
        # Remove constraints, reset solution and update initial state    
        for constr in [a1]: m.remove(constr)
        m.reset()
    return Z_out, Z_in, Z_state
   
def saa_opt(config, bess, forecasts, trainY):
    'Sample Average Approximation: returns actions that minimize in-sample decision costs'
    horizon = config['horizon']
    in_eff = bess['in_eff']
    out_eff = bess['out_eff']
    c_in = bess['c_in']
    c_out = bess['c_out']
    z0 = bess['z0']
    B_min = bess['B_min']
    B_max = bess['B_max']

    #Training dataset: sample path observations of uncertain parameter y
    sample_paths = trainY.values.reshape(-1,24)
    N = len(sample_paths)
    horizon = sample_paths.shape[1]

    saa_model = gp.Model()
    
    saa_model.setParam('OutputFlag', 0)

    #Problem variables
    z_state = saa_model.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = B_min, ub=B_max, name = 'state')
    z_charge = saa_model.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0, ub = c_in, name = 'charge')
    z_discharge = saa_model.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0, ub = c_out, name = 'discharge')

    # Auxiliary variables for vector operations
    net_diff = saa_model.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'net diff')
    deviation = saa_model.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0)
    t = saa_model.addMVar(1 , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph')
    Profit = saa_model.addMVar(N , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph')

    #Constraints and state transition
    saa_model.addConstr( z_state[1:] == z_state[0:-1] + in_eff*z_charge[0:-1] - out_eff*z_discharge[0:-1])
    saa_model.addConstr( z_state[0] == z0)
    saa_model.addConstr( z_state[-1] + in_eff*z_charge[-1] - out_eff*z_discharge[-1] == z0)
    saa_model.addConstr( net_diff == z_discharge - z_charge)
    saa_model.addConstr( deviation == z_state - z0)

    #!!!!! Not sure about scaling with 1/N
    saa_model.addConstr( Profit == sample_paths@net_diff)

    saa_model.addConstr( t >= - Profit.sum()/N + bess['gamma']*(deviation@deviation) \
                         + bess['epsilon']*(z_charge@z_charge) + bess['epsilon']*(z_discharge@z_discharge))

    saa_model.setObjective( t.sum(), gp.GRB.MINIMIZE)
    saa_model.optimize()

    n_test_days = int(len(forecasts)/horizon)

    saa_out = np.tile(z_discharge.X, n_test_days)
    saa_in = np.tile(z_charge.X, n_test_days)
    saa_state = np.tile(z_state.X, n_test_days)
    
    return saa_out, saa_in, saa_state
     
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

#%% Problem parameters
def bess_params():
    'BESS technical parameters'
    
    bess = {}
    bess['c_in'] = 0.5  #Limit on charge speed
    bess['c_out'] = 0.2  #Limit on discharge speed
    bess['B_max'] = 1  #SOC max
    bess['B_min'] = 0  #SOC min
    bess['in_eff'] = 0.8  #Charge efficiency
    bess['out_eff'] = 1.1  #Discharge efficiency
    bess['z0'] = 0.5  #Initial SOC
    bess['gamma'] = 0.5   #Quadratic penalty for deviating from initial conditions
    bess['epsilon'] = 0.5   #Regularization parameter for charge/discharge
    
    return bess

def problem_parameters():
    parameters = {} 
    parameters['train_date'] = '2019-01-01'
    parameters['start_date'] = '2018-01-01'
    parameters['quant'] = np.arange(.01, 1, .01)    #For probabilistic forecasts
    parameters['n_scen'] = 200    #Number of scenarios
    parameters['horizon'] = 24 #Optimization/forecast horizon
    parameters['train'] = True # Train forecasting models + prescriptive trees
    parameters['save'] = False # Train forecasting models + prescriptive trees
    
    return parameters

#%% Main Script
    
# Load data
market_df = pd.read_csv(cd+'\\data\\Market_Data_processed.csv', index_col = 0, parse_dates= True)
market_df.index = market_df.index.round('min')

#%%
# Configuration

config = problem_parameters()

# Different starting points to vary the training sample size
start_points = ['2018-06-01', '2018-01-01', '2017-06-08', '2017-01-08']
model_names = ['6m', '1y', '1_5y', '2y']

# Grid of parameters            
gamma = np.array([0.01, 0.05, 0.1, 0.5, 1])
epsilon = np.array([0.01, 0.05, 0.1, 0.5, 1])

_, _, _, _, Actual_Price = create_supervised(market_df, config)
    
if config['train'] == True:
    Sensitivity_Results = []
    Forecast_Results_Total = []
    Task_Obj_Results = []
    bess = bess_params()
    
    # Price forecasting module: Forecasts need to be generated once per sample size
    exp_point_pred = []
    exp_scenarios = []
    for j, date in enumerate(start_points):
        print('Start date: ', date)
        # Set date in configuration
        config['start_date'] = date    
        # Supervised learning sets
        trainY, testY, trainX, testX, Actual_Price = create_supervised(market_df, config)
        
        # Generate price forecasts for all sample sizes, store in list
        Point_Pred, Prob_Pred, Scenarios = forecasting_module(config, trainY, testY, trainX, testX, Actual_Price)
        exp_point_pred.append(Point_Pred) 
        exp_scenarios.append(Scenarios)

    # Loop through parameter combinations
    for temp_gamma, temp_epsilon in zip(gamma, epsilon):
        print('gamma: ', temp_gamma)
        print('epsilo: ', temp_epsilon)
        # Update BESS parameters            
        bess['gamma'] = temp_gamma
        bess['epsilon'] = temp_epsilon
         
        Result_aggr = []
        Forecasts_aggr = []
        task_obj_aggr = []
        for j, date in enumerate(start_points):
        
            config['start_date'] = date
            
            # Supervised learning sets
            trainY, testY, trainX, testX, _ = create_supervised(market_df, config)
        
            Results_Profit = pd.DataFrame(index = testY.index)
            task_obj = pd.DataFrame(index = testY.index)
            Forecasts = pd.DataFrame(data = exp_point_pred[j], columns = ['Expected'], index = testY.index)
    
            # Estimate deterministic, SAA, stochastic solutions
            det_out, det_in, det_state = det_opt(config, bess, exp_point_pred[j].reshape(-1))
            saa_out, saa_in, saa_state = saa_opt(config, bess, exp_point_pred[j], trainY) 
            
            # Evaluate performance
            Results_Profit['Deterministic'] = Actual_Price.values.reshape(-1) * (det_out - det_in)
            Results_Profit['SAA'] = Actual_Price.values.reshape(-1)* (saa_out - saa_in)
        
            #### Prescriptive Trees            
            #Create wide supervised learning set
            new_trainY, new_trainX, new_testX = create_supervised_prescriptive(trainY, testY, trainX, testX, config)
        
            model = EnsemblePrescriptiveTree(n_estimators = 25, Nmin = 10)
            model.fit(new_trainX, new_trainY, k = 0, c_in = bess['c_in'], c_out = bess['c_out'], 
                  B_max = bess['B_max'], B_min = bess['B_min'], gamma = bess['gamma'], epsilon = bess['epsilon'],
                  in_eff = bess['in_eff'], out_eff = bess['out_eff'], z0 = bess['z0'], parallel = True,
                  n_jobs = -1, cpu_time = True)
            
            print('Generating Predictive Prescriptions')
            Prescription = model.predict_constr(new_testX, new_trainX, new_trainY)
    
            cost_oriented_Predictions = model.cost_oriented_forecast(new_testX, new_trainX, new_trainY).reshape(-1)
            Forecasts['PT'] = cost_oriented_Predictions
            Forecasts_aggr.append(Forecasts)
    
            pt_state = np.array([prescr['z_state'] for prescr in Prescription]).reshape(-1)
            pt_out =  np.array([prescr['z_discharge'] for prescr in Prescription]).reshape(-1)
            pt_in =  np.array([prescr['z_charge'] for prescr in Prescription]).reshape(-1)
    
            Results_Profit['PT'] = Actual_Price.values.reshape(-1)*(pt_out - pt_in)
            
            task_obj['Deterministic'] = -Actual_Price.values.reshape(-1) * (det_out - det_in) + bess['gamma']*(det_state-0.5)**2\
            +bess['epsilon']*det_in**2++bess['epsilon']*det_out**2
            task_obj['SAA'] = -Actual_Price.values.reshape(-1) * (saa_out - saa_in) + bess['gamma']*(saa_state-0.5)**2\
                +bess['epsilon']*saa_out**2++bess['epsilon']*saa_in**2
            task_obj['PT'] = -Actual_Price.values.reshape(-1) * (pt_out - pt_in) + bess['gamma']*(pt_state-0.5)**2\
                +bess['epsilon']*pt_in**2++bess['epsilon']*pt_out**2
                        
            Result_aggr.append(Results_Profit)
            task_obj_aggr.append(task_obj)
        
        Sensitivity_Results.append(Result_aggr)
        Forecast_Results_Total.append(Forecasts_aggr)
        Task_Obj_Results.append(task_obj_aggr)
        
        
    pickle.dump(Forecast_Results_Total, open(cd+'\\results\\Forecast_Results_Total', 'wb'))            
    pickle.dump(Sensitivity_Results, open(cd+'\\results\\Sensitivity_Results', 'wb'))
    pickle.dump(Task_Obj_Results, open(cd+'\\results\\Task_Obj_Results', 'wb'))

else:
    # Load previous results
    Sensitivity_Results = pickle.load(open(cd+'\\results\\Sensitivity_Results', 'rb'))
    Forecast_Results_Total = pickle.load(open(cd+'\\results\\Forecast_Results_Total', 'rb'))            
    Task_Obj_Results = pickle.load(open(cd+'\\results\\Task_Obj_Results', 'rb'))            

#%%%%% Results plots
    
sample_length = ['6 months', '1 year', '1.5 years', '2 years']

boxcolor = ['tab:blue', 'tab:green']
naive = np.abs(Actual_Price.values[168:]-Actual_Price.values[:-168]).mean()

plt.figure(figsize=(3.5,2), constrained_layout=True)

for j in range(len(start_points)):    
    ev_metric = []
    cost_or_metric = []
    for i, comb_forecast_results in enumerate(Forecast_Results_Total):
        # Error metric is defined here
        temp_error = comb_forecast_results[j] - Actual_Price.values
        temp_metric = np.abs(temp_error).mean(axis=0)
    
        ev_metric.append(temp_metric['Expected'])
        cost_or_metric.append(temp_metric['PT'])

    ev_line = plt.plot(j, ev_metric[0], 'o', color = 'tab:blue')
    boxplot = plt.boxplot(cost_or_metric,  patch_artist=True, 
            positions = [j])
    plt.setp(boxplot['boxes'], color= 'tab:green')        
naive_line = plt.plot(range(4), 4*[naive], '--', color = 'black')
#plt.grid()
plt.ylabel('Mean Absolute Error (EUR/MWh)')
plt.legend(ev_line+boxplot['boxes'] + naive_line, ['FO-Det', 'PF', 'Seasonal Naive'])
#plt.tight_layout()
plt.xlabel('Sample Size $n$')
plt.xticks(range(4), sample_length)
if config['save']: plt.savefig(cd+'\\figures\\BESS_forecast_MAE.pdf')
plt.show()

#%% Parreto front plot
# dictionary for subplots
ax_lbl = [['6m', '1y', '1.5y', '2y']]
gs_kw = dict(width_ratios=[1, 1], height_ratios=[1, 1])

#fig, ax = plt.subplot_mosaic(ax_lbl, constrained_layout = True, figsize = (3.5, 1.5*2), 
#                             gridspec_kw = gs_kw)

fig, axs = plt.subplots(4,1, figsize=(3.5, 4), constrained_layout=True, sharex = True)

start_points = ['2018-06-01', '2018-01-01', '2017-06-08', '2017-01-08']

boxcolor = ['tab:blue', 'black', 'tab:green']

line_style = ['-', '--', '-.', 'dotted']

ind=[0,1,2,3]

for j, date in enumerate(start_points):

    aggr_prof = pd.concat([r[j].sum() for r in Sensitivity_Results[:-1]])
    aggr_obj = pd.concat([r[j].sum() for r in Task_Obj_Results[:-1]])
    aggr_penalty = aggr_obj+aggr_prof 
    
    for k,m in enumerate(['Deterministic', 'SAA', 'PT']):        
        axs[ind[j]].plot(aggr_penalty[m], aggr_prof[m], linestyle = line_style[j],
                 color = boxcolor[k], linewidth=2)
        
        for g in range(len(gamma[:-1])):
            axs[ind[j]].plot(aggr_penalty[m][g], aggr_prof[m][g], 'd', markersize = g*1.8+2.75, color = boxcolor[k])
            
            
    axs[ind[j]].set_ylim([1200, 2300])    
    axs[ind[j]].set_ylabel('Profit (EUR/MWh)')

plt.xlim([0, 700])    
plt.xlabel(r'$\gamma || z^{soc}-z_0||_2^2+\epsilon || z^{out}||_2^2 +\epsilon || z^{in}||_2^2$')

#Empty lines for second legend
plt.plot([], color='tab:blue', linewidth=2, label='FO-Det')
plt.plot([], color='black', linewidth=2, label='SAA')
plt.plot([], color='tab:green', linewidth=2, label='PF')
#fig.legend(loc = [0.25, 0.883], ncol = 3)

fig.legend( fontsize=6, ncol=3, loc = (1, .8), 
                 bbox_to_anchor=(0.1, -.05))

#Empty lines for second legend
m1=plt.plot([], linestyle=line_style[0], color='gray', label='6m')
m2=plt.plot([], linestyle=line_style[1], color='gray', label='1y')
m3=plt.plot([], linestyle=line_style[2], color='gray', label='1.5y')
m4=plt.plot([], linestyle=line_style[3], color='gray', label='2y')
leg2 = fig.legend(m1+m2+m3+m4, ['6m', '1y', '1.5y', '2y'], fontsize=6, ncol=4, loc = (1, .8), 
                 bbox_to_anchor=(0.1, -.1))
#fig.add_artist(leg2)
#fig.legend( fontsize=6, ncol=4, loc = (1, .8), 
#                 bbox_to_anchor=(0.1, -.2))

if config['save']: plt.savefig(cd+'\\figures\\BESS_pareto.pdf')
plt.show()



#%% Task-loss improvement relative to optimization parameters (NOT INCLUDED IN PAPER)
# Percentage wise improvement against the SAA solution, as function of sample size and parameters gamma, epsilon
markers = ['o', 's', 'p', 'd']
marker_col = ['black', 'tab:orange', 'tab:red', 'tab:purple']
fig,ax=plt.subplots(figsize=(6,3))
for i in range(len(Task_Obj_Results)):
    for j, task_loss in enumerate(Task_Obj_Results[i]):
        plt.scatter(i, 100*(task_loss['PT'].sum()-task_loss['SAA'].sum())/task_loss['SAA'].sum(), marker = markers[j], 
                    color = marker_col[j], linewidths=3)
        
plt.xlabel('Parameters $\{\gamma,\epsilon\}$')
plt.xticks(range(len(gamma)), gamma)
plt.ylabel('Task-loss Improvement (%)')
ax.set_axisbelow(True)
plt.legend(['6m', '1y', '1.5y', '2y'])
plt.grid()
fig.tight_layout()
if config['save']: plt.savefig(cd+'\\figures\\BESS_loss_reg_param.pdf')
fig.show()




