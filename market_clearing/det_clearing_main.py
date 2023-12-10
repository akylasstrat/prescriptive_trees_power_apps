# -*- coding: utf-8 -*-
"""
Value-oriented market clearing/ deterministic

Uncertainty only on demand/ no wind in this example.

@author: a.stratigakos@imperial.ac.uk
"""

from matpowercaseframes import CaseFrames
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import pickle
import cvxpy as cp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from scipy import interpolate, stats
import gurobipy as gp
import time
from utility_functions import *

# Add current directory to path
cd = os.path.dirname(__file__)  
sys.path.append(cd)

from EnsemblePrescriptiveTree import EnsemblePrescriptiveTree

from forecast_utility_functions import *
from optimization_utility_functions import *

plt.rcParams['figure.dpi'] = 600

def create_supervised(df, config, target_var, predictor_var):
    'Import data, create supervised learning set'
    #n_days = len(df)/config['horizon']
    #train_split = int(n_days*config['split'])*24
    
    trainY = df[target_var][config['start_date']:config['split_date']].to_frame()
    testY = df[target_var][config['split_date']:].to_frame()
    
    trainX = df[predictor_var][config['start_date']:config['split_date']]
    testX = df[predictor_var][config['split_date']:]

    target = df[target_var][config['split_date']:].to_frame()
    
    return trainY, testY, trainX, testX, target



######### Forecasting functions
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
    #cov[cov<0] = 0
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
    if plot==True:
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

def forecasting_module(config, load_df, plot = True, title='Load'):
    ''' Function to generate demand predictions. Returns data frame with point forecasts and scenarios (from prob forecasts)'''

    horizon = config['horizon']
    trainY, testY, trainX, testX, target = create_supervised(load_df, config, 'sc_LOAD', ['Temperature', 'Day', 'Hour', 'Month'])

    nTrees = 200
    reg_model = RandomForestRegressor(n_estimators = nTrees, min_samples_leaf = 2, random_state=0)
    reg_model.fit(trainX, trainY)

    #Retrieve Probabilistic Predictions
    tree_pred =  [reg_model.estimators_[tree].predict(testX).reshape(-1) for tree in range(nTrees)]  #Individual tree prediction
    Prob_Pred = np.quantile(np.array(tree_pred).T , config['quant'], axis = 1).T #Find quantile from ensemble predictions

    Point_Pred = reg_model.predict(testX).reshape(-1,1)
    #Evaluate point predictions
    print('Forecast Accuracy\nMAPE\t RMSE\t MAE\n', eval_point_pred(Point_Pred, target.values, digits=2))

    #Reliability plot (probabilistic predictions) 
    h = 1  #Starting point
    step = horizon   #Hourly predictions
    reliability_plot(testY[h::step].values, Prob_Pred[h::step], config['quant'], boot = 100, label = None, plot = plot)
    
    # Scenario Generation
    Scenarios = generate_scenarios(target.values, Prob_Pred, config['quant'].round(2),
                                   horizon = horizon, n_scen = config['n_scen'], plot = False)

    # Plot forecasts
    if plot:
        start = 24*12
        stop = start+24
        for i in range(10):
            if i==0:
                plt.plot(config['peak_load']*Scenarios[start:stop, i], 'black', alpha = 0.5, label = 'Scenarios')
            else:
                plt.plot(config['peak_load']*Scenarios[start:stop, i], 'black', alpha = 0.5)
        
        plt.plot(config['peak_load']*target.values[start:stop], '-o',color = 'red', linewidth = 2, label = 'Actual')        
        plt.plot(config['peak_load']*Point_Pred[start:stop], '-o', color = 'y', linewidth = 2, label = 'Point Forecast')        
        
        plt.fill_between( range(len(target[start:stop])),  config['peak_load']*Prob_Pred[start:stop, np.where(config['quant'].round(2) == .05)[0][0]],\
                         config['peak_load']*Prob_Pred[start:stop, np.where(config['quant'].round(2) == .95)[0][0]], facecolor='blue', alpha = 0.5, label = '90% PI' )
        plt.fill_between( range(len(target[start:stop])),  config['peak_load']*Prob_Pred[start:stop, np.where(config['quant'].round(2) == .01)[0][0]], \
                         config['peak_load']*Prob_Pred[start:stop, np.where(config['quant'].round(2) == .99)[0][0]], facecolor='blue', alpha = 0.25, label = '98% PI' )
        plt.legend()
        plt.xlabel('Hour')
        plt.ylabel('Load (MW)')
        plt.tight_layout()
        plt.savefig(cd+'\\figures\\Load_Scenario_Plot.pdf')
        plt.show()
        
    load_forecast_df = pd.DataFrame(np.column_stack((target.values, Point_Pred, Scenarios)), columns = ['Target', 'Expected']+['Scen_'+str(n) for n in range(config['n_scen'])] )
    return load_forecast_df

def deterministic_opt(grid, config, Node_demand_expected):
    'Solves deterministic DA economic dispatch given point forecasts'
    horizon = config['horizon']
    
    #DA Variables
    p_G = cp.Variable((grid['n_unit'], horizon))
    R_up = cp.Variable((grid['n_unit'], horizon))
    R_down = cp.Variable((grid['n_unit'], horizon))
    flow_da = cp.Variable((grid['n_lines'],horizon))
    theta_da = cp.Variable((grid['n_nodes'], horizon))
    Demand_slack = cp.Variable((grid['n_loads'], horizon))

    Det_solutions = []
    print('Solving deterministic optimization problem for each day')
    ndays = int(Node_demand_expected.shape[1]/horizon)
    
    for i in range(ndays):
        if i%25==0:print('Day ', i)

        start = i*horizon
        stop = (i+1)*horizon
        
        ###### DA constraints
        Constraints_DA = []
        #Generator Constraints
        Constraints_DA += [p_G <= grid['Pmax'].repeat(24,axis=1),
                         p_G[:,1:]-p_G[:,:-1] <= grid['Ramp_up_rate'].repeat(horizon-1,axis=1),
                         p_G[:,:-1]-p_G[:,1:] <= grid['Ramp_down_rate'].repeat(horizon-1,axis=1), 
                         R_up <= grid['R_up_max'].repeat(24,axis=1),
                         R_down <= grid['R_down_max'].repeat(24,axis=1),
                         Demand_slack >= 0, p_G >= 0, R_down >= 0, R_up>=0]
        
        
        #DA Network flow
        Constraints_DA += [flow_da == grid['b_diag']@grid['A']@theta_da,
                        flow_da <= grid['Line_Capacity'].repeat(24,axis=1), 
                        flow_da >= -grid['Line_Capacity'].repeat(24,axis=1),
                        theta_da[0,:] == 0]
        
        #DA Node balance
        Constraints_DA += [ grid['node_G']@p_G + grid['node_L']@(Demand_slack-Node_demand_expected[:, start:stop]) == grid['B']@theta_da]
        #DA Objective
        DA_cost = cp.sum(grid['Cost']@p_G) + grid['VOLL']*cp.sum(Demand_slack)
            
        prob = cp.Problem(cp.Minimize( DA_cost) , Constraints_DA )
        prob.solve( solver = 'GUROBI', verbose = False)
        if prob.objective.value==None: print('Infeasible or unbounded')
        
        solution = {'p': p_G.value, 'slack': Demand_slack.value, 'flow':flow_da.value, 'theta': theta_da.value, 
                    'R_up': R_up.value, 'R_down': R_down.value, 'LMP': -Constraints_DA[-1].dual_value}

        Det_solutions.append(solution)    
    return Det_solutions

def deterministic_opt_gp(grid, config, Node_demand_expected):
    'Solves deterministic DA economic dispatch given point forecasts in Gurobi'
    horizon = config['horizon']
    
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)
    m.setParam('LPWarmStart', 2)
    m.setParam('Presolve', 2)
    
    #DA Variables
    p_G = m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    R_up = m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0)
    R_down = m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0)
    Demand_slack = m.addMVar((grid['n_loads'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0)

    flow_da = m.addMVar((grid['n_lines'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    theta_da = m.addMVar((grid['n_nodes'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            
    # DA Constraints
    #gen limits
    m.addConstrs( p_G[:,t] <= grid['Pmax'].reshape(-1) for t in range(horizon))
    m.addConstrs( R_up[:,t] <= grid['R_up_max'].reshape(-1) for t in range(horizon))
    m.addConstrs( R_down[:,t] <= grid['R_down_max'].reshape(-1) for t in range(horizon))

    m.addConstrs(p_G[:,t+1]-p_G[:,t] <= grid['Ramp_up_rate'].reshape(-1) for t in range(horizon-1))
    m.addConstrs(p_G[:,t]-p_G[:,t+1] <= grid['Ramp_down_rate'].reshape(-1) for t in range(horizon-1))
    
    # Network flow
    m.addConstrs(grid['b_diag']@grid['A']@theta_da[:,t] <= grid['Line_Capacity'].reshape(-1) for t in range(horizon))
    m.addConstrs(grid['b_diag']@grid['A']@theta_da[:,t] >= -grid['Line_Capacity'].reshape(-1) for t in range(horizon))

    m.addConstr(theta_da[0,:] == 0)

    #Node balance
    Det_solutions = []
    print('Solving deterministic optimization problem for each day')
    ndays = int(Node_demand_expected.shape[1]/horizon)

    for i in range(ndays):
        if i%25==0:print('Day ', i)

        start = i*horizon
        stop = (i+1)*horizon

        # !!!! Changes each day
        node_balance = m.addConstrs(grid['node_G']@p_G[:,t] + grid['node_L']@Demand_slack[:,t]\
                                    -grid['node_L']@Node_demand_expected[:, start:stop][:,t] - grid['B']@theta_da[:,t] == 0 for t in range(horizon))
    
        DA_cost = sum([grid['Cost']@p_G[:,t] + grid['VOLL']*Demand_slack[:,t].sum() for t in range(horizon)]) 
        
        # Objective
        m.setObjective(DA_cost, gp.GRB.MINIMIZE)                    
        m.optimize()
        for c in [node_balance]:
            m.remove(c)
            
        solution = {'p': p_G.X, 'slack': Demand_slack.X, 'flow':grid['b_diag']@grid['A']@theta_da.X,\
                    'theta': theta_da.X, 'R_up': R_up.X, 'R_down': R_down.X}
        Det_solutions.append(solution)    

        #print('Runtime: ', m.Runtime)
    return Det_solutions

def stochastic_opt_gp(grid, config, Node_demand_expected):
    'Solves deterministic DA economic dispatch given point forecasts in Gurobi'
    horizon = config['horizon']
    Nscen = config['n_scen']
    
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    
    #DA Variables
    p_G = m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    R_up = m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0)
    R_down = m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0)
    Demand_slack = m.addMVar((grid['n_loads'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0)
    flow_da = m.addMVar((grid['n_lines'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    theta_da = m.addMVar((grid['n_nodes'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
    #RT Variables
    r_up= [m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0) for scen in range(Nscen)]
    r_down= [m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0) for scen in range(Nscen)]
    L_shed= [m.addMVar((grid['n_loads'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0) for scen in range(Nscen)]
    flow_rt= [m.addMVar((grid['n_lines'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY) for scen in range(Nscen)]
    theta_rt= [m.addMVar((grid['n_nodes'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY) for scen in range(Nscen)]
    G_shed = [m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0) for scen in range(Nscen)]
    
    ###### DA Constraints
    #gen limits
    m.addConstrs( p_G[:,t] <= grid['Pmax'].reshape(-1) for t in range(horizon))
    m.addConstrs( R_up[:,t] <= grid['R_up_max'].reshape(-1) for t in range(horizon))
    m.addConstrs( R_down[:,t] <= grid['R_down_max'].reshape(-1) for t in range(horizon))

    m.addConstrs(p_G[:,t+1]-p_G[:,t] <= grid['Ramp_up_rate'].reshape(-1) for t in range(horizon-1))
    m.addConstrs(p_G[:,t]-p_G[:,t+1] <= grid['Ramp_down_rate'].reshape(-1) for t in range(horizon-1))
    
    # Network flow
    m.addConstrs(grid['b_diag']@grid['A']@theta_da[:,t] <= grid['Line_Capacity'].reshape(-1) for t in range(horizon))
    m.addConstrs(grid['b_diag']@grid['A']@theta_da[:,t] >= -grid['Line_Capacity'].reshape(-1) for t in range(horizon))
    m.addConstr(theta_da[0,:] == 0)
    
    DA_cost = sum([grid['Cost']@p_G[:,t] + grid['VOLL']*Demand_slack[:,t].sum() for t in range(horizon)]) 

    ###### RT constraints
    RT_cost = 0
    
    Constraints_RT = []
    for scen in range(Nscen): 
        m.addConstrs( r_up[scen][:,t] <=-p_G[:,t] + grid['Pmax'].reshape(-1) for t in range(horizon))
        m.addConstrs( r_up[scen][:,t] <= R_up[:,t] for t in range(horizon))

        m.addConstrs( r_down[scen][:,t] <= R_down[:,t] for t in range(horizon))
        m.addConstrs( r_down[scen][:,t] <= p_G[:,t] for t in range(horizon))

        m.addConstrs( G_shed[scen][:,t] <= p_G[:,t] for t in range(horizon))

        RT_cost = RT_cost + 1/Nscen*sum([grid['Cost_reg_up']@r_up[scen][:,t] - grid['Cost_reg_down']@r_down[scen][:,t] \
                                 + grid['VOLL']*L_shed[scen][:,t].sum() +  grid['gshed']@G_shed[scen][:,t] \
                                     for t in range(horizon)])



    #!!!!!
    m.addConstrs( L_shed[scen][:,t] <= Node_demand_scenarios[:,start:stop,scen][:,t] for t in range(horizon))


    #Node injections
    Constraints_RT += [ grid['node_G']@(r_up[scen]-r_down[scen]-G_shed[scen]) \
                       + grid['node_L']@(L_shed[scen]-Node_demand_scenarios[:,start:stop,scen]+Node_demand_expected[:,start:stop]) \
                       == grid['B']@(theta_rt[scen]-theta_da)]
    
        
    #Node balance
    Stoch_solutions = []
    print('Solving stochastic optimization problem for each day')
    ndays = int(Node_demand_expected.shape[1]/horizon)

    for i in range(ndays):
        if i%25==0:print('Day ', i)

        start = i*horizon
        stop = (i+1)*horizon

        # !!!! Changes each day
        node_balance_da = m.addConstrs(grid['node_G']@p_G[:,t] + grid['node_L']@Demand_slack[:,t]\
                                    -grid['node_L']@Node_demand_expected[:, start:stop][:,t] - grid['B']@theta_da[:,t] == 0 for t in range(horizon))

        # !!!! Changes each day
        node_balance_rt = [m.addConstrs(grid['node_G']@(r_up[scen][:,t]-r_down[scen][:,t]-G_shed[scen][:,t]) \
                           + grid['node_L']@L_shed[scen][:,t]+grid['node_L']@(Node_demand_expected[:,start:stop] - Node_demand_scenarios[:,start:stop,scen])[:,t] \
                           == grid['B']@(theta_rt[scen][:,t]-theta_da[:,t]) for t in range(horizon)) for scen in N_scen]
                                   
        # Objective
        m.setObjective(DA_cost+RT_cost, gp.GRB.MINIMIZE)                    
        m.optimize()
        
        for c in [node_balance]:
            m.remove(c)
            
        solution = {'p': p_G.X, 'slack': Demand_slack.X, 'flow':grid['b_diag']@grid['A']@theta_da.X,\
                    'theta': theta_da.X, 'R_up': R_up.X, 'R_down': R_down.X}
        Det_solutions.append(solution)    

    return Det_solutions

def stochastic_opt(grid, config, Node_demand_expected, Node_demand_scenarios):
    'Solves deterministic DA economic dispatch given point forecasts'
    horizon = config['horizon']
    Nscen = config['n_scen']
    
    #DA Variables
    p_G = cp.Variable((grid['n_unit'], horizon))
    R_up = cp.Variable((grid['n_unit'], horizon))
    R_down = cp.Variable((grid['n_unit'], horizon))
    flow_da = cp.Variable((grid['n_lines'],horizon))
    theta_da = cp.Variable((grid['n_nodes'], horizon))
    Demand_slack = cp.Variable((grid['n_loads'], horizon))

    #RT Variables
    r_up= [cp.Variable((grid['n_unit'], horizon)) for scen in range(Nscen)]
    r_down= [cp.Variable((grid['n_unit'], horizon)) for scen in range(Nscen)]
    L_shed= [cp.Variable((grid['n_loads'],horizon)) for scen in range(Nscen)]
    flow_rt= [cp.Variable((grid['n_lines'],horizon)) for scen in range(Nscen)]
    theta_rt = [cp.Variable((grid['n_nodes'], horizon)) for scen in range(Nscen)]
    G_shed = [cp.Variable((grid['n_unit'], horizon)) for scen in range(Nscen)]

    Stoch_solutions = []
    
    print('Solving Stochastic Optimization...')
    ndays = int(Node_demand_expected.shape[1]/horizon)
    for i in range(ndays):
        if i%25==0:print('Day: ',i)
        start = i*horizon
        stop = (i+1)*horizon
    
            
        ###### DA constraints
        Constraints_DA = []
        #Generator Constraints
        Constraints_DA += [p_G <= grid['Pmax'].repeat(horizon,axis=1),
                        p_G[:,1:]-p_G[:,:-1] <= grid['Ramp_up_rate'].repeat(horizon-1,axis=1),
                        p_G[:,:-1]-p_G[:,1:] <= grid['Ramp_down_rate'].repeat(horizon-1,axis=1), 
                        R_up <= grid['R_up_max'].repeat(horizon,axis=1),
                        R_down <= grid['R_down_max'].repeat(horizon,axis=1),
                        p_G>=0, R_up>=0, R_down>=0, Demand_slack >= 0]
    
        
        DA_cost = cp.sum(grid['Cost']@p_G) + grid['VOLL']*cp.sum(Demand_slack)
                
        #DA Network flow
        Constraints_DA += [flow_da == grid['b_diag']@grid['A']@theta_da,
                        flow_da <= grid['Line_Capacity'].repeat(horizon,axis=1), 
                        flow_da >= -grid['Line_Capacity'].repeat(horizon,axis=1),
                        theta_da[0,:] == 0]
    
        #DA Node balance
        Constraints_DA += [ grid['node_G']@p_G + grid['node_L']@(Demand_slack-Node_demand_expected[:,start:stop] ) == grid['B']@theta_da]
        DA_cost = cp.sum(grid['Cost']@p_G) + grid['VOLL']*cp.sum(Demand_slack)
    
        
        ###### RT constraints
        RT_cost = 0
        
        Constraints_RT = []
        for scen in range(Nscen):       
            # Feasbility limits 
            Constraints_RT += [ r_up[scen] <= -p_G + grid['Pmax'].repeat(horizon,axis=1),
                               r_up[scen] <= R_up,      
                                  r_down[scen] <= p_G,
                                 r_down[scen] <= R_down,
                                 L_shed[scen] <= Node_demand_scenarios[:,start:stop,scen],
                                 G_shed[scen] <= p_G,
                                 r_up[scen] >= 0, r_down[scen] >= 0, 
                                 L_shed[scen] >= 0, G_shed[scen] >= 0]

    
        ############## Real-time balancing problem
        #RT Network flow
        Constraints_RT += [flow_rt[scen] == grid['b_diag']@grid['A']@theta_rt[scen],
                           flow_rt[scen] <= grid['Line_Capacity'].repeat(horizon,axis=1), 
                           flow_rt[scen] >= -grid['Line_Capacity'].repeat(horizon,axis=1),
                           theta_rt[scen][0,:] == 0] 
            
        #Node injections
        Constraints_RT += [ grid['node_G']@(r_up[scen]-r_down[scen]-G_shed[scen]) \
                           + grid['node_L']@(L_shed[scen]-Node_demand_scenarios[:,start:stop,scen]+Node_demand_expected[:,start:stop]) \
                           == grid['B']@(theta_rt[scen]-theta_da)]
        
        RT_cost = RT_cost + 1/Nscen*cp.sum( grid['Cost_reg_up']@r_up[scen] 
                                           - grid['Cost_reg_down']@r_down[scen]\
                                            + grid['VOLL']*cp.sum(L_shed[scen],axis=0) + grid['gshed']*cp.sum(G_shed[scen],axis=0) ) 
        
        # Actually only care about RT costs not the DA costs (these just depend on demand)
        prob = cp.Problem(cp.Minimize(DA_cost+RT_cost) , Constraints_DA+Constraints_RT)
        prob.solve( solver = 'GUROBI', verbose = False)
        
        solution = {'p': p_G.value, 'flow':flow_da.value, 'theta': theta_da.value, 
                    'R_up': grid['R_up_max'].repeat(horizon,axis=1), 
                    'R_down': grid['R_down_max'].repeat(horizon,axis=1), 
                    'LMP': -Constraints_DA[-1].dual_value}

        Stoch_solutions.append(solution)
        if prob.objective.value ==  None:
            print('Infeasible or unbound')    
    return Stoch_solutions
    'Solves deterministic DA economic dispatch given point forecasts'
    horizon = config['horizon']
    Nscen = config['n_scen']
    
    #DA Variables
    p_G = cp.Variable((grid['n_unit'], horizon))
    R_up = cp.Variable((grid['n_unit'], horizon))
    R_down = cp.Variable((grid['n_unit'], horizon))
    flow_da = cp.Variable((grid['n_lines'],horizon))
    theta_da = cp.Variable((grid['n_nodes'], horizon))
    Demand_slack = cp.Variable((grid['n_loads'], horizon))

    #RT Variables
    r_up= [cp.Variable((grid['n_unit'], horizon)) for scen in range(Nscen)]
    r_down= [cp.Variable((grid['n_unit'], horizon)) for scen in range(Nscen)]
    L_shed= [cp.Variable((grid['n_loads'],horizon)) for scen in range(Nscen)]
    flow_rt= [cp.Variable((grid['n_lines'],horizon)) for scen in range(Nscen)]
    theta_rt = [cp.Variable((grid['n_nodes'], horizon)) for scen in range(Nscen)]
    G_shed = [cp.Variable((grid['n_unit'], horizon)) for scen in range(Nscen)]

    Stoch_solutions = []
    
    print('Solving Stochastic Optimization...')
    ndays = int(Node_demand_expected.shape[1]/horizon)
    for i in range(ndays):
        if i%25==0:print('Day: ',i)
        start = i*horizon
        stop = (i+1)*horizon
    
            
        ###### DA constraints
        Constraints_DA = []
        #Generator Constraints
        Constraints_DA += [p_G <= grid['Pmax'].repeat(horizon,axis=1),
                        p_G[:,1:]-p_G[:,:-1] <= grid['Ramp_up_rate'].repeat(horizon-1,axis=1),
                        p_G[:,:-1]-p_G[:,1:] <= grid['Ramp_down_rate'].repeat(horizon-1,axis=1), 
                        R_up <= grid['R_up_max'].repeat(horizon,axis=1),
                        R_down <= grid['R_down_max'].repeat(horizon,axis=1),
                        p_G>=0, R_up>=0, R_down>=0, Demand_slack >= 0]
    
        
        DA_cost = cp.sum(grid['Cost']@p_G) + grid['VOLL']*cp.sum(Demand_slack)
                
        #DA Network flow
        Constraints_DA += [flow_da == grid['b_diag']@grid['A']@theta_da,
                        flow_da <= grid['Line_Capacity'].repeat(horizon,axis=1), 
                        flow_da >= -grid['Line_Capacity'].repeat(horizon,axis=1),
                        theta_da[0,:] == 0]
    
        #DA Node balance
        Constraints_DA += [ grid['node_G']@p_G + grid['node_L']@(Demand_slack-Node_demand_expected[:,start:stop] ) == grid['B']@theta_da]
        DA_cost = cp.sum(grid['Cost']@p_G) + grid['VOLL']*cp.sum(Demand_slack)
    
        # Actually only care about RT costs not the DA costs (these just depend on demand)
        prob = cp.Problem(cp.Minimize(DA_cost) , Constraints_DA)
        prob.solve( solver = 'GUROBI', verbose = False)
        
        
        ###### RT constraints
        start_1 = time.time()
        RT_cost = 0
        
        Constraints_RT = []
        for scen in range(Nscen):       
            # Feasbility limits 
            Constraints_RT += [ r_up[scen] <= -p_G.value + grid['Pmax'].repeat(horizon,axis=1),
                               r_up[scen] <= R_up.value,      
                                  r_down[scen] <= p_G.value,
                                 r_down[scen] <= R_down.value,
                                 L_shed[scen] <= Node_demand_scenarios[:,start:stop,scen],
                                 G_shed[scen] <= p_G.value,
                                 r_up[scen] >= 0, r_down[scen] >= 0, 
                                 L_shed[scen] >= 0, G_shed[scen] >= 0]

    
            ############## Real-time balancing problem
            #RT Network flow
            Constraints_RT += [flow_rt[scen] == grid['b_diag']@grid['A']@theta_rt[scen],
                               flow_rt[scen] <= grid['Line_Capacity'].repeat(horizon,axis=1), 
                               flow_rt[scen] >= -grid['Line_Capacity'].repeat(horizon,axis=1),
                               theta_rt[scen][0,:] == 0] 
            
            #Node injections
            Constraints_RT += [ grid['node_G']@(r_up[scen]-r_down[scen]-G_shed[scen]) \
                               + grid['node_L']@(L_shed[scen]-Node_demand_scenarios[:,start:stop,scen]+Node_demand_expected[:,start:stop]) \
                               == grid['B']@(theta_rt[scen]-theta_da.value)]
        
            RT_cost = RT_cost + 1/Nscen*cp.sum( grid['Cost_reg_up']@r_up[scen] 
                                               - grid['Cost_reg_down']@r_down[scen]\
                                                + grid['VOLL']*cp.sum(L_shed[scen],axis=0) + grid['gshed']*cp.sum(G_shed[scen],axis=0) ) 
        
        # Actually only care about RT costs not the DA costs (these just depend on demand)
        prob = cp.Problem(cp.Minimize(RT_cost) , Constraints_RT)
        prob.solve( solver = 'GUROBI', verbose = False)
        print('SAA time: ', time.time()-start_1)
        
        ###### RT constraints
        #RT Variables
        r_up= cp.Variable((grid['n_unit'], horizon))
        r_down= cp.Variable((grid['n_unit'], horizon))
        L_shed= cp.Variable((grid['n_loads'],horizon))
        flow_rt= cp.Variable((grid['n_lines'],horizon))
        theta_rt = cp.Variable((grid['n_nodes'], horizon))
        G_shed = cp.Variable((grid['n_unit'], horizon))
        
        start_2 = time.time()
        RT_cost = 0
        
        for scen in range(Nscen):       
            Constraints_RT = []

            # Feasbility limits 
            Constraints_RT += [ r_up <= -p_G.value + grid['Pmax'].repeat(horizon,axis=1),
                               r_up <= R_up.value,      
                                  r_down <= p_G.value,
                                 r_down <= R_down.value,
                                 L_shed <= Node_demand_scenarios[:,start:stop,scen],
                                 G_shed <= p_G.value,
                                 r_up >= 0, r_down >= 0, 
                                 L_shed >= 0, G_shed >= 0]

    
            ############## Real-time balancing problem
            #RT Network flow
            Constraints_RT += [flow_rt == grid['b_diag']@grid['A']@theta_rt,
                               flow_rt <= grid['Line_Capacity'].repeat(horizon,axis=1), 
                               flow_rt >= -grid['Line_Capacity'].repeat(horizon,axis=1),
                               theta_rt[0,:] == 0] 
                
            #Node injections
            Constraints_RT += [ grid['node_G']@(r_up-r_down-G_shed) \
                               + grid['node_L']@(L_shed-Node_demand_scenarios[:,start:stop,scen]+Node_demand_expected[:,start:stop]) \
                               == grid['B']@(theta_rt-theta_da.value)]
        
            RT_cost = 1/Nscen*cp.sum( grid['Cost_reg_up']@r_up 
                                               - grid['Cost_reg_down']@r_down\
                                                + grid['VOLL']*cp.sum(L_shed,axis=0) + grid['gshed']*cp.sum(G_shed,axis=0) ) 
        
            # Actually only care about RT costs not the DA costs (these just depend on demand)
            prob = cp.Problem(cp.Minimize(RT_cost) , Constraints_RT)
            prob.solve( solver = 'GUROBI', verbose = False)
        print('SAA time: ', time.time()-start_2)
        
        
        
        
    return Stoch_solutions

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
    return total_costs, da_costs, rt_costs, stats_out

def evaluate_single_day(day, solutions, Node_demand_actual, col_names, grid, config, plot = True):
    'Function that takes as input the obtained decisions and returns dataframe with realize costs'
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
            plt.title(col_names[j])
            plt.legend()
            plt.show()
    return
    
#%% Problem parameters
def problem_parameters():
    parameters = {} 
    # Script parameters
    parameters['train'] = False # If true, then train learning components, else load results
    parameters['save'] = True # Save DA dispatch decisions and trained models

    # Optimization Parameters
    parameters['n_scen'] = 100    # Number of scenarios for probabilistic
    parameters['horizon'] = 24 # Optimization horizon (DO NOT CHANGE)
    parameters['peak_load'] = 2700  # Peak hourly demand
    parameters['wind_capacity'] = 200  # (not used)
    
    # Forecasting parameters (only for the forecasting_module.py)
    parameters['quant'] = np.arange(.01, 1, .01)   #For probabilistic forecasts
    
    # Starting dates create training samples of size 6months, 1y, 1.5y and 2y
    #parameters['start_date'] = '2010-06-01' # Controls for sample size
    #parameters['start_date'] = '2010-01-01'
    parameters['start_date'] = '2009-06-01' 
    parameters['split_date'] = '2011-01-01' # Validation split
    return parameters

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main script

# Set up configuration
config = problem_parameters()

#pglib_path = 'C:/Users/akylas.stratigakos/pglib-opf/'
pglib_path =  'C:/Users/akyla/pglib-opf-21.07/'

aggr_wind_df = pd.read_csv('C:\\Users\\akyla\\df-forecast-comb\\data\\GEFCom2014-processed.csv', index_col = 0, header = [0,1])

# normal
#Cases = ['pglib_opf_case5_pjm.m', 'pglib_opf_case30_ieee.m', 'pglib_opf_case39_epri.m', 'pglib_opf_case57_ieee.m', 
#         'pglib_opf_case118_ieee.m', 'pglib_opf_case300_ieee.m']

Cases = ['pglib_opf_case14_ieee.m']

target_bus = 13
capacity = 100

for case in Cases:
    
    case_name = case.split('.')[0]    
    matgrid = CaseFrames(pglib_path + case)
    grid = grid_dict(pglib_path + case)

#%%
grid['C_up'] = 1.8*grid['Cost']
grid['C_down'] = 0.8*grid['Cost']

R_u_max = (grid['Pmax']/3).round(2)
R_d_max = R_u_max

grid['R_u_max'] = R_u_max
grid['R_d_max'] = R_d_max
#%% Create data

# number of observations to train prob. forecasting model
N_sample = len(aggr_wind_df)//2
pred_col = ['wspeed10', 'wdir10_rad', 'wspeed100', 'wdir100_rad']

trainY = aggr_wind_df['Z1']['POWER'][:N_sample].to_frame()
testY = aggr_wind_df['Z1']['POWER'][N_sample:].to_frame()

trainX = aggr_wind_df['Z1'][pred_col][:N_sample]
testX = aggr_wind_df['Z1'][pred_col][N_sample:]


#%% Standard forecasting

from sklearn.ensemble import ExtraTreesRegressor

et_model = ExtraTreesRegressor(n_estimators = 20, min_samples_leaf = 2 ).fit(trainX, trainY)
PO_pred = et_model.predict(testX).reshape(-1,1)

#%% Value-oriented forecasting

pt_model = EnsemblePrescriptiveTree(n_estimators = 20, Nmin = 2, max_features = 4, type_split = 'quant')
pt_model.fit(trainX.values, trainY.values, grid = grid, config = config, parallel = False, capacity = capacity, 
             bus = target_bus, scenario_reduction = True, network = False, num_reduced_scen = 50)
VO_pred = pt_model.cost_oriented_forecast(testX.values, trainX.values, trainY.values)

#%% NN-differential opt. layer

import cvxpy as cp
import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer
import copy

def to_np(x):
    return x.detach().numpy()


#%% NN/Prediction-oriented training

tensor_trainX = torch.FloatTensor(trainX.values[:-1000])
tensor_trainY = torch.FloatTensor(trainY.values[:-1000])

tensor_valX = torch.FloatTensor(trainX.values[-1000:])
tensor_valY = torch.FloatTensor(trainY.values[-1000:])

tensor_testX = torch.FloatTensor(testX.values)
tensor_testY = torch.FloatTensor(testY.values)


patience = 25
batch_size = 100
num_epochs = 1000

train_data = torch.utils.data.TensorDataset(tensor_trainX, tensor_trainY)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

PO_NNmodel = nn.Sequential(
    nn.Linear(4, 30),
    nn.ReLU(),
    nn.Linear(30, 30),
    nn.ReLU(),
    nn.Linear(30, 1))

Losses = []
optimizer = torch.optim.Adam(PO_NNmodel.parameters(), lr=1e-3)
best_val_loss = float('inf')

for epoch in range(num_epochs):
    
    PO_NNmodel.train()
    running_loss = 0.0

    '''
    train_inputs, train_labels = next(iter(train_loader))
    
    optimizer.zero_grad()
    
    # forward pass
    y_hat = PO_NNmodel(train_inputs)
    # evaluate loss and backward pass        
    loss = (y_hat - train_labels).norm(2)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()

    '''
    for i, data in enumerate(train_loader, 0):
        # sample batch
        train_inputs, train_labels = data
        optimizer.zero_grad()
        
        # forward pass
        y_hat = PO_NNmodel(train_inputs)
        # evaluate loss and backward pass        
        loss = (y_hat - train_labels).norm(2)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    
    # Validation
    PO_NNmodel.eval()
    with torch.no_grad():
        
        val_outputs = PO_NNmodel(tensor_valX)


        val_loss = (val_outputs - tensor_valY).norm(2) 

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss / len(train_loader):.4f} - Val Loss: {val_loss.item():.4f}")

    # Implement early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(PO_NNmodel.state_dict())
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break
        
PO_NNmodel.load_state_dict(best_model_weights)

# generate accuracy-oriented predictions
PO_NN_pred = to_np(PO_NNmodel(tensor_testX).clamp(0,1))
print(eval_point_pred(PO_NN_pred, testY))
    
#%% NN/ Decision-focused learning
patience = 10
batch_size = 50
num_epochs = 1000
check_validation = False
train_data = torch.utils.data.TensorDataset(tensor_trainX, tensor_trainY)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

###### DA market layer
# Variables of inner problem
p_gen = cp.Variable((grid['n_unit']))
slack = cp.Variable((1))

cost = cp.Variable((1))    
wind_prediction = cp.Parameter((1))
constraints = [p_gen >= 0, p_gen <= grid['Pmax'].reshape(-1)] \
            + [slack >=0]\
            + [p_gen.sum() + capacity*wind_prediction + slack.sum() == grid['Pd'].sum()]\
            + [cost == grid['Cost']@p_gen + grid['VOLL']*slack]

objective = cp.Minimize( sum(cost) ) 
problem = cp.Problem(objective, constraints)
dcopf_layer = CvxpyLayer(problem, parameters=[wind_prediction], variables=[p_gen, slack, cost] )

DO_NNmodel = nn.Sequential(
    nn.Linear(4, 30),
    nn.ReLU(),
    nn.Linear(30, 30),
    nn.ReLU(), 
    nn.Linear(30, 1),
    dcopf_layer
    )


#### RT market-clearing layer
p_gen_param = cp.Parameter((grid['n_unit']))
w_pred_error = cp.Parameter((1))

r_up = cp.Variable((grid['n_unit']))
r_down = cp.Variable((grid['n_unit']))

rt_slack_up = cp.Variable((1))
rt_slack_down = cp.Variable((1))

rt_cost = cp.Variable((1))    

rt_constraints = [r_up >= 0, r_down >= 0, r_up <= grid['Pmax'].reshape(-1)-p_gen_param, 
               r_down <= p_gen_param, rt_slack_up >= 0, rt_slack_down >= 0] \
            + [r_up.sum() - r_down.sum() + capacity*w_pred_error + rt_slack_up.sum() - rt_slack_down.sum() == 0]\
            + [rt_cost == (grid['C_up']-grid['Cost'])@r_up + (grid['Cost'] - grid['C_down'])@r_down+ 
               grid['VOLL']*(rt_slack_up.sum() + rt_slack_down.sum())]

rt_objective = cp.Minimize( sum(rt_cost) ) 
rt_problem = cp.Problem(rt_objective, rt_constraints)
rt_layer = CvxpyLayer(rt_problem, parameters=[p_gen_param, w_pred_error], 
                      variables=[r_up, r_down, rt_slack_up, rt_slack_down, rt_cost] )

Losses = []
optimizer = torch.optim.Adam(DO_NNmodel.parameters(), lr=1e-3)
best_val_loss = float('inf')

for epoch in range(num_epochs):
    
    # enable train functionality
    DO_NNmodel.train()
    running_loss = 0.0

    train_inputs, train_labels = next(iter(train_loader))
    # clear gradients before backward pass    
    optimizer.zero_grad()
    
    #### forward pass/ solve for DA-market
    dcopf_output = DO_NNmodel(train_inputs)
    # generator setpoints
    p_hat = dcopf_output[0]
    # predictions
    y_pred = DO_NNmodel[:-1](train_inputs)
    w_error_hat = train_labels - y_pred
    # solve RT market
    rt_output = rt_layer(p_hat, w_error_hat)
    # cost of redispatching
    loss = rt_output[-1].sum()
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()

    '''
    for i, data in enumerate(train_loader, 0):
        # sample batch
        train_inputs, train_labels = data
        optimizer.zero_grad()
        
        # forward pass
        y_hat = DO_NNmodel(train_inputs)
        # evaluate loss and backward pass        
        loss = (y_hat - train_labels).norm(2)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    '''
    
    # Validation
    DO_NNmodel.eval()
    
    with torch.no_grad():
        
        val_outputs = DO_NNmodel(tensor_valX)
        
        # predictions
        val_predictions = DO_NNmodel[:-1](tensor_valX)
        
        val_w_error_hat = tensor_valY - val_predictions
        # solve RT market
        val_rt_output = rt_layer(val_outputs[0], val_w_error_hat)
        # cost of redispatching
        val_rt_loss = val_rt_output[-1].mean()        

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss / len(train_loader):.4f} - Val Loss: {val_rt_loss.item():.4f}")

    # Implement early stopping
    if val_rt_loss < best_val_loss:
        best_val_loss = val_rt_loss
        best_model_weights = copy.deepcopy(DO_NNmodel.state_dict())
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break
        
DO_NNmodel.load_state_dict(best_model_weights)

# generate value-oriented predictions/ skip last layer of the model
DO_NN_pred = to_np(DO_NNmodel[:-1](tensor_testX).clamp(0,1))
print(eval_point_pred(DO_NN_pred, testY))

#%%
for n in range(num_epochs):
    
    ix = np.random.choice(range(len(trainY)), batch_size, replace = False)
    
    batchX = trainX.values[ix]
    batchY = trainY.values[ix]
    
    output = model(torch.FloatTensor(batchX))
    
    p_hat = output[0]
    
    y_pred = model[:-1](torch.FloatTensor(batchX))
    w_error_hat = torch.FloatTensor(batchY) - y_pred
    
    rt_output = rt_layer(p_hat, w_error_hat)
    
    
    loss = rt_output[-1]
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    Losses.append(to_np(loss))
    
    
#%%

plt.plot(testY.values[:168], label = 'Actual')
plt.plot(PO_NN_pred[:168])
plt.plot(DO_NN_pred[:168])
plt.show()


#%%
PO_da_cost, PO_rt_cost, PO_da_sol, PO_rt_sol = DA_RT_maker(grid, capacity*PO_NN_pred, capacity*testY.values, target_bus,
                network = True, plot = False, verbose = 0, return_ave_cpu = False)

VO_da_cost, VO_rt_cost, VO_da_sol, VO_rt_sol = DA_RT_maker(grid, capacity*DO_NN_pred, capacity*testY.values, target_bus,
                network = True, plot = False, verbose = 0, return_ave_cpu = False)

PO_total_cost = PO_da_cost + PO_rt_cost
VO_total_cost = VO_da_cost + VO_rt_cost

print('Mean DA cost')
print(f'PO:{PO_da_cost.mean()}')
print(f'VO:{VO_da_cost.mean()}')

print('Mean RT cost')
print(f'PO:{PO_rt_cost.mean()}')
print(f'VO:{VO_rt_cost.mean()}')

print('Mean Total cost')
print(f'PO:{PO_total_cost.mean()}')
print(f'VO:{VO_total_cost.mean()}')
t = 72

plt.plot(grid['Pd'].sum() - capacity*testY.values[:72], label = 'Actual')
plt.plot(PO_da_sol['p'].sum(1)[:72], label = 'DA')
plt.plot(PO_da_sol['p'].sum(1)[:72] + PO_rt_sol['r_up'].sum(1)[:72], '--', color = 'black', label = 'DA+r_up')
plt.legend()
plt.show()