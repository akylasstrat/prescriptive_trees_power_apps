# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:13:29 2023

@author: a.stratigakos
"""

import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from matpowercaseframes import CaseFrames
import pickle

def create_demands(nsamples, grid, demand_std = 0.05, distribution = 'uniform', range_ = 0.4):
    ''' Function to generate net demands.
        - nsamples: number of observations
        - grid: dictionary with the test case
        - distribution: uniform or normal
        - range_: the demands vary in interval [d - range_*abs(d), d] -> handles negative load values properly
        - demand_std: standard deviation in percentage, for normal distribution'''

    if distribution == 'uniform':
        samples = np.random.uniform(low=0, high=range_, size = (nsamples, grid['n_loads']))
        demands = grid['Pd'].reshape(1,-1) - np.abs(grid['Pd'].reshape(1,-1))*samples
    
    elif distribution == 'normal':
        # Multivariate normal distribution
        std_Pd = demand_std*grid['Pd']        
        ub_Pd = grid['Pd']
        lb_Pd = grid['Pd'] - range_*np.abs(grid['Pd'])
        range_Pd = ub_Pd - lb_Pd
        mean_Pd = ((ub_Pd-lb_Pd)/2+lb_Pd).reshape(-1)
        
        # generate correlation matrix *must be symmetric*    
        np.random.seed(0)
            
        a = np.random.rand(grid['n_loads'], grid['n_loads'])
        R = np.tril(a) + np.tril(a, -1).T
        for i in range(grid['n_loads']): R[i,i] = 1
        # estimate covariance matrix
        S_cov = np.diag(std_Pd)@R@np.diag(std_Pd)        
        # sample demands, project them into support
        demands = np.random.multivariate_normal(mean_Pd, S_cov, size = nsamples).round(2)
        
        for d in range(grid['n_loads']):
            demands[demands[:,d]>ub_Pd[d], d] = ub_Pd[d]
            demands[demands[:,d]<lb_Pd[d], d] = lb_Pd[d]

    return demands

def percentage_infeasible(prescriptions, grid, net_demand, tolerance = 1e-2):
    ''' Returns infeasibility rate and boolean mask with infeasible instances
        - prescriptions: the model output
        - grid: dictionary with test network
        - net_demand: net demand observations
        - tolerance: to deal with numerical issues'''
    node_G = grid['node_G']
    node_L = grid['node_L']
    PTDF = grid['PTDF']
    
    node_inj = node_G@prescriptions[:,:grid['n_unit']].T - node_L@net_demand.T
    line_flows = (PTDF@node_inj).T
    
    line_flow_infeas = 1*(np.abs(line_flows) > grid['Line_Capacity'].reshape(1,-1) + tolerance).any(1)
    balance_infeas = 1*(np.abs(prescriptions.sum(1)) > net_demand.sum(1)*1.01) 
    
    total_infeas = line_flow_infeas + balance_infeas - line_flow_infeas*balance_infeas

    print('Total % infeasible flows: ', (100*(np.abs(line_flows) > grid['Line_Capacity'].reshape(1,-1)).sum()/line_flows.size))    
    print('% of instances with infeasible flows: ', (100*line_flow_infeas.sum()/line_flow_infeas.size).round(2))    
    print('% balance infeasible: ', (100*balance_infeas.sum()/balance_infeas.size).round(2))    
    print('% of infeasible instances: ', (100*total_infeas.sum()/total_infeas.size).round(2))    
    mask = total_infeas==1
    return (100*total_infeas.sum()/total_infeas.size).round(2), mask

def DA_RT_maker(grid, w_predictions, w_actual, bus,
                network = True, plot = False, verbose = 0, return_ave_cpu = False):
    ''' Clears the forward (DA) market, returns the solutions in dictionary. 
        Creates the problem once in GUROBI, solves for the length of data using horizon step.
        - grid: dictionary with the details of the network
        - demands: net load demands at each node
        - network: if True, solves a DC-OPF, else solves an Economic Dispatch problem
        - horizon: the solution horizon (24 for solving the DA horizon)
        - return_ave_cpu: estimates average cpu time to solve one instance
        - verbose: if ~0, prints GUROBI output
        - plot: if True, creates some plots for check '''
         
    n_samples = len(w_predictions)    
    # Solve DA problem
    #n_samples = int(len(demands)/horizon)

    # Parameters
    node_Wind = np.zeros((grid['n_nodes'], 1))
    node_Wind[bus] = 1
    Pmax = grid['Pmax']
    node_G = grid['node_G']
    node_L = grid['node_L']
    Cost = grid['Cost']
    PTDF = grid['PTDF']
    VoLL = grid['VOLL']
    VoWS = grid['gshed']

    ###### DA constraints
    da_market = gp.Model()
    da_market.setParam('OutputFlag', verbose)

    # DA Variables
    p_G = da_market.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    slack_u = da_market.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'slack')
    flow_da = da_market.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    #theta_da = m.addMVar((grid['n_nodes']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    node_inj = da_market.addMVar((grid['n_nodes']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
    exp_w = da_market.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
    #### Problem Constraints
    
    #generator technical limits
    da_market.addConstr( p_G <= Pmax.reshape(-1))
    #da_market.addConstr( exp_w == predicted_y)
    
    # Network constraints
    da_market.addConstr(node_inj == (node_G@p_G + node_Wind@exp_w - node_L@grid['Pd'] ))
    da_market.addConstr(flow_da == PTDF@node_inj )                    
    da_market.addConstr(flow_da <= grid['Line_Capacity'].reshape(-1))
    da_market.addConstr(flow_da >= -grid['Line_Capacity'].reshape(-1))
        
    # Node balance for t for DC-OPF
    da_market.addConstr( p_G.sum() + exp_w.sum() == grid['Pd'].sum())

    # DA cost for specific day/ expression        
    da_market.setObjective(Cost@p_G, gp.GRB.MINIMIZE)                    
    
    
    #da_market.optimize()
    #DA_cost = da_market.ObjVal
    
    ###### Declare model for RT market 
    #RT Variables
    rt_prob = gp.Model()
    rt_prob.setParam('OutputFlag', 0)

    ###### DA constraints
    # DA Variables
    p_DA = rt_prob.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    r_up = rt_prob.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    r_down = rt_prob.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    L_shed = rt_prob.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    flow_rt = rt_prob.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    #theta_rt = rt_prob.addMVar((grid['n_nodes']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    node_inj_rt = rt_prob.addMVar((grid['n_nodes']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    w_rt = rt_prob.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
    # Technical limits
    rt_prob.addConstr( r_up <= -p_DA + Pmax.reshape(-1))
    rt_prob.addConstr( r_up <= grid['R_u_max'].reshape(-1))
    
    rt_prob.addConstr( r_down <= p_DA)
    rt_prob.addConstr( r_down <= grid['R_d_max'].reshape(-1))

    rt_prob.addConstr( L_shed <= grid['Pd'].reshape(-1))
        
    # Network constraints
    rt_prob.addConstr(node_inj_rt == (node_G@(r_up-r_down + p_DA) + node_Wind@(w_rt) - node_L@(grid['Pd']-L_shed) ))
    rt_prob.addConstr(flow_rt == PTDF@node_inj_rt)                    
    rt_prob.addConstr(flow_rt <= grid['Line_Capacity'].reshape(-1))
    rt_prob.addConstr(flow_rt >= -grid['Line_Capacity'].reshape(-1))
    
    rt_prob.addConstr( p_DA.sum() + r_up.sum() -r_down.sum() + w_rt.sum() + L_shed.sum() == grid['Pd'].sum())
    rt_prob.setObjective(grid['C_up']@r_up - grid['C_down']@r_down + VoLL*L_shed.sum())
    
    DA_cost = []
    RT_cost = []
    da_solutions = {'p': [], 'flow_da':[]}
    rt_solutions = {'r_up': [], 'r_down':[], 'L_shed':[], 'flow_rt':[]}

    for i in range(n_samples):
        if i%250==0: print(f'Sample:{i}')
        # Solve DA market with predictions, save results
        c1 = da_market.addConstr(exp_w == w_predictions[i])
        da_market.optimize()
        
        DA_cost.append(da_market.ObjVal)
        da_solutions['p'].append(p_G.X)
        da_solutions['flow_da'].append(flow_da.X)
        
        #Solve RT market with actual wind realization, store results
        c2 = rt_prob.addConstr(w_rt == w_actual[i])
        c3 = rt_prob.addConstr(p_DA == p_G.X)
        
        rt_prob.optimize()
        
        RT_cost.append(rt_prob.ObjVal)
        
        rt_solutions['r_up'].append(r_up.X)
        rt_solutions['r_down'].append(r_down.X)
        rt_solutions['L_shed'].append(L_shed.X)
        rt_solutions['flow_rt'].append(flow_rt.X)        

        da_market.remove(c1)
        for constr in [c2,c3]: rt_prob.remove(constr)
        
        if i%10==0:        
            try:
                assert((flow_da.X.T<=grid['Line_Capacity']+.001).all())
                assert((flow_da.X.T>=-grid['Line_Capacity']-.001).all())
            except:
                print('Infeasible flows')
                
    # turn lists into arrays
    for k in da_solutions.keys(): da_solutions[k] = np.array(da_solutions[k])
    for k in rt_solutions.keys(): rt_solutions[k] = np.array(rt_solutions[k])
    
    return np.array(DA_cost), np.array(RT_cost), da_solutions, rt_solutions
    
def dc_opf(grid, demands, horizon = 1, network = True, plot = False, verbose = 0, return_ave_cpu = False):
    ''' Clears the forward (DA) market, returns the solutions in dictionary. 
        Creates the problem once in GUROBI, solves for the length of data using horizon step.
        - grid: dictionary with the details of the network
        - demands: net load demands at each node
        - network: if True, solves a DC-OPF, else solves an Economic Dispatch problem
        - horizon: the solution horizon (24 for solving the DA horizon)
        - return_ave_cpu: estimates average cpu time to solve one instance
        - verbose: if ~0, prints GUROBI output
        - plot: if True, creates some plots for check '''
         
    n_samples = int(len(demands)/horizon)

    # Declare model parameters and variables
    m = gp.Model()
    m.setParam('OutputFlag', verbose)
    # Parameters
    Pmax = grid['Pmax']
    node_G = grid['node_G']
    node_L = grid['node_L']
    Cost = grid['Cost']
    PTDF = grid['PTDF']
    VoLL = grid['VOLL']
    VoWS = grid['gshed']

    # DA Variables
    p_G = m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    slack_u = m.addMVar((grid['n_loads'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0, ub=0, name = 'slack_up')
    slack_d = m.addMVar((grid['n_loads'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0, ub=0, name = 'slack_down')
    
    flow_da = m.addMVar((grid['n_lines'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    theta_da = m.addMVar((grid['n_nodes'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
    # DA variables for uncertain parameters
    node_net_forecast_i = m.addMVar((grid['n_loads'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'node demand pred')

    # Store solutions in dict
    Det_solutions = {'p': [], 'flow_da': [], 'theta_da': [], 's_up':[], 's_down':[]}

    #### Problem Constraints
    
    #generator technical limits
    m.addConstrs( p_G[:,t] <= Pmax.reshape(-1) for t in range(horizon))

    if network == True:
        node_inj = m.addMVar((grid['n_nodes'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

        m.addConstrs(node_inj[:,t] == (node_G@p_G[:,t] - node_L@node_net_forecast_i[:,t] - node_L@slack_d[:,t] 
                                           + node_L@slack_u[:,t]) for t in range(horizon))

        m.addConstrs(flow_da[:,t] == PTDF@node_inj[:,t] for t in range(horizon))            
        #m.addConstrs(node_inj[:,t] == grid['A'].T@flow_da[:,t] for t in range(horizon))            
        #m.addConstrs(flow_da[:,t] == np.linalg.pinv(grid['A'].T)@node_inj[:,t] for t in range(horizon))
        
        m.addConstrs(flow_da[:,t] <= grid['Line_Capacity'].reshape(-1) for t in range(horizon))
        m.addConstrs(flow_da[:,t] >= -grid['Line_Capacity'].reshape(-1) for t in range(horizon))
        
    # Node balance for t for DC-OPF
    m.addConstrs( p_G[:,t].sum() + slack_u[:,t].sum() - slack_d[:,t].sum() == node_net_forecast_i[:,t].sum() for t in range(horizon))

    # DA cost for specific day/ expression
    DA_cost = sum([Cost@p_G[:,t] + slack_u[:,t].sum()*VoLL + slack_d[:,t].sum()*VoWS for t in range(horizon)]) 
    
    # Loop over days, optimize each day
    ave_cpu_time = 0
    for i in range(n_samples):
        if i%500==0:
            print('Sample: ', i)
            
        # demand is in MW (parameter that varies)
        c1 = m.addConstrs(node_net_forecast_i[:,t] == demands[i*horizon:(i+1)*horizon].T[:,t] for t in range(horizon)) 
                         
        # Set objective and solve
        m.setObjective(DA_cost, gp.GRB.MINIMIZE)                    
        m.optimize()
        ave_cpu_time += m.runtime/n_samples
        #print(m.ObjVal)
        # sanity check 
        if plot:
            if i%10==0:
                plt.plot(p_G.X.T.sum(1), label='p_Gen')
                plt.plot(p_G.X.T.sum(1) + slack_d.X.T.sum(1), '--', label='p_Gen+Slack')
                plt.plot(node_net_forecast_i.X.T.sum(1), 'o', color='black', label='Net Forecast')
                plt.legend()
                plt.show()
        if i%10==0:
            try:
                assert((flow_da.X.T<=grid['Line_Capacity']+.001).all())
                assert((flow_da.X.T>=-grid['Line_Capacity']-.001).all())
            except:
                print('Infeasible flows')
        # append solutions
        Det_solutions['p'].append(p_G.X)
        Det_solutions['s_up'].append(slack_u.X)
        Det_solutions['s_down'].append(slack_d.X)
        Det_solutions['flow_da'].append(flow_da.X)
        Det_solutions['theta_da'].append(theta_da.X)
            
        # remove constraints with uncertain parameters, reset solution
        for cosntr in [c1]:
            m.remove(cosntr)
        m.reset()

    print(ave_cpu_time)
    if return_ave_cpu:
        return Det_solutions, ave_cpu_time
    else:
        return Det_solutions

def projection_dc_opf(pred_prescr, grid, demands, horizon = 1, network = True, plot = False, verbose = 0, 
                      dist = 'L2', return_cpu_time = False, tight_equality = True):
    ''' Projection step on the feasible set for DC-OPF
        - pred_prescr: model predictions
        - grid: dictionary with the network
        - demands: net load observations
        - dist: l2 or l1 projection
        - network: if True, solves a DC-OPF, else solves an Economic Dispatch formulation
        - horizon: the solution horizon (24 for solving the DA horizon)'''
    
    n_samples = int(len(demands)/horizon)

    # Declare model parameters and variables
    m = gp.Model()
    m.setParam('OutputFlag', verbose)
    # Parameters
    Pmax = grid['Pmax']
    node_G = grid['node_G']
    node_L = grid['node_L']
    Cost = grid['Cost']
    PTDF = grid['PTDF']
    VoLL = grid['VOLL']
    VoWS = grid['gshed']

    # DA Variables
    p_G = m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    
    flow_da = m.addMVar((grid['n_lines'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    theta_da = m.addMVar((grid['n_nodes'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
    error = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    abs_error = m.addMVar(grid['n_unit'], vtype = gp.GRB.CONTINUOUS, lb = 0)
    
    # DA variables for uncertain parameters
    node_net_forecast_i = m.addMVar((grid['n_loads'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'node demand pred')

    forecast_pG_i = m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY) #!!!!!! care for lower bound here
    
    # Store solutions in dict
    Det_solutions = {'p': [], 'flow_da': [], 'theta_da': [], 's_up':[], 's_down':[]}

    #### Problem Constraints
    
    #generator technical limits
    m.addConstrs( p_G[:,t] <= Pmax.reshape(-1) for t in range(horizon))

    if network == True:
        node_inj = m.addMVar((grid['n_nodes'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

        m.addConstrs(node_inj[:,t] == (node_G@p_G[:,t] - node_L@node_net_forecast_i[:,t]) for t in range(horizon))

        m.addConstrs(flow_da[:,t] == PTDF@node_inj[:,t] for t in range(horizon))            
        #m.addConstrs(node_inj[:,t] == grid['A'].T@flow_da[:,t] for t in range(horizon))            
        #m.addConstrs(flow_da[:,t] == np.linalg.pinv(grid['A'].T)@node_inj[:,t] for t in range(horizon))
        
        m.addConstrs(flow_da[:,t] <= grid['Line_Capacity'].reshape(-1) for t in range(horizon))
        m.addConstrs(flow_da[:,t] >= -grid['Line_Capacity'].reshape(-1) for t in range(horizon))
        
    # Node balance for t for DC-OPF
    #m.addConstrs( p_G[:,t].sum() == node_net_forecast_i[:,t].sum() for t in range(horizon))

    # Objective: cost of projection to feasible set
    m.addConstr( error == (p_G - forecast_pG_i)[:,0])
    #m.addConstr( abs_error == gp.abs_(error))

    for g in range(grid['n_unit']):
         m.addConstr( abs_error[g] == gp.abs_(error[g]))

    #    m.addGenConstrAbs(abs_error[g], error[g])

    #Proj_cost = (p_G - forecast_pG_i)[0]@(p_G - forecast_pG_i)[0]
    if tight_equality:
        m.addConstrs( p_G[:,t].sum() == node_net_forecast_i[:,t].sum() for t in range(horizon))
    else:
        m.addConstrs( p_G[:,t].sum() >= node_net_forecast_i[:,t].sum() for t in range(horizon))
        

    #Proj_cost = (p_G - forecast_pG_i)[0]@(p_G - forecast_pG_i)[0]
    #Proj_cost = error@error
    #Proj_cost = abs_error.sum()
    
    # DA cost for specific day/ expression
    #DA_cost = sum([Cost@p_G[:,t] + slack_u[:,t].sum()*VoLL + slack_d[:,t].sum()*VoWS for t in range(horizon)]) 
    
    # Loop over days, optimize each day
    ave_cpu_cost = 0
    for i in range(n_samples):
        if i%500==0:
            print('Sample: ', i)
            
        # demand is in MW (parameter that varies)
        c1 = m.addConstrs(node_net_forecast_i[:,t] == demands[i*horizon:(i+1)*horizon].T[:,t] for t in range(horizon)) 
        c2 = m.addConstrs(forecast_pG_i[:,t] == pred_prescr[i*horizon:(i+1)*horizon].T[:,t] for t in range(horizon)) 
                         
        # Set objective and solve
        if dist == 'L2':
            m.setObjective( error@error, gp.GRB.MINIMIZE)                    
        elif dist == 'L1':
            m.setObjective( abs_error.sum(), gp.GRB.MINIMIZE)                    
    
        m.optimize()
        
        ave_cpu_cost += m.runtime/n_samples
        #print(m.ObjVal)
        # sanity check 
        if plot:
            if i%10==0:
                plt.plot(p_G.X.T.sum(1), label='p_Gen')
                #plt.plot(p_G.X.T.sum(1) + slack_d.X.T.sum(1), '--', label='p_Gen+Slack')
                plt.plot(node_net_forecast_i.X.T.sum(1), 'o', color='black', label='Net Forecast')
                plt.legend()
                plt.show()
        if i%10==0:
            try:
                assert((flow_da.X.T<=grid['Line_Capacity']+.001).all())
                assert((flow_da.X.T>=-grid['Line_Capacity']-.001).all())
            except:
                print('Infeasible flows')
        # append solutions
        Det_solutions['p'].append(p_G.X)
        #Det_solutions['s_up'].append(slack_u.X)
        #Det_solutions['s_down'].append(slack_d.X)
        Det_solutions['flow_da'].append(flow_da.X)
        Det_solutions['theta_da'].append(theta_da.X)
            
        # remove constraints with uncertain parameters, reset solution
        for cosntr in [c1, c2]:
            m.remove(cosntr)
    if return_cpu_time:
        return Det_solutions, ave_cpu_cost
    else:   
        return Det_solutions

def dc_opf_model(grid, demands, horizon = 1, network = True, plot = False, verbose = 0):
    ''' Creates a model to solve the DC-OPF problem, returns model in the main script
        - grid: dictionary with the details of the network
        - demands: net load demands at each node
        - network: if True, solves a DC-OPF, else solves an Economic Dispatch problem
        - horizon: the solution horizon (24 for solving the DA horizon)
        - return_ave_cpu: estimates average cpu time to solve one instance
        - verbose: if ~0, prints GUROBI output
        - plot: if True, creates some plots for check '''
            
    n_samples = int(len(demands)/horizon)

    # Declare model parameters and variables
    m = gp.Model()
    m.setParam('OutputFlag', verbose)
    # Parameters
    Pmax = grid['Pmax']
    node_G = grid['node_G']
    node_L = grid['node_L']
    Cost = grid['Cost']
    PTDF = grid['PTDF']

    # DA Variables
    p_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    
    # DA variables for uncertain parameters
    demand_i = m.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'node demand pred')

    #### Problem Constraints
    
    #generator technical limits
    m.addConstr( p_G <= Pmax.reshape(-1))

    if network == True:
        #node_inj = m.addMVar((grid['n_nodes']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

        #m.addConstr(node_inj == node_G@p_G - node_L@demand_i )
        m.addConstr(PTDF@(node_G@p_G - node_L@demand_i) <= grid['Line_Capacity'].reshape(-1))
        m.addConstr(PTDF@(node_G@p_G - node_L@demand_i) >= -grid['Line_Capacity'].reshape(-1))
        
    # Node balance for t for DC-OPF
    m.addConstr( p_G.sum() == demand_i.sum() )
    
    m.setObjective(Cost@p_G, gp.GRB.MINIMIZE)                    
    
    m._vars = {}
    m._vars['node_d_i'] = demand_i
    m._vars['p_G'] = p_G

    return m

def evaluate_prescriptions(pred_prescr, actual_gen, grid, model_name = None):
    ''' Estimate mean increase in cost for out-of-sample predictions and makes some plots
        - pred_prescr: model output
        - actual_gen: ground truth solution from GUROBI
        - grid: dictionary with grid details
        - model_name (optional): added in plot title'''
    # prediction error on generation setpoints
    plt.bar(np.arange(grid['n_unit']), 100*np.abs(pred_prescr[:,:grid['n_unit']]-actual_gen).mean(0)/grid['Pmax'])
    plt.ylabel('MAE (%)')
    plt.xlabel('Generators')
    plt.title('Gen. setpoint mismatch')
    plt.show()
    
    print('Deviation from gen. setpoint')
    print('MAE: ', np.abs(pred_prescr[:,:grid['n_unit']]-actual_gen).mean())
    print('RMSE: ', np.sqrt(np.square(pred_prescr[:,:grid['n_unit']]-actual_gen).mean() ) )
    
    print('Aggregated total error:')
    
    #(np.abs(bn_prescriptions[:,:grid['n_unit']]-g_opt_test).mean(0)/grid['Pmax']).mean()
    aggr_mae = (np.abs(pred_prescr[:,:grid['n_unit']]-actual_gen).mean(0)/grid['Pmax']).mean()
    print('MAE: ',  aggr_mae)
    
    #print('MAE: ', np.abs(pred_prescr[:,:grid['n_unit']].sum(1)-actual_gen.sum(1)).mean())
    #print('RMSE: ', np.sqrt(np.square(pred_prescr[:,:grid['n_unit']].sum(1)-actual_gen.sum(1)).mean() ) )
    
    # suboptimality of prescriptions
    opt_cost = actual_gen@grid['Cost']
    prescription_cost = pred_prescr[:,:grid['n_unit']]@grid['Cost']
    ave_cost = 100*((prescription_cost - opt_cost)/opt_cost).mean()
    std = 100*((prescription_cost - opt_cost)/opt_cost).std()
    
    print('Suboptimality Gap-mean (%): '+str(ave_cost))
    
    fig,ax = plt.subplots()
    plt.hist(100*(prescription_cost - opt_cost)/opt_cost, label = '$v(x)$', 
             bins = 50)
    plt.xlabel('Percentage (%)')
    plt.ylabel('frequency')
    plt.title(f'{model_name}: Suboptimality gap')
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (ave_cost, ),
        r'$\sigma=%.2f$' % (std, )))
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax.text(0.8, 0.95, textstr, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=props)        
    plt.show()
    ave_cost = 100*(prescription_cost - opt_cost)/opt_cost
    return ave_cost.mean().round(2), aggr_mae.round(2)

def grid_dict(path, save = False):
    ''' reads .m file with matpowercaseframes, returns dictionary with problem matrices'''

    matgrid = CaseFrames(path)
    # set cardinalities
    gen_mask = matgrid.gen.PMAX > 0
        
    num_nodes = len(matgrid.bus)
    num_lines = len(matgrid.branch)
    
    num_gen = len(matgrid.gen[gen_mask]) 
    num_load = len(matgrid.bus)  # assume demand at each node

    # Construct incidence matrix
    A = np.zeros((num_lines, num_nodes))
    
    for l in range(num_lines):
        temp_line = matgrid.branch.iloc[l]
        #A[l, temp_line['F_BUS'].astype(int)-1] = 1
        #A[l, temp_line['T_BUS'].astype(int)-1] = -1
        A[l, np.where(matgrid.bus.BUS_I == temp_line['F_BUS'])[0]] = 1
        A[l, np.where(matgrid.bus.BUS_I == temp_line['T_BUS'])[0]] = -1
        
    # Construct diagonal reactance matrix
    react = 1/matgrid.branch['BR_X'].values
    b_diag = np.diag(react)
    
    # Bus susceptance matrix
    B_susc = A.T@b_diag@A
    
    B_line = b_diag@A
    B_inv = np.zeros(B_susc.shape)
    B_inv[1:,1:] = np.linalg.inv(B_susc[1:,1:])
    PTDF = B_line@B_inv
    
    node_G = np.zeros((num_nodes, num_gen))
    #print(matgrid.gen)
    for i in range(len(matgrid.gen[gen_mask])):
        node_G[np.where(matgrid.bus.BUS_I == matgrid.gen[gen_mask].GEN_BUS.iloc[i])[0], i] = 1
        
    node_L = np.diag(np.ones(num_nodes))
    
    node_demand = matgrid.bus.PD.values
    Line_cap = matgrid.branch.RATE_A.values
    
    grid = {}
    grid['Pd'] = matgrid.bus['PD'].values
    grid['Pmax'] = matgrid.gen['PMAX'].values[gen_mask]
    grid['Pmin'] = matgrid.gen['PMIN'].values[gen_mask]
    grid['Cost'] = matgrid.gencost['COST_1'].values[gen_mask]
    
    grid['Line_Capacity'] = Line_cap
    grid['node_G'] = node_G
    grid['node_L'] = node_L
    grid['B_susc'] = B_susc
    grid['A'] = A
    grid['b_diag'] = b_diag
    grid['B_line'] = B_line
    grid['PTDF'] = PTDF
    
    # Cardinality of sets
    grid['n_nodes'] = num_nodes
    grid['n_lines'] = num_lines
    grid['n_unit'] = num_gen
    grid['n_loads'] = num_load
    
    #Other parameters set by user
    grid['VOLL'] = 500   #Value of Lost Load
    grid['VOWS'] = 35   #Value of wind spillage
    grid['gshed'] = 200   #Value of wind spillage
    
    grid['B_line'] = grid['b_diag']@grid['A']
    B_inv = np.zeros(grid['B_susc'].shape)
    B_inv[1:,1:] = np.linalg.inv(grid['B_susc'][1:,1:])
    grid['PTDF'] = grid['B_line']@B_inv
    
    #if save:  
    #    pickle.dump(grid, open(cd+'\\data\\'+network.split('.')[0]+'.sav', 'wb'))
    return grid

def setpoint_limit_check(prescriptions, grid):
    'Function to check if generation setpoints are feasible'
    try:
        assert((prescriptions[:,:grid['n_unit']] <= grid['Pmax'].reshape(1,-1)+3).all())
        assert((prescriptions[:,:grid['n_unit']] >= np.zeros((1,grid['n_unit']))-3).all())
        print('Setpoints are feasible')
    except:
        print('Generator constraint violation')
        print('Upper bound %: ', 100*(prescriptions[:,:grid['n_unit']] > grid['Pmax'].reshape(1,-1)+1e-3).sum()/prescriptions[:,:grid['n_unit']].size)
        print('Lower bound %: ', 100*(prescriptions[:,:grid['n_unit']] < 0).sum()/prescriptions[:,:grid['n_unit']].size)

def lineflow_limit_check(line_flows, grid):
    'Function to check if line flows are feasible'
    try:
        assert((line_flows <= grid['Line_Capacity'].reshape(1,-1)+1e-2).all())
        assert((line_flows >= -grid['Line_Capacity'].reshape(1,-1)-1e-2).all())
        print('Line flows are feasible')
    except:
        line_flows <= grid['Line_Capacity'].reshape(1,-1)
        
        print('Line flow constraint violation')
        print('Upper bound %: ', 100*(line_flows <= grid['Line_Capacity'].reshape(1,-1)).sum()/line_flows.size)
        print('Lower bound %: ', 100*(line_flows >= -grid['Line_Capacity'].reshape(1,-1)).sum()/line_flows.size)
