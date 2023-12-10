# -*- coding: utf-8 -*-
"""
Deterministic market clearing

@author: a.stratigakos
"""

import cvxpy as cp
import numpy as np
import gurobipy as gp
import time 
from sklearn.cluster import KMeans

def opt_problem(Y, grid, config, weights = None, capacity = 100, bus = 0, prescribe = False, 
                parallel = False, scenario_reduction = False, num_reduced_scen = 50, network = True):
    
    ''' Function that solves the Sample Average Approximation of the optimization problem
        Input: Uncertain quantities, all other parameters required by the solution
        Any parameters regarding the Gurobi optimizer, such minimum MIP Gap, maximum Time Limit etc., 
        should be inserted in the model before optimize()
        Inputs: Y, kwargs from the original problem'''
            
    if type(weights) != np.ndarray:
        if weights == None:
            weights = np.ones(len(Y))/len(Y)

    if scenario_reduction == False:
        wind_samples = capacity*Y
    else:
        if len(Y) <= num_reduced_scen:
            # Insufficient samples, do not apply scenario reduction
            wind_samples = capacity*Y
        else:
            
            #indices = np.random.choice(np.arange(len(Y)), size = num_reduced_scen, replace = False)
            #wind_samples = capacity*Y[indices]
            
            # Apply scenario reduction
            #Fit k-means, assign labels, find distances and medioid scenarios
            kmeans = KMeans(n_clusters = num_reduced_scen, random_state=0, max_iter=500).fit(Y)
            #labels = kmeans.predict(Y)
            distances = kmeans.transform(Y)
            med_Y = np.zeros((num_reduced_scen, config['horizon']))
            for c in range(num_reduced_scen):
                med_Y[c] = Y[np.argsort(distances[:,c])[0]]
            
            wind_samples = capacity*med_Y
            
    predicted_y = capacity*weights@Y
    num_or_samples = len(Y)    #Initial number of samples/scenarios

    Nscen = len(wind_samples)    #Number of scenarios used in the optimization algorithm
    
    # Solve DA problem
    #n_samples = int(len(demands)/horizon)

    # Declare model parameters and variables
    m = gp.Model()
    m.setParam('OutputFlag', 0)
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
        
    if prescribe == False:
        
        #return np.square(Y-Y.mean()).sum(), Y.mean()
    
        ###### DA constraints
        # DA Variables
        p_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
        slack_u = m.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'slack')
        flow_da = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        #theta_da = m.addMVar((grid['n_nodes']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        node_inj = m.addMVar((grid['n_nodes']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        
        exp_w = m.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        
        #### Problem Constraints
        
        #generator technical limits
        m.addConstr( p_G <= Pmax.reshape(-1))
        m.addConstr( exp_w == predicted_y)
        
        if network:
            # Network constraints
            m.addConstr(node_inj == (node_G@p_G + node_Wind@exp_w - node_L@grid['Pd'] + node_L@slack_u))
            m.addConstr(flow_da == PTDF@node_inj )                    
            m.addConstr(flow_da <= grid['Line_Capacity'].reshape(-1))
            m.addConstr(flow_da >= -grid['Line_Capacity'].reshape(-1))
            
        # Node balance for t for DC-OPF
        m.addConstr( p_G.sum() + slack_u.sum() + exp_w.sum() == grid['Pd'].sum())

        # DA cost for specific day/ expression        
        m.setObjective(Cost@p_G + slack_u.sum()*VoLL, gp.GRB.MINIMIZE)                    
        m.optimize()
        
        DA_cost = m.ObjVal
        solution = {'p': p_G.X, 'flow':flow_da.X}
        
        ############## Real-time balancing problem (after solving the DA problem)
        #RT Variables
        rt_prob = gp.Model()
        rt_prob.setParam('OutputFlag', 0)

        ###### DA constraints
        # DA Variables
        r_up = rt_prob.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
        r_down = rt_prob.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
        L_shed = rt_prob.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0)
        flow_rt = rt_prob.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        #theta_rt = rt_prob.addMVar((grid['n_nodes']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        node_inj_rt = rt_prob.addMVar((grid['n_nodes']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        w_rt = rt_prob.addMVar((1), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        
        # Technical limits
        rt_prob.addConstr( r_up <= -p_G.X + Pmax.reshape(-1))
        rt_prob.addConstr( r_up <= grid['R_u_max'].reshape(-1))
        
        rt_prob.addConstr( r_down <= p_G.X)
        rt_prob.addConstr( r_down <= grid['R_d_max'].reshape(-1))

        rt_prob.addConstr( L_shed <= grid['Pd'].reshape(-1))
        
        #rt_prob.addConstrs( w_rt[i] == 0 for i in range(grid['n_nodes']) if i != bus)
        
        if network:
            # Network constraints
            rt_prob.addConstr(node_inj_rt == (node_G@(r_up-r_down + p_G.X) \
                                              + node_Wind@(w_rt) - node_L@(grid['Pd']-L_shed) ))
            rt_prob.addConstr(flow_rt == PTDF@node_inj_rt)                    
            rt_prob.addConstr(flow_rt <= grid['Line_Capacity'].reshape(-1))
            rt_prob.addConstr(flow_rt >= -grid['Line_Capacity'].reshape(-1))
        
        rt_prob.addConstr(+ r_up.sum() -r_down.sum() + w_rt.sum() -exp_w.X.sum() + L_shed.sum() == 0)
        exp_RT_cost = 0
                    
        for scen in range(Nscen):      
            c1 = rt_prob.addConstr( w_rt ==  wind_samples[scen])
                                                     
            # Set objective and solve
            RT_cost_i = grid['C_up']@r_up - grid['C_down']@r_down + VoLL*L_shed.sum() 
        
#            RT_cost = RT_cost + 1/Nscen*cp.sum( grid['Cost_reg_up']@r_up[scen] 
#                                               -grid['Cost_reg_down']@r_down[scen]\
#                                                + grid['VOLL']*cp.sum(L_shed[scen],axis=0) + grid['gshed']*cp.sum(G_shed[scen],axis=0) ) 

            rt_prob.setObjective(RT_cost_i, gp.GRB.MINIMIZE)                    
            rt_prob.optimize()
            
            exp_RT_cost += (1/Nscen)*((grid['C_up']-grid['Cost'])@r_up.X + (grid['Cost']-grid['C_down'])@r_down.X)

            rt_prob.remove(c1)
            #rt_prob.reset()
            
        try:
            #!!! multiply output with initial num_samples, since we are comparing aggregated costs for splitting nodes
            return num_or_samples*(exp_RT_cost) , solution
        except:

            return 10e10, []        