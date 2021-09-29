# -*- coding: utf-8 -*-
"""
BESS arbitrage

@author: a.stratigakos
"""

import gurobipy as gp
import numpy as np
import scipy.sparse as sp

def opt_problem(Y, k, c_in, c_out, B_max, B_min, in_eff, out_eff, z0, gamma, epsilon,
                weights = None, prescribe = False, parallel = False):
    
    ''' Function that solves the Sample Average Approximation of the optimization problem
        Input: Uncertain quantities, all other parameters required by the solution
        Any parameters regarding the Gurobi optimizer, such minimum MIP Gap, maximum Time Limit etc., 
        should be inserted in the model before optimize()
        Inputs: Y, kwargs from the original problem'''
    
    
    if type(weights) != np.ndarray:
        if weights == None:
            weights = np.ones(len(Y))/len(Y)
    W_diag = sp.diags(weights)
    
    if prescribe == False:
        
        #Sample Average Approximation
        n_days = len(Y)
        horizon = Y.shape[1]
        m = gp.Model()
    
        m.setParam('OutputFlag', 0)
    
        #Problem variables
        z_state = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = B_min, ub=B_max, name = 'state')
        z_charge = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0, ub = c_in, name = 'charge')
        z_discharge = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0, ub = c_out, name = 'discharge')
        
        # Auxiliary variables for vector operations
        net_diff = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'net diff')
        deviation = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0)
        t = m.addMVar(1 , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph')

        Profit = m.addMVar(n_days , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'Prescription Cost')
        
        # State transition
        m.addConstr( z_state[1:] == z_state[0:-1] + in_eff*z_charge[0:-1] - out_eff*z_discharge[0:-1])
        m.addConstr( z_state[0] == z0)
        m.addConstr( z_state[-1] + in_eff*z_charge[-1] - out_eff*z_discharge[-1] == z0)
        # Aux variables
        m.addConstr( net_diff == z_discharge - z_charge)
        m.addConstr( deviation == z_state - z0)
        m.addConstr( Profit == Y@net_diff )

        m.addConstr( t >= -Profit.sum()/n_days + + gamma*(deviation@deviation) \
                         + epsilon*(z_charge@z_charge) + epsilon*(z_discharge@z_discharge))
        
        m.setObjective( t.sum(), gp.GRB.MINIMIZE)
        m.optimize()
        
        bess_actions = {}
        bess_actions['z_state'] = z_state.X
        bess_actions['z_discharge'] = z_discharge.X
        bess_actions['z_charge'] = z_charge.X
        
        return n_days*m.objVal, bess_actions

    else:
        #Sample Average Approximation
        n_days = len(Y)
        horizon = Y.shape[1]
        m = gp.Model()
    
        m.setParam('OutputFlag', 0)
    
        #Problem variables
        z_state = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = B_min, ub=B_max, name = 'state')
        z_charge = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0, ub = c_in, name = 'charge')
        z_discharge = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0, ub = c_out, name = 'discharge')

        # Auxiliary variables for vector operations
        net_diff = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'net diff')
        deviation = m.addMVar(horizon , vtype = gp.GRB.CONTINUOUS, lb = 0)
        t = m.addMVar(1 , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph')
        Profit = m.addMVar(n_days , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'Prescription Cost')
        w_Profit = m.addMVar(n_days , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'Prescription Cost')

        # Transition constraints
        m.addConstr( z_state[1:] == z_state[0:-1] + in_eff*z_charge[0:-1] - out_eff*z_discharge[0:-1])
        m.addConstr( z_state[0] == z0)
        m.addConstr( z_state[-1] + in_eff*z_charge[-1] - out_eff*z_discharge[-1] == z0)

        # Aux variables
        m.addConstr( net_diff == z_discharge - z_charge)
        m.addConstr( deviation == z_state - z0)
        m.addConstr( Profit == Y@net_diff )
        m.addConstr( w_Profit == W_diag@Profit )
        
        m.addConstr( t >= -w_Profit.sum() + + gamma*(deviation@deviation) \
                         + epsilon*(z_charge@z_charge) + epsilon*(z_discharge@z_discharge))
       
        
        m.setObjective( t.sum(), gp.GRB.MINIMIZE)
        m.optimize()
        
        bess_actions = {}
        bess_actions['z_state'] = z_state.X
        bess_actions['z_discharge'] = z_discharge.X
        bess_actions['z_charge'] = z_charge.X
        
        return m.objVal, bess_actions