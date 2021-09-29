# -*- coding: utf-8 -*-
"""
Two-stage problem, uncertain demand

@author: a.stratigakos
"""

import cvxpy as cp
import numpy as np
from sklearn.cluster import KMeans

def opt_problem(Y, grid, config, weights = None, prescribe = False, parallel = False, scenario_reduction = False, num_reduced_scen=10):
    
    ''' Function that solves the Sample Average Approximation of the optimization problem
        Input: Uncertain quantities, all other parameters required by the solution
        Any parameters regarding the Gurobi optimizer, such minimum MIP Gap, maximum Time Limit etc., 
        should be inserted in the model before optimize()
        Inputs: Y, kwargs from the original problem'''
    
    
    if type(weights) != np.ndarray:
        if weights == None:
            weights = np.ones(len(Y))/len(Y)
    horizon = config['horizon']
    
    predicted_y = (config['peak_load']*weights@Y).reshape(1,horizon) # Expected Y
    num_samples = len(Y)    #Initial number of samples/scenarios

    if scenario_reduction == False:
        Demand_Samples = config['peak_load']*Y
    else:
        if len(Y) <= num_reduced_scen:
            # Insufficient samples, do not apply scenario reduction
            Demand_Samples = config['peak_load']*Y
        else:
            # Apply scenario reduction
            #Fit k-means, assign labels, find distances and medioid scenarios
            kmeans = KMeans(n_clusters = num_reduced_scen, random_state=0).fit(Y)
            #labels = kmeans.predict(Y)
            distances = kmeans.transform(Y)
            med_Y = np.zeros((num_reduced_scen, config['horizon']))
            for c in range(num_reduced_scen):
                med_Y[c] = Y[np.argsort(distances[:,c])[0]]
            Demand_Samples = config['peak_load']*med_Y

    Nscen = len(Demand_Samples)    #Number of scenarios used in the optimization algorithm
    Node_demand_scenarios = np.zeros((grid['n_loads'], horizon, Nscen))    
    for scen in range(Nscen):
        Node_demand_scenarios[:,:,scen] = np.outer(grid['node_demand_percentage'], Demand_Samples[scen,:].T)
    Node_demand_expected = np.outer(grid['node_demand_percentage'], predicted_y.T)

    # Sample Average Approximation
    
    # Problem variables
    n_unit = grid['n_unit']
    n_lines = grid['n_lines']
    n_nodes = grid['n_nodes']
    n_loads = grid['n_loads']
    
    #DA Variables
    p_G = cp.Variable((n_unit, horizon))
    R_up = cp.Variable((n_unit, horizon))
    R_down = cp.Variable((n_unit, horizon))
    flow_da = cp.Variable((n_lines,horizon))
    theta_da = cp.Variable((n_nodes, horizon))
    Demand_slack = cp.Variable((n_loads, horizon))
    
    if prescribe == False:
        
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
        Constraints_DA += [ grid['node_G']@p_G + grid['node_L']@(Demand_slack-Node_demand_expected ) == grid['B']@theta_da]
        DA_cost = cp.sum(grid['Cost']@p_G) + grid['VOLL']*cp.sum(Demand_slack)
        
                
        # Actually only care about RT costs not the DA costs (these just depend on demand)
        prob = cp.Problem(cp.Minimize(DA_cost) , Constraints_DA)
        prob.solve( solver = 'GUROBI', verbose = False)
        
        solution = {'p': p_G.value, 'flow':flow_da.value, 'theta': theta_da.value, 
                    'R_up': grid['R_up_max'].repeat(horizon,axis=1), 
                    'R_down': grid['R_down_max'].repeat(horizon,axis=1)}
        
        ############## Real-time balancing problem
        
        #RT Variables
        r_up= [cp.Variable((n_unit, horizon)) for scen in range(Nscen)]
        r_down= [cp.Variable((n_unit, horizon)) for scen in range(Nscen)]
        L_shed= [cp.Variable((n_loads,horizon)) for scen in range(Nscen)]
        flow_rt= [cp.Variable((n_lines,horizon)) for scen in range(Nscen)]
        theta_rt = [cp.Variable((n_nodes, horizon)) for scen in range(Nscen)]
        G_shed = [cp.Variable((n_unit, horizon)) for scen in range(Nscen)]
        ###### RT constraints
        RT_cost = 0
        
        Constraints_RT = []
        for scen in range(Nscen):       
            # Feasbility limits 
            Constraints_RT += [ r_up[scen] <= -solution['p'] + grid['Pmax'].repeat(horizon,axis=1),
                               r_up[scen] <= solution['R_up'],      
                                 r_down[scen] <= solution['p'],
                                 r_down[scen] <= solution['R_down'],
                                 L_shed[scen] <= Node_demand_scenarios[:,:,scen], 
                                 G_shed[scen] <= solution['p'],
                                 r_up[scen] >= 0, r_down[scen] >= 0, 
                                 L_shed[scen] >= 0, G_shed[scen] >= 0]

            #RT Network flow
            Constraints_RT += [flow_rt[scen] == grid['b_diag']@grid['A']@theta_rt[scen],
                               flow_rt[scen] <= grid['Line_Capacity'].repeat(horizon,axis=1), 
                               flow_rt[scen] >= -grid['Line_Capacity'].repeat(horizon,axis=1),
                               theta_rt[scen][0,:] == 0] 
            
            #Node injections
            Constraints_RT += [ grid['node_G']@(r_up[scen]-r_down[scen]-G_shed[scen]) \
                               + grid['node_L']@(L_shed[scen]-Node_demand_scenarios[:,:,scen]+Node_demand_expected) \
                               == grid['B']@(theta_rt[scen]-solution['theta'])]
            
            RT_cost = RT_cost + 1/Nscen*cp.sum( grid['Cost_reg_up']@r_up[scen] 
                                               -grid['Cost_reg_down']@r_down[scen]\
                                                + grid['VOLL']*cp.sum(L_shed[scen],axis=0) + grid['gshed']*cp.sum(G_shed[scen],axis=0) ) 
        
        prob = cp.Problem(cp.Minimize(RT_cost) , Constraints_RT)
        prob.solve( solver = 'GUROBI', verbose = False)
        try:
            #!!! multiply output with initial num_samples, since we are comparing aggregated costs for splitting nodes
            return num_samples*(RT_cost.value), solution
        except:
            return 10e10, []        
        
    else:
        #DA Variables
        p_G = cp.Variable((n_unit, horizon))
        R_up = cp.Variable((n_unit, horizon))
        R_down = cp.Variable((n_unit, horizon))
        flow_da = cp.Variable((n_lines,horizon))
        theta_da = cp.Variable((n_nodes, horizon))
        Demand_slack = cp.Variable((n_loads, horizon))
        
        #RT Variables
        r_up= [cp.Variable((n_unit, horizon)) for scen in range(Nscen)]
        r_down= [cp.Variable((n_unit, horizon)) for scen in range(Nscen)]
        L_shed= [cp.Variable((n_loads,horizon)) for scen in range(Nscen)]
        flow_rt= [cp.Variable((n_lines,horizon)) for scen in range(Nscen)]
        theta_rt = [cp.Variable((n_nodes, horizon)) for scen in range(Nscen)]
        G_shed = [cp.Variable((n_unit, horizon)) for scen in range(Nscen)]
        
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
        Constraints_DA += [ grid['node_G']@p_G + grid['node_L']@(Demand_slack-Node_demand_expected ) == grid['B']@theta_da]
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
                                 L_shed[scen] <= Node_demand_scenarios[:,:,scen],
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
                               + grid['node_L']@(L_shed[scen]-Node_demand_scenarios[:,:,scen]+Node_demand_expected) \
                               == grid['B']@(theta_rt[scen]-theta_da)]
            
            RT_cost = RT_cost + weights[scen]*cp.sum( grid['Cost_reg_up']@r_up[scen] 
                                               - grid['Cost_reg_down']@r_down[scen]\
                                                + grid['VOLL']*cp.sum(L_shed[scen],axis=0) + grid['gshed']*cp.sum(G_shed[scen],axis=0) ) 
        
        # Actually only care about RT costs not the DA costs (these just depend on demand)
        prob = cp.Problem(cp.Minimize(DA_cost+RT_cost) , Constraints_DA+Constraints_RT)
        prob.solve( solver = 'GUROBI', verbose = False)
        
        solution = {'p': p_G.value, 'flow':flow_da.value, 'theta': theta_da.value, 
                    'R_up': grid['R_up_max'].repeat(horizon,axis=1), 
                    'R_down': grid['R_down_max'].repeat(horizon,axis=1), 
                    'LMP':-Constraints_DA[-1].dual_value}
        
        
        return (DA_cost+RT_cost).value, solution
