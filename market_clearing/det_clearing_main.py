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

import cvxpy as cp
import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer
import copy

class tensor_DA_layer(torch.nn.Module):
    def __init__(self):
        super(tensor_DA_layer, self).__init__()

    def forward(self, x, grid):
        argsort_ind = np.argsort(grid['Cost'])
        total_d = (grid['Pd'].sum() - grid['w_capacity']*x).reshape(-1)        
        cum_gen = torch.FloatTensor(np.zeros(total_d.shape))
        p_gen = torch.FloatTensor( np.zeros((len(x), grid['n_unit'])) ).requires_grad_()
        zeros_tensor = torch.FloatTensor(np.zeros(total_d.shape))

        for i, g in enumerate(argsort_ind):
            # cumulative sum of previous (i-th is always 0)            
            P_max_tensor = torch.FloatTensor(grid['Pmax'][g]*np.ones(total_d.shape))
            p_gen.detach()[:,g] = torch.min( total_d - cum_gen, P_max_tensor )
            p_gen.detach()[:,g] = torch.max( p_gen[:,g], zeros_tensor)
            cum_gen = cum_gen + p_gen[:,g]
            
        return p_gen
    
class tensor_RT_layer(torch.nn.Module):
    def __init__(self):
        super(tensor_RT_layer, self).__init__()

    def forward(self, error, p_gen, grid):
        argsort_ind_up = np.argsort(grid['C_up'])
        # downward regulation cost sorted in descending order
        argsort_ind_down = np.argsort(-grid['C_down'])

        r_up = torch.FloatTensor(np.zeros((len(error), grid['n_unit'])))
        r_down = torch.FloatTensor(np.zeros((len(error), grid['n_unit'])))
        
        mask_up = (error < 0).reshape(-1) #Error<0: expected > actual: upward regulation needed
        mask_down = (error > 0).reshape(-1) #downward regulation needed

        cum_gen_up = torch.FloatTensor(np.zeros(error.shape).reshape(-1))
        cum_gen_down = torch.FloatTensor(np.zeros(error.shape).reshape(-1))
        zeros_tensor = torch.FloatTensor( np.zeros(error.shape).reshape(-1) ) 
        for g in argsort_ind_up:
            r_up[mask_up, g] = torch.min( torch.abs(grid['w_capacity']*error[mask_up].reshape(-1)) - cum_gen_up[mask_up], grid['Pmax'][g] - p_gen[mask_up, g])
            r_up[mask_up, g] = torch.max( r_up[mask_up, g], zeros_tensor[mask_up])
            cum_gen_up[mask_up] = cum_gen_up[mask_up] + r_up[mask_up, g]
            
        for g in argsort_ind_down:
            r_down[mask_down, g] = torch.min( grid['w_capacity']*error[mask_down].reshape(-1) - cum_gen_down[mask_down], p_gen[mask_down, g])
            r_down[mask_down, g] = torch.max( r_down[mask_down, g], zeros_tensor[mask_down])
            cum_gen_down[mask_down] = cum_gen_down[mask_down] + r_down[mask_down, g]

        return r_up, r_down

class e2e_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU()):
        super(e2e_MLP, self).__init__()
        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)

        self.model = nn.Sequential(*layers)
        self.DA_layer = tensor_DA_layer()

    def forward(self, x, grid):
        # Implement the forward pass of your MLP using the layers
        x = self.model(x)
        x = self.DA_layer(x, grid)
        
        return x

    def predict(self, x):
        with torch.no_grad():            
            return self.model(x)

    def train_model(self, train_loader, val_loader, criterion, optimizer, epochs=10, patience=5):
        best_val_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch
            for inputs, labels in train_loader:
                # clear gradients
                optimizer.zero_grad()
                # forward pass
                outputs = self(inputs)

                # loss evaluation
                loss = criterion(outputs, labels)
                # backward pass
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            average_train_loss = running_loss / len(train_loader)

            val_loss = self.evaluate(val_loader, criterion)
            
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return

    def evaluate(self, data_loader, criterion):
        # evalaute loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        average_loss = total_loss / len(data_loader)
        return average_loss
    
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU()):
        super(MLP, self).__init__()
        # create sequential model
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with torch.no_grad():            
            return self.model(x).detach().numpy()

    def train_model(self, train_loader, val_loader, criterion, optimizer, epochs=10, patience=5):
        best_val_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch
            for inputs, labels in train_loader:
                # clear gradients
                optimizer.zero_grad()
                # forward pass
                outputs = self(inputs)

                # loss evaluation
                loss = criterion(outputs, labels)
                # backward pass
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            average_train_loss = running_loss / len(train_loader)

            val_loss = self.evaluate(val_loader, criterion)
            
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return

    def evaluate(self, data_loader, criterion):
        # evalaute loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        average_loss = total_loss / len(data_loader)
        return average_loss
    
def to_np(x):
    return x.detach().numpy()

def create_data_loader(X, Y, batch_size = 64, shuffle = True):
    dataset = torch.utils.data.TensorDataset(X,Y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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

def dc_opf_closed(predictions, grid):
    # closed-form solution for DC-opf (no grid)
    argsort_ind = np.argsort(grid['Cost'])
    total_d = (grid['Pd'].sum() - predictions).reshape(-1)

    cum_gen = np.zeros(total_d.shape)
    
    p_gen = np.zeros((len(predictions), grid['n_unit'], ))
    
    for g in argsort_ind:
        p_gen[:,g] = np.minimum( total_d - cum_gen, grid['Pmax'][g])
        p_gen[:,g] = np.maximum( p_gen[:,g], 0)
        
        cum_gen += p_gen[:,g]
    DA_cost = grid['Cost']@p_gen.T
    return p_gen, DA_cost

def rt_market_closed(error, p_gen, grid):
    # closed-form solution for RT balancing (no grid)

    argsort_ind_up = np.argsort(grid['C_up'])
    # downward regulation cost sorted in descending order
    argsort_ind_down = np.argsort(-grid['C_down'])

    r_up = np.zeros((len(error), grid['n_unit'], ))
    r_down = np.zeros((len(error), grid['n_unit'], ))
    
    mask_up = (error < 0).reshape(-1) #Error<0: expected > actual: upward regulation needed
    mask_down = (error > 0).reshape(-1) #downward regulation needed

    cum_gen_up = np.zeros(error.shape).reshape(-1)
    cum_gen_down = np.zeros(error.shape).reshape(-1)
        
    for g in argsort_ind_up:
        r_up[mask_up, g] = np.minimum( np.abs(error[mask_up].reshape(-1)) - cum_gen_up[mask_up], grid['Pmax'][g] - p_gen[mask_up, g])
        r_up[mask_up, g] = np.maximum( r_up[mask_up, g], 0)
        cum_gen_up[mask_up] += r_up[mask_up, g]
        
    for g in argsort_ind_down:
        r_down[mask_down, g] = np.minimum( error[mask_down].reshape(-1) - cum_gen_down[mask_down], p_gen[mask_down, g])
        r_down[mask_down, g] = np.maximum( r_down[mask_down, g], 0)
        cum_gen_down[mask_down] += r_down[mask_down, g]
    
    rt_cost = (grid['C_up']-grid['Cost'])@r_up.T + (grid['Cost'] -  grid['C_down'])@r_down.T
    return r_up, r_down, rt_cost
    
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

Cases = ['pglib_opf_case14_ieee.m', 'pglib_opf_case57_ieee.m', 'pglib_opf_case118_ieee.m']

w_bus = [13, 37, 36]
w_bus_dict = {}

w_cap = [100,600, 500]
w_cap_dict = {}

for i, case in enumerate(Cases):
    w_bus_dict[case] = w_bus[i]
    w_cap_dict[case] = w_cap[i]
#%%
for case in Cases[2:3]:
    
    case_name = case.split('.')[0]    
    matgrid = CaseFrames(pglib_path + case)
    grid = grid_dict(pglib_path + case)

grid['C_up'] = 1.8*grid['Cost']
grid['C_down'] = 0.8*grid['Cost']
grid['w_capacity'] = w_cap_dict[case]
grid['w_bus'] = w_bus_dict[case]

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


#%% DTs: Standard forecasting
'''
from sklearn.ensemble import ExtraTreesRegressor

et_model = ExtraTreesRegressor(n_estimators = 20, min_samples_leaf = 2 ).fit(trainX, trainY)
PO_pred = et_model.predict(testX).reshape(-1,1)

# DTs: Value-oriented forecasting

pt_model = EnsemblePrescriptiveTree(n_estimators = 20, Nmin = 2, max_features = 4, type_split = 'quant')
pt_model.fit(trainX.values, trainY.values, grid = grid, config = config, parallel = False, capacity = capacity, 
             bus = target_bus, scenario_reduction = True, network = False, num_reduced_scen = 50)
VO_pred = pt_model.cost_oriented_forecast(testX.values, trainX.values, trainY.values)
'''

#%% NN/Prediction-oriented training

tensor_trainX = torch.FloatTensor(trainX.values[:-1000])
tensor_trainY = torch.FloatTensor(trainY.values[:-1000])

tensor_valX = torch.FloatTensor(trainX.values[-1000:])
tensor_valY = torch.FloatTensor(trainY.values[-1000:])

tensor_testX = torch.FloatTensor(testX.values)
tensor_testY = torch.FloatTensor(testY.values)


train_data = torch.utils.data.TensorDataset(tensor_trainX, tensor_trainY)
val_data = torch.utils.data.TensorDataset(tensor_valX, tensor_valY)

patience = 25
batch_size = 264
num_epochs = 1000

train_loader = create_data_loader(tensor_trainX, tensor_trainY, batch_size = batch_size)
val_loader = create_data_loader(tensor_valX, tensor_valY, batch_size = 500)

input_size = trainX.shape[1]
output_size = trainY.shape[1]
n_hidden_layers = 2
n_nodes = 30 # per hidden layer
criterion = nn.MSELoss()

PO_NNmodel = MLP(input_size = input_size, hidden_sizes = n_hidden_layers*[n_nodes], output_size = output_size)

# initialize optimizer
optimizer = torch.optim.Adam(PO_NNmodel.parameters(), lr=1e-3)

PO_NNmodel.train_model(train_loader, val_loader, criterion, optimizer, epochs = 200, patience = 10)

PO_NN_pred = to_np(PO_NNmodel.forward(tensor_testX).clamp(0,1))
print(eval_point_pred(PO_NN_pred, testY.values))

#%% Linear model/ Prediction-oriented learning
PO_LRmodel = MLP(input_size = input_size, hidden_sizes = [], output_size = output_size)

# initialize optimizer
optimizer = torch.optim.Adam(PO_LRmodel.parameters(), lr=1e-3)

PO_LRmodel.train_model(train_loader, val_loader, criterion, optimizer, epochs = 200, patience = 10)

PO_LR_pred = to_np(PO_LRmodel.forward(tensor_testX).clamp(0,1))
print(eval_point_pred(PO_LR_pred, testY.values))

#%% NN/ Decision-focused learning
patience = 5
batch_size = 100
num_epochs = 1000
check_validation = False

DO_NNmodel = MLP(input_size = input_size, hidden_sizes = n_hidden_layers*[n_nodes], output_size = output_size)
# initialize pre-trained weights

pretrained_state_dict = PO_NNmodel.state_dict()
DO_NNmodel.load_state_dict(pretrained_state_dict)

###### DA market layer
# Variables of inner problem
p_gen = cp.Variable((grid['n_unit']))
slack_up = cp.Variable((1))
slack_down = cp.Variable((1))

cost = cp.Variable((1))    
wind_prediction = cp.Parameter((1))
constraints = [p_gen >= 0, p_gen <= grid['Pmax'].reshape(-1)] \
            + [slack_up >= 0, slack_down >= 0]\
            + [p_gen.sum() + grid['w_capacity']*wind_prediction + slack_up.sum() - slack_down.sum()== grid['Pd'].sum()]\
            + [cost == grid['Cost']@p_gen + grid['VOLL']*(slack_up.sum()+ slack_down.sum())]

objective = cp.Minimize( sum(cost) ) 
problem = cp.Problem(objective, constraints)
dcopf_layer = CvxpyLayer(problem, parameters=[wind_prediction], variables=[p_gen, slack_up, slack_down, cost] )

# append layer to NN
DO_NNmodel.model.add_module(f'dcopf_layer', dcopf_layer)

#### RT market-clearing layer
p_gen_param = cp.Parameter((grid['n_unit']))
w_pred_error = cp.Parameter((1))

r_up = cp.Variable((grid['n_unit']))
r_down = cp.Variable((grid['n_unit']))

rt_slack = cp.Variable((1))
aux = cp.Variable((1))
#rt_slack_down = cp.Variable((1))

rt_cost = cp.Variable((1))    

rt_constraints = [r_up >= 0, r_down >= 0, r_up <= 1.05*grid['Pmax'].reshape(-1)-p_gen_param, r_down <= p_gen_param]\
            + [r_up <= grid['R_u_max'], r_down <= grid['R_d_max']] \
            + [r_up.sum() - r_down.sum() + grid['w_capacity']*w_pred_error + rt_slack == 0]\
            + [rt_cost == (grid['C_up']-grid['Cost'])@r_up + (grid['Cost'] - grid['C_down'])@r_down+ 
               grid['VOLL']*aux] +[aux >= rt_slack, aux >= -rt_slack]

rt_objective = cp.Minimize( sum(rt_cost)) 
rt_problem = cp.Problem(rt_objective, rt_constraints)
rt_layer = CvxpyLayer(rt_problem, parameters=[p_gen_param, w_pred_error], 
                      variables=[r_up, r_down, rt_slack, aux, rt_cost])

Losses = []
optimizer = torch.optim.Adam(DO_NNmodel.parameters(), lr=1e-3)
best_val_loss = float('inf')

for epoch in range(num_epochs):
    
    # enable train functionality
    DO_NNmodel.train()
    running_loss = 0.0

    #train_inputs, train_labels = next(iter(train_loader))
    for train_inputs, train_labels in train_loader:
        
        # clear gradients before backward pass    
        optimizer.zero_grad()
        
        #### forward pass/ solve for DA-market
        dcopf_output = DO_NNmodel.forward(train_inputs)
        # generator setpoints
        p_hat = dcopf_output[0].clamp(0)
        # predictions
        y_pred = DO_NNmodel.model[:-1].forward(train_inputs)
        w_error_hat = train_labels - y_pred
        # solve RT market
        rt_output = rt_layer(p_hat, w_error_hat, solver_args={'max_iters':10000})
        # cost of redispatching
        loss = rt_output[-1].sum()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        #break
    
    average_train_loss = running_loss / len(train_loader)
                
    DO_NNmodel.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            
            val_dcopf_outputs = DO_NNmodel.forward(val_inputs)        
            # predictions
            val_predictions = DO_NNmodel.model[:-1].forward(val_inputs)
            # forecast error
            val_w_error_hat = val_labels - val_predictions
            # solve RT market
            val_rt_output = rt_layer(val_dcopf_outputs[0].clamp(0), val_w_error_hat)
            # cost of redispatching
            total_val_loss += val_rt_output[-1].mean()       
            #break
    ave_val_rt_loss = total_val_loss/len(val_loader)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss / len(train_loader):.4f} - Val Loss: {ave_val_rt_loss.item():.4f}")

    # Implement early stopping
    if ave_val_rt_loss < best_val_loss:
        best_val_loss = ave_val_rt_loss
        best_model_weights = copy.deepcopy(DO_NNmodel.state_dict())
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break
        
DO_NNmodel.load_state_dict(best_model_weights)

# generate value-oriented predictions/ skip last layer of the model
DO_NN_pred = to_np(DO_NNmodel.model[:-1].forward(tensor_testX).clamp(0,1))
print(eval_point_pred(DO_NN_pred, testY))

#%% LR Decision-focused learning
patience = 5
batch_size = 100
num_epochs = 1000
check_validation = False

DO_LRmodel = MLP(input_size = input_size, hidden_sizes = [], output_size = output_size)
# initialize pre-trained weights

pretrained_state_dict = PO_LRmodel.state_dict()
DO_LRmodel.load_state_dict(pretrained_state_dict)

# append layer to NN
DO_LRmodel.model.add_module(f'dcopf_layer', dcopf_layer)

Losses = []
optimizer = torch.optim.Adam(DO_LRmodel.parameters(), lr=1e-3)
best_val_loss = float('inf')

for epoch in range(num_epochs):
    
    # enable train functionality
    DO_LRmodel.train()
    running_loss = 0.0

    #train_inputs, train_labels = next(iter(train_loader))
    for train_inputs, train_labels in train_loader:
        
        # clear gradients before backward pass    
        optimizer.zero_grad()
        
        #### forward pass/ solve for DA-market
        dcopf_output = DO_LRmodel.forward(train_inputs)
        # generator setpoints
        p_hat = dcopf_output[0].clamp(0)
        # predictions
        y_pred = DO_LRmodel.model[:-1].forward(train_inputs)
        w_error_hat = train_labels - y_pred
        # solve RT market
        rt_output = rt_layer(p_hat, w_error_hat, solver_args={'max_iters':10000})
        # cost of redispatching
        loss = rt_output[-1].sum()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        #break
    
    average_train_loss = running_loss / len(train_loader)
                
    DO_LRmodel.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            
            val_dcopf_outputs = DO_LRmodel.forward(val_inputs)        
            # predictions
            val_predictions = DO_LRmodel.model[:-1].forward(val_inputs)
            # forecast error
            val_w_error_hat = val_labels - val_predictions
            # solve RT market
            val_rt_output = rt_layer(val_dcopf_outputs[0], val_w_error_hat, solver_args={'max_iters':10000})
            # cost of redispatching
            total_val_loss += val_rt_output[-1].mean()       
            #break
    ave_val_rt_loss = total_val_loss/len(val_loader)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss / len(train_loader):.4f} - Val Loss: {ave_val_rt_loss.item():.4f}")

    # Implement early stopping
    if ave_val_rt_loss < best_val_loss:
        best_val_loss = ave_val_rt_loss
        best_model_weights = copy.deepcopy(DO_LRmodel.state_dict())
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break

DO_LRmodel.load_state_dict(best_model_weights)

# generate value-oriented predictions/ skip last layer of the model
DO_LR_pred = to_np(DO_LRmodel.model[:-1].forward(tensor_testX).clamp(0,1))
print(eval_point_pred(DO_LR_pred, testY))

#%%

models = ['LR_PO', 'NN_PO', 'LR_DO', 'NN_DO']
Predictions_df = pd.DataFrame(data = [], columns = ['Target'] + models)

Predictions_df['Target'] = testY.values.reshape(-1)
Predictions_df['LR_PO'] = PO_LR_pred
Predictions_df['LR_DO'] = DO_LR_pred

Predictions_df['NN_PO'] = PO_NN_pred
Predictions_df['NN_DO'] = DO_NN_pred

DA_cost = Predictions_df[models].copy()
RT_cost = Predictions_df[models].copy()

#%%
plt.plot(testY.values[:200], label = 'Actual')
plt.plot(PO_NN_pred[:200], label = 'Accuracy-oriented')
plt.plot(DO_NN_pred[:200], label = 'Value-oriented')
plt.legend()
plt.show()

for m in models:
    
    da_cost, rt_cost, da_sol, rt_sol = DA_RT_maker(grid, grid['w_capacity']*Predictions_df[m].values, 
                                                   grid['w_capacity']*Predictions_df['Target'].values, grid['w_bus'],
                                                   network = True, plot = False, verbose = 0, return_ave_cpu = False)

    DA_cost[m] = da_cost
    RT_cost[m] = rt_cost

Predictions_df.to_csv(f'{cd}\\results\\{case_name}_predictions.csv')
DA_cost.to_csv(f'{cd}\\results\\{case_name}_da_cost.csv')
RT_cost.to_csv(f'{cd}\\results\\{case_name}_rt_cost.csv')

Total_cost = DA_cost + RT_cost

print('Mean DA cost')
print(DA_cost.mean())

print('Mean RT cost')
print(RT_cost.mean())

print('Mean Total cost')
print(Total_cost.mean())
#%%
Total_cost.mean().plot(kind='bar', ylim = (29750, 31e3))
plt.show()
#%%
# sanity check
t = 72
plt.plot(grid['Pd'].sum() - grid['w_capacity']*testY.values[:72], label = 'Actual')
plt.plot(da_sol['p'].sum(1)[:72], label = 'DA')
plt.plot(da_sol['p'].sum(1)[:72] + rt_sol['r_up'].sum(1)[:72] - rt_sol['r_down'].sum(1)[:72], '--', color = 'black', label = 'DA+r_up')
plt.legend()
plt.show()

stop_here
#%%
Predictions_df = pd.read_csv(f'{cd}\\results\\pglib_opf_case14_ieee_predictions.csv', index_col = 0)

DA_cost = pd.read_csv(f'{cd}\\results\\pglib_opf_case14_ieee_da_cost.csv', index_col = 0)
RT_cost = pd.read_csv(f'{cd}\\results\\pglib_opf_case14_ieee_rt_cost.csv', index_col = 0)
Total_cost = DA_cost+RT_cost
#%%
rmse = np.sqrt(np.square(Predictions_df[models] - Predictions_df['Target'].values.reshape(-1,1)).mean())

fig, ax  = plt.subplots()
Total_cost.mean().plot(kind='bar', ylim = (1830, 1870), ax = ax, width = 0.5)
plt.xticks(np.arange(4), ['LinReg', 'NN', 'LinReg-DF', 'NN-DF'], rotation = 0)
plt.show()

plt.bar(np.arange(4), 100*rmse, width = 0.5)
plt.xticks(np.arange(4), ['LinReg', 'NN', 'LinReg-DF', 'NN-DF'], rotation = 0)
plt.ylim((10, 30))
plt.show()
#%% NN/ Decision-focused learning/ closed-form
patience = 5
batch_size = 50
num_epochs = 1000
check_validation = False

DO_NNmodel = e2e_MLP(input_size = input_size, hidden_sizes = n_hidden_layers*[n_nodes], output_size = output_size)
# initialize pre-trained weights

pretrained_state_dict = PO_NNmodel.state_dict()
DO_NNmodel.load_state_dict(pretrained_state_dict)

#DO_NNmodel.model.add_module(f'dcopf_layer', tensor_DA_layer)
#DO_NNmodel.model.add_module(f'dcopf_layer', tensor_RT_layer)
torch.autograd.set_detect_anomaly(True)

Losses = []
optimizer = torch.optim.Adam(DO_NNmodel.parameters(), lr=1e-3)
best_val_loss = float('inf')

c_up_tensor = torch.FloatTensor((grid['C_up']-grid['Cost'])) 
c_down_tensor = torch.FloatTensor((grid['Cost'] - grid['C_down'])) 

for epoch in range(num_epochs):
    
    # enable train functionality
    DO_NNmodel.train()
    running_loss = 0.0

    #train_inputs, train_labels = next(iter(train_loader))
    for train_inputs, train_labels in train_loader:
        
        # clear gradients before backward pass    
        optimizer.zero_grad()
        
        #### forward pass/ solve for DA-market
        p_gen_hat = DO_NNmodel.forward(train_inputs, grid)
        y_pred = DO_NNmodel.predict(train_inputs)
        
        w_error_hat = grid['w_capacity']*(train_labels - y_pred)
        
        # Pass through custom layers        
        #!!!! Do not forget the capacity
        
        rt_output = rt_layer(p_gen_hat, w_error_hat)
        loss = rt_output[-1].mean()
        
        #r_up_hat, r_down_hat = tensor_RT_layer().forward(w_error_hat, p_gen_hat, grid)
        #loss = (r_up_hat@c_up_tensor + r_down_hat@c_down_tensor).sum()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
            
    average_train_loss = running_loss / len(train_loader)
                
    DO_NNmodel.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            
            # predictions
            val_p_gen_hat = DO_NNmodel.forward(val_inputs, grid)
            val_predictions = DO_NNmodel.predict(val_inputs)
            val_w_error_hat = val_labels - val_predictions
            
            # Closed-form solution          
            #val_r_up_hat, val_r_down_hat = tensor_RT_layer().forward(val_w_error_hat, val_p_gen_hat, grid)
            #total_val_loss += (val_r_up_hat@c_up_tensor + val_r_down_hat@c_down_tensor).mean()
            
            # Solve problem
            val_rt_output = rt_layer(val_p_gen_hat, val_w_error_hat)
            # cost of redispatching
            total_val_loss += val_rt_output[-1].mean()       

    ave_val_rt_loss = total_val_loss/len(val_loader)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss / len(train_loader):.4f} - Val Loss: {ave_val_rt_loss.item():.4f}")

    # Implement early stopping
    if ave_val_rt_loss < best_val_loss:
        best_val_loss = ave_val_rt_loss
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
