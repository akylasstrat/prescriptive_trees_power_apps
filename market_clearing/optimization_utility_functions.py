# -*- coding: utf-8 -*-
"""
Functions for the optimization problem

@author: akylas.stratigakos@mines-paristech.fr
"""
import pandas as pd

def load_ieee24(ieee_24_path):
    'Loads IEEE24 data, returns a dictionary'
    path = ieee_24_path
    
    # Read separate dataframes with IEEE data
    Unit_df = pd.read_excel(path, 'Units')
    #Wind_nodes = pd.read_excel(cd+'\\IEEE24Wind_Data.xlsx', 'Wind')
    #System_Demand = pd.read_excel(cd+'\\IEEE24Wind_Data.xlsx', 'Load')['System Demand'].values
    Branch_df = pd.read_excel(ieee_24_path, 'Branch')
    
    Node_demand_df = pd.read_excel(path, 'NodeDemand')
    Node_demand_df['Percentage'] = Node_demand_df['%of system Load ']/100
    
    
    #class NewClass(object): pass
    grid = {}
    
    grid['node_demand_percentage'] = Node_demand_df['Percentage'].values
    
    # Line capacity
    Line_Capacity = Branch_df['Capacity'].values.reshape(-1,1)
    #!!!!!! Create congestion (see paper)
    Line_Capacity[(Branch_df['FROM']==15)*(Branch_df['TO']==21)] = 400
    Line_Capacity[(Branch_df['FROM']==14 )*(Branch_df['TO']==16)] = 250
    Line_Capacity[(Branch_df['FROM']==13)*(Branch_df['TO']==23)] = 250

    grid['Line_Capacity'] = Line_Capacity
    #Matrices that map Generators, Loads and Wind Plants to nodes
    grid['node_G'] = pd.read_excel(ieee_24_path, 'AG', header = None).values
    grid['node_L'] = pd.read_excel(ieee_24_path, 'AL', header = None).values
    grid['node_W'] = pd.read_excel(ieee_24_path, 'AW', header = None).values
    
    # Susceptance matrix, incidence graph matrix, and diagonal matrix with susceptances
    grid['B'] = pd.read_excel(ieee_24_path, 'B', index_col = 0).values #Susceptance Matrix
    grid['A'] = pd.read_excel(ieee_24_path, 'A', index_col = 0).values
    grid['b_diag'] = pd.read_excel(ieee_24_path, 'b_diag', header = None).values
    
    #Generators operational limits
    grid['Pmax'] = Unit_df['Pimax'].values.reshape(-1,1)
    grid['Pmin'] = Unit_df['Pimin'].values.reshape(-1,1)
    grid['R_up_max'] = Unit_df['Ri+'].values.reshape(-1,1)
    grid['R_down_max'] = Unit_df['Ri-'].values.reshape(-1,1)
    grid['Ramp_up_rate'] = Unit_df['RiU'].values.reshape(-1,1)
    grid['Ramp_down_rate'] = Unit_df['RiD'].values.reshape(-1,1)
    #Cost vectors
    grid['Cost'] = Unit_df['Ci'].values
    #Cost_up = Unit_df['Ciu'].values #Reserve up cost
    #Cost_down = Unit_df['Cid'].values #Reserve down cost
    grid['Cost_reg_up'] = Unit_df['Ci+'].values   #Upward regulation cost
    grid['Cost_reg_down'] = Unit_df['Ci-'].values #Downward regulation cost
    
    # Cardinality of sets
    grid['n_nodes'] = len(grid['B'])
    grid['n_lines'] = len(grid['A'])
    grid['n_unit'] = grid['node_G'].shape[1]
    grid['n_wind'] = grid['node_W'].shape[1]
    grid['n_loads'] = grid['node_L'].shape[1]
    
    #Other parameters set by user
    grid['Wind_capacity'] = 200 #MW
    grid['VOLL'] = 500   #Value of Lost Load
    grid['VOWS'] = 35   #Value of wind spillage
    grid['gshed'] = 200   #Value of wind spillage
    
    return grid
