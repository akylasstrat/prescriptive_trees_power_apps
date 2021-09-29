# -*- coding: utf-8 -*-
"""
Utility functions for forecasting. Includes:
    - Functions to create supervised learning set (for ID horizon forecasts)
    - Function to create historical lagged predictors based on Partial Autocorrelation Function
    - Evaluation functions for point and probabilistic predictions

@author: a.stratigakos
"""
import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import shift
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

############################## Preprocessing, data manipulation, preliminary analysis

def create_supervised(df, config):
    'Import data, create supervised learning set'
    #df = pd.read_excel('Market_Data_processed.xlsx', index_col = 0)
    #df.index = df.index.round('min')
    predictors = df.columns[df.columns !='DA Price']

    train_split = np.where(df.index == config['train_date'])[0][0]
    start_split = np.where(df.index == config['start_date'])[0][0]

    trainY = df['DA Price'][start_split:train_split].to_frame()
    testY = df['DA Price'][train_split:].to_frame()
    
    trainX = df[predictors][start_split:train_split]
    testX = df[predictors][train_split:]

    target =df['DA Price'][train_split:].to_frame()
    
    return trainY, testY, trainX, testX, target

def create_supervised_prescriptive(trainY, testY, trainX, testX, config):
    'Supervised learning set for prescriptive trees (wide format)'
    h = config['horizon']    
    
    new_trainY = trainY.values.reshape(-1,h)
    
    full_pred_list = trainX.columns.values
    var_list = [p for p in full_pred_list if p not in ['Day', 'Month']]
    
    new_trainX = trainX[var_list].values.reshape(-1,len(var_list)*h)
    new_testX = testX[var_list].values.reshape(-1,len(var_list)*h)
    
    new_trainX = np.column_stack((new_trainX, trainX[['Day', 'Month']].values[::h]))
    new_testX = np.column_stack((new_testX, testX[['Day', 'Month']].values[::h]))

    return new_trainY, new_trainX, new_testX


def clean_entsoe_data(df, freq, impute = False):
    ''' Data to fill missing values and remove duplicates for daylightsavings
    Not needed if everything is in UTC'''
    #freq = number of observations in an hour (1 for hourly, 4 for 15-min)
    types = (df.dtypes == 'float64').values + (df.dtypes == 'int64').values
    ind = np.where(types)[0]    #Quantitative indexes           
    if impute is False:
        #Simply replace (works for quant+categorical values)
        #Check for NaNs
        if any(df.iloc[:,ind[0]].isnull()):
            print('Index of NaNs \n', df.index[df.iloc[:,ind[0]].isnull()] )
            NaN_ind = np.where(df.iloc[:,ind[0]].isnull())[0]
    
            #Replace NaNs
            df.iloc[NaN_ind] = df.iloc[NaN_ind-freq].values

        #Check for duplicates and average them
        if any(df.index.duplicated()):
            print('Duplicated Indices \n', df.index[df.index.duplicated()])
            df  = df.loc[~df.index.duplicated(keep='first')]    
    else:
        #Impute missing values (only for quantitative)
        #Check for NaNs
        if any(df.iloc[:,ind[0]].isnull()):
            print('Index of NaNs \n', df.index[df.iloc[:,ind[0]].isnull()] )
            NaN_ind = np.where(df.iloc[:,ind[0]].isnull())[0]
            #Linear Interpolation (for quantitative values only)
            for i in ind:    
                y_0 = df.iloc[NaN_ind[0]-1, i]   #Start
                y_t = df.iloc[NaN_ind[-1]+1, i]  #Stop
                dx = freq + 1
                dy = (y_0-y_t)/dx
                df.iloc[NaN_ind,i] = np.arange(1,dx)*dy+y_0
        #Check for duplicates and average them
        if any(df.index.duplicated()):
            print('Duplicated Indices \n', df.index[df.index.duplicated()])
            dupl_ind = np.where(df.index.duplicated())[0]   #Index of duplicated values
            ave = (df.iloc[dupl_ind,types] + df.iloc[dupl_ind-freq,types])/2    #Average value
            df.iloc[dupl_ind] = ave
            df  = df.loc[~df .index.duplicated(keep='last')]
    return df

def feat_imp(model, testX, testY, number_feat = 10):
    ''' Estimates feature importance using the permutation importance, saves output, plots results'''
    
    result = permutation_importance(model, testX, testY, n_repeats=25, random_state=2, n_jobs=-1)
    
    forest_importances = pd.Series(result.importances_mean, index=testX.columns)
    n_feat = number_feat
    index = forest_importances.values.argsort()[::-1]
    
    fig, ax = plt.subplots()
    forest_importances[index[:n_feat]].plot.barh(xerr=result.importances_std[index[:n_feat]], ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean Accuracy Decrease")
    fig.tight_layout()
    plt.show()
    
def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation between x and y. 
    Inputs:
        -lag : int, default 0
        -datax, datay : pandas.Series objects of equal length
    Output: crosscorr : float
    """
    return [datax.corr(datay.shift(l)) for l in range(lag)]


######################### Forecast evaluation
#!!!!!!!
'''2-D arrays used throughout'''

def eval_point_pred(predictions, actual, digits = None):
    ''' Returns point forecast metrics: MAPE, RMSE, MAE '''
    assert predictions.shape == actual.shape, "Shape missmatch"
    
    mape = np.mean(abs( (predictions-actual)/actual) )
    rmse = np.sqrt( np.mean(np.square( predictions-actual) ) )
    mae = np.mean(abs(predictions-actual))
    if digits is None:
        return mape,rmse, mae
    else: 
        return round(mape, digits), round(rmse, digits), round(mae, digits)
    
def pinball(prediction, target, quantiles):
    ''' Estimates Pinball loss '''
    num_quant = len(quantiles)
    pinball_loss = np.maximum( (np.tile(target, (1,num_quant)) - prediction)*quantiles,(prediction - np.tile(target , (1,num_quant) ))*(1-quantiles))
    return pinball_loss  

def CRPS(target, quant_pred, quantiles, digits = None):
    ''' Estimates CRPS (Needs check) '''
    n = len(quantiles)
    #Conditional prob
    p = 1. * np.arange(n) / (n - 1)
    #Heaviside function
    H = quant_pred > target 
    if digits == None:
        return np.trapz( (H-p)**2, quant_pred).mean()
    else:
        return round(np.trapz( (H-p)**2, quant_pred).mean(), digits)

def pit_eval(target, quant_pred, quantiles, plot = False, nbins = 20):
    '''Evaluates Probability Integral Transformation
        returns np.array and plots histogram'''
    #n = len(target)
    #y = np.arange(1, n+1) / n
    y = quantiles
    PIT = [ y[np.where(quant_pred[i,:] >= target[i])[0][0]] if any(quant_pred[i,:] >= target[i]) else 1 for i in range(len(target))]
    PIT = np.asarray(PIT).reshape(len(PIT))
    if plot:
        plt.hist(PIT, bins = nbins)
        plt.show()
    return PIT

def reliability_plot(target, pred, quantiles, boot = 100, label = None, plot = True):
    ''' Reliability plot with error consistency bars '''
    cbands = []
    for j in range(boot):
        #Surgate Observations
        Z = np.random.uniform(0,1,len(pred))
        
        Ind = 1* (Z.reshape(-1,1) < np.tile(quantiles,(len(pred),1)))
        cbands.append(np.mean(Ind, axis = 0))
    
    ave_proportion = np.mean(1*(pred>target), axis = 0)
    cbands = 100*np.sort( np.array(cbands), axis = 0)
    lower = int( .05*boot)
    upper = int( .95*boot)
 
    ave_proportion = np.mean(1*(pred>target), axis = 0)
    if plot:
        plt.vlines(100*quantiles, cbands[lower,:], cbands[upper,:])
        plt.plot(100*quantiles,100*ave_proportion, '-*')
        plt.plot(100*quantiles,100*quantiles, '--')
        plt.legend(['Observed', 'Target'])
        plt.show()
    return

def brier_score(predictions, actual, digits = None):
    ''' Evaluates Brier Score '''
    if digits == None:
        return np.mean(np.square(predictions-actual))
    else:
        return round(np.mean(np.square(predictions-actual)), digits)
    
def PINAW_plot(target, pred, quantiles, plot = True):
    '''Plots Prediction Interval Normalized Average Width (PINAW)/ sharpness evaluation
    Makes sense only if there are several models to compare '''
    
    PINAW = np.zeros(( int(len(quantiles)/2)))
    PIs = []
    for j in range(int(len(quantiles)/2)):
        lower = pred[:,j]
        upper = pred[:,-1-j]
        PINAW[j] = 100*np.mean(upper - lower)/(target.max()-target.min())
        PIs.append(quantiles[-1-j] - quantiles[j])
    plt.figure(dpi = 600, facecolor='w', edgecolor='k')
    plt.plot(PIs[::-1], PINAW.T[::-1])
    plt.ylabel('[%]')
    plt.xlabel('Prediction Intervals [%]')
    plt.title('Prediction Interval Normalized Average Width')
    plt.show()

def energy_score(target, pred_scenarios, step):
    '''Estimates Energy Score of trajectories
        step: length of trajectory vector/ forecast horizon'''
    ES = []
    n_scen = pred_scenarios.shape[1]

    for i in range(step, len(target), step):
        dist = np.linalg.norm(pred_scenarios[i-step:i] - target[i-step:i], axis = 0).mean()
        t = 0
        for j in range(n_scen):
            t = t + np.linalg.norm(pred_scenarios[i-step:i, j:j+1] - pred_scenarios[i-step:i],  axis = 0).sum()
        t = t/(2*n_scen**2)
        ES.append(dist - t)
    return(np.array(ES).mean())

################################### Other functions
def find_freq(freq_str):
    if (freq_str == '1h') or (freq_str == 'H') :
        return 1
    elif freq_str == '15min':
        return 4
    elif freq_str == '4h':
        return 1/4
    else:
        print('Check time series frequency')
        return

def VaR(data, quant = .05, digits = 3):
    ''' Estimates Value at Risk at quant level'''
    if digits is None:
        return np.quantile(data, q = quant)
    else:
        return round(np.quantile(data, q = quant), digits)

def CVaR(data, quant = .05, digits = 3):
    ''' Estimates Conditional Value at Risk at quant level'''
    VaR = np.quantile(data, q = quant)
    if digits is None:
        return data[data<=VaR].mean()
    else:
        return round(data[data<=VaR].mean(), digits)
    