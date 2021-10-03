# Prescriptive Trees for Value-oriented Forecasting and Optimization

This repository contains the code to reproduce the experiments of the paper "*Prescriptive Trees for Value-oriented Forecasting and Optimization: Applications on Storage Scheduling and Market Clearing*" (preprint available soon). Each application is contained in a separate folder. The key parts are:

- `*_main.py`: Run the experiments.
- `opt_problem.py`: Function that defines the specific optimization problem, and determines the sample average approximation.
- `GreedyPrescriptiveTree.py`: Train a prescriptive with greedy node splits (similar to CART).
- `EnsemblePrescriptiveTree.py`: Train an ensemble of prescriptive trees (*prescriptive forest*).

# Intro

Decision-making in the presence of contextual information is a ubiquitous problem in modern power systems. The typical data-decisions pipeline comprises forecasting and optimization components deployed in sequence. However, the loss function employed during learning is only a proxy for task-specific costs (e.g. scheduling, trading). This work describes a data-driven alternative to improve prescriptive performance in conditional stochastic optimization problems based on nonparametric machine learning. Specifically, we describe prescriptive trees that minimize task-specific costs during learning, embedded with a scenario reduction procedure to reduce computations, and then derive a weighted Sample Average Approximation of the original problem. 

# Storage Scheduling and Electricity Price Forecasting

Experiments on scheduling a generic storage device for price arbitrage.

# Stochastic Market Clearing and Load Forecasting

Stochastic market clearing with uncertain load and network constraints.
