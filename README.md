# Prescriptive Trees for Value-oriented Forecasting and Optimization

This repository contains the code to reproduce the experiments of the paper:

> Stratigakos, A., Camal, S., Michiorri, A., & Kariniotakis, G. (2021). [Prescriptive Trees for Integrated Forecasting and Optimization Applied in Trading of Renewable Energy](https://hal.archives-ouvertes.fr/hal-03363876v1), working paper or preprint.
}

 Each application is contained in a separate folder. The key parts are:

 - `GreedyPrescriptiveTree.py`: Train a prescriptive with greedy node splits (similar to CART).
 - `EnsemblePrescriptiveTree.py`: Train an ensemble of prescriptive trees (*prescriptive forest*). Different randomization algorithms included (Random Forest, ExtraTrees)
- `opt_problem.py`: Function that defines the specific optimization problem. During training, determines a Sample Average Approximation (SAA) of the data subset (determined by the tree). During prediction, determines a weighted SAA conditional on contextual information.
- `*_main.py`: Run the experiments.

## Intro

Decision-making in the presence of contextual information is a ubiquitous problem in modern power systems. The typical data-decisions pipeline comprises forecasting and optimization components deployed in sequence. However, the loss function employed during learning is only a proxy for task-specific costs (e.g. scheduling, trading). This work describes a data-driven alternative to improve prescriptive performance in conditional stochastic optimization problems based on nonparametric machine learning. Specifically, we describe prescriptive trees that minimize task-specific costs during learning, embedded with a scenario reduction procedure to reduce computations, and then derive a weighted Sample Average Approximation of the original problem. We present experimental results in two problems: storage scheduling for price arbitrage and stochastic market clearing with network constraints, respectively associated with electricity price and load forecasting.

## Storage Scheduling and Electricity Price Forecasting :battery:

Experiments on scheduling a generic storage device for price arbitrage.

```
storage
|
|--- data: contains input data
|--- figures: stores plots
|--- results: stores results
|--- storage_scheduling_main.py: run the experiments, generate plots
|--- Utility_functions.py: helper functions for data manipulation and probabilistic Forecasting
  ```

## Stochastic Market Clearing and Load Forecasting :electric_plug:

Stochastic market clearing with uncertain load and network constraints.

```
market_clearing
|
|--- data: contains input data
|--- figures: stores plots
|--- results
    |--- aggregated_results: results per sample size. To recreate plots, store results in the respective folder
    |--- CPU_results: results with scenario reduction
|--- market_clearing_main.py: run the main experiment
|--- cpu_time_test.py: run the scenario reduction experiment
|--- results_graphs.py: generate results tables and plots
|--- forecast_utility_functions.py: helper functions for data manipulation, probabilistic forecasting, and scenario generation
|--- optimization_utility_functions.py: load the network data
  ```
