# Prescriptive Trees for Value-oriented Forecasting and Optimization

This repository contains the code to reproduce the experiments of the following paper:

> @unpublished{stratigakos:hal-03363876,
  TITLE = {{Prescriptive Trees for Value-oriented Forecasting and Optimization: Applications on Storage Scheduling and Market Clearing}},
  AUTHOR = {Stratigakos, Akylas and Camal, Simon and Michiorri, Andrea and Kariniotakis, Georges},
  URL = {https://hal.archives-ouvertes.fr/hal-03363876},
  NOTE = {working paper or preprint},
  YEAR = {2021},
  MONTH = Oct,
  KEYWORDS = {Data-driven optimization ; decision trees ; electricity market ; prescriptive analytics ; value-oriented forecasting},
  PDF = {https://hal.archives-ouvertes.fr/hal-03363876/file/Preprint_PrescriptiveTrees_Submission.pdf},
  HAL_ID = {hal-03363876},
  HAL_VERSION = {v1},
}

Each application is contained in a separate folder. The key components are:


- `GreedyPrescriptiveTree.py`: Train a prescriptive with greedy node splits (similar to CART).
- `EnsemblePrescriptiveTree.py`: Train an ensemble of prescriptive trees (*prescriptive forest*). Different randomization algorithms are implemented (Random Forests, ExtraTrees).
- `opt_problem.py`: Function that defines the specific optimization problem. During learning, it returns a Sample Average Approximation (SAA) of the original problem, for the specific data subset. During prediction, determines a weighted SAA conditioned on features.
- `*_main.py`: Run the experiments.

## Intro

Decision-making in the presence of contextual information is a ubiquitous problem in modern power systems. The typical data-decisions pipeline comprises forecasting and optimization components deployed in sequence. However, the loss function employed during learning is only a proxy for task-specific costs (e.g. scheduling, trading). This work describes a data-driven alternative to improve prescriptive performance in conditional stochastic optimization problems based on nonparametric machine learning. Specifically, we describe prescriptive trees that minimize task-specific costs during learning, embedded with a scenario reduction procedure to reduce computations, and then derive a weighted Sample Average Approximation of the original problem. We present experimental results for two problems: storage scheduling for price arbitrage and stochastic market clearing with network constraints, respectively associated with electricity price and load forecasting.

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
