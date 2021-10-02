# Prescriptive Trees for Value-oriented Forecasting and Optimization: Applications on Storage Scheduling and Market Clearing

This repository contains the code to reproduce the experiments of the title paper. Each application is contained in a separate folder. The key parts are:
- `*_main.py`: Run the experiments.
- `_opt_problem.py`: Function that contains the optimization problem, determines the sample average approximation.
- `GreedyPrescriptiveTree.py`: Train a prescriptive with greedy node splits.
-  `EnsemblePrescriptiveTree.py`: Train an ensemble of prescriptive trees, following either the Random Forest or the ExtraTrees algorithm.
