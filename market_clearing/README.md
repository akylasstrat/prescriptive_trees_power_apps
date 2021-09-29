Economic dispatch with **two-stage** stochastic optimization.

Uncertainty:
- Load (single series, disseminated to all nodes)
- **wind is not considered here**

Files:
- `demand_economic_dispatch_main.py`: main script to run the optimization
- `opt_problem.py`: describes and solves the optimization problem, used internally by prescriptive trees. **Note**: `economic_dispatch_main.py` and `opt_problem.py` should be solving the same problem.
- `forecasting_module.py`: generates point, probabilistic and scenario forecasts for wind and load, saves output.
-`Utility_functions.py`: auxiliary functions
