#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main driver for the improved SQP-based optimal control,
translated from the MATLAB script.

Author: [Your Name]
Dependencies: numpy
"""

import numpy as np
from nlp_solve import NLP_solve
from simulation_optimal import Simulation_optimal
from deal_results import deal_results
from export_results import export_results_package
from params import Params
params = Params()
# ----------------------------------------
# Section 1: Problem Basic Information
# ----------------------------------------
# User input part 1: Basic problem information
flag_minmax = True        # True indicates minimization problem, False indicates maximization
flag_terminaltime_fixed = True  # Whether the terminal time is fixed or free

number_state_variables       = 6    # Number of state variables in the system
number_control_variables     = 2    # Number of control inputs
number_design_parameters     = 0    # Number of design parameters
number_constraints_eq        = 0    # Number of equality constraints
number_constraints_ineq      = 0    # Number of inequality constraints
number_pathconstraints_eq    = 0    # Number of path equality constraints
number_pathconstraints_ineq  = 0    # Number of path inequality constraints

# ----------------------------------------
# Section 2: Solution Strategy Settings
# ----------------------------------------
# User input part 2: Solution strategy
number_intervals = 10
algorithm_ODE    = 0        # ODE solver choice: 0 -> ode45, 1 -> ode15s
algorithm_NLP    = 0        # NLP algorithm: 0 -> SQP, 1 -> Interior Point, 2 -> Trust-region
tolerance_ODE    = 1e-6
tolerance_NLP    = 1e-4
flag_time_scaling= False    # Whether to apply time scaling


# ----------------------------------------
# Section 3: Mathematical Model Definition
# ----------------------------------------
# User input part 3: Mathematical model
t_initial = 0.0             # Initial time
t_terminal = 30.0          # Terminal time
t_pre       = 57.0        # 约等于 400/7，让首个给药出现在 ~day 400
t_plot_end  = 100.0       # 100 周 ≈ 700 天

#   Xs(0)=1e6 (敏感肿瘤细胞)
#   Xr(0)=1e5 (耐药肿瘤细胞)
#   L(0)=1e4  (CD8+ T 细胞)
#   I(0)=0    (ICI 药物初始)
#   M(0)=0    (化疗药物初始)
x_initial_vector = np.array([1, 0.1, 0.01, 0.0, 0.0, 0.0])

# Control bounds: shape (number_control_variables, number_intervals)
# vI ∈ [0,1]（或 {0,1}），vM ∈ [0,5]
u_lower = np.vstack([  # shape (2, N)
    np.zeros(number_intervals),      # vI ≥ 0
    np.zeros(number_intervals)       # vM ≥ 0
])
u_upper = np.vstack([
    np.ones(number_intervals),       # vI ≤ 1
    5*np.ones(number_intervals)      # vM ≤ 5
])

# 仅允许前四段化疗：对其余段强制上界为 0
last_chemo = params.K_M_indices[-1] if len(params.K_M_indices) > 0 else -1
if number_intervals > last_chemo + 1:
    u_upper[1, last_chemo+1:] = 0.0

idx = np.arange(number_intervals)
last_ici = params.K_I_indices[-1] if len(params.K_I_indices) > 0 else -1
if number_intervals > last_ici + 1:
    u_upper[0, last_ici+1:] = 0.0

# Design parameter and terminal time bounds (empty here)
t_terminal_lower = None
t_terminal_upper = None
p_lower = None
p_upper = None

# ----------------------------------------
# Section 4: Initial Guess for Optimization Variables
# ----------------------------------------
# User input part 4: Initial guess
u_guess = np.vstack([
     0.6 * np.ones(number_intervals),   # vI 初始 0.6（不贴上界）
     2.5 * np.ones(number_intervals)    # vM 初始 2.5（不贴上界）
 ])
# 初值与允许窗口一致
u_guess[1, params.K_M_indices[-1]+1:] = 0.0   # chemo 仅前4段
u_guess[0, params.K_I_indices[-1]+1:] = 0.0   # ICI 仅前6段

t_terminal_guess = t_terminal
p_guess = np.empty((0,))

# Flatten the initial control guess into a 1D vector for the optimizer
system_parameters_init = u_guess.T.flatten()

# ----------------------------------------
# Main execution flow
# ----------------------------------------
# The following four functions must be implemented:
#   - NLP_solve(params_init, ... )
#   - Simulation_optimal(opt_params, penalty_factor, smooth_factor)
#   - Deal_results(...)
#   - Draw_results(...)

if __name__ == "__main__":
    # Call NLP solver
    # Expected to return: system_parameters_optimal, penalty_factor, smooth_factor
    system_parameters_optimal, penalty_factor, smooth_factor = \
        NLP_solve(system_parameters_init,
                  flag_minmax,
                  flag_terminaltime_fixed,
                  number_state_variables,
                  number_control_variables,
                  number_design_parameters,
                  number_constraints_eq,
                  number_constraints_ineq,
                  number_pathconstraints_eq,
                  number_pathconstraints_ineq,
                  number_intervals,
                  algorithm_ODE,
                  algorithm_NLP,
                  tolerance_ODE,
                  tolerance_NLP,
                  flag_time_scaling,
                  t_initial,
                  t_terminal,
                  x_initial_vector,
                  u_lower,
                  u_upper,
                  t_terminal_lower,
                  t_terminal_upper,
                  p_lower,
                  p_upper,
                  u_guess,
                  t_terminal_guess,
                  p_guess)
    print("🏁 NLP_solve finished. opt_params[0:2] =", system_parameters_optimal[:2])
    # Simulate optimal trajectory
    xt_axis, x_optimal, number_points_interval = Simulation_optimal(
        system_parameters_optimal,
        penalty_factor,
        smooth_factor,
        x_initial_vector,
        number_intervals,
        t_initial,
        t_terminal,
        t_pre,
        t_plot_end
    )

    # Process results
    results = deal_results(
        system_parameters_optimal,
        xt_axis,
        x_optimal,
        number_points_interval,
        flag_minmax,
        flag_time_scaling,
        flag_terminaltime_fixed,
        number_control_variables,
        number_intervals,
        number_control_variables,
        number_intervals,
        t_initial,
        t_terminal,
        t_pre
    )

    # Plot results
    paths = export_results_package(
        results,
        prefix="ours_run1",      # 你可改成病人ID或策略名
        outdir="exports"
    )
    print("Exported files:", paths)

# Implement the stub functions NLP_solve, Simulation_optimal, Deal_results, Draw_results below or in separate modules.
