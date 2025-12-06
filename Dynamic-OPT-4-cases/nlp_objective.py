import numpy as np
from scipy.integrate import solve_ivp
from ode_system import ODE_system
from params import Params
params = Params()
def NLP_objective(system_parameters, penalty_factor, smooth_factor,x_initial_vector, number_intervals, t_initial, t_final):
    """
    Compute the objective value and its gradient for an NLP problem using ODE integration.
    Args:
        system_parameters (np.ndarray): Optimization variables (system parameters).
        penalty_factor (float): Penalty factor for constraints.
        smooth_factor (float): Smoothing factor for the objective.
    Returns:
        tuple: (objective_value, objective_gradient)
    """
    # Numeric parameters (set as Python integers)
    number_state_variables = 6
    number_system_parameters = 2 * number_intervals


    # Initial condition: state + sensitivity (flattened)
    y0 = np.concatenate([
        x_initial_vector,
        np.zeros(number_state_variables * number_system_parameters, dtype=float)
    ])

    # Time interval for each step
    t_interval = (t_final - t_initial) / number_intervals
    tau = params.tau_weeks  # 脉冲窗口（周）

    # Loop over intervals and integrate
    for k in range(number_intervals):
        # 取该段控制（仍用“输注速率变量”）
        j1 = 2 * k
        j2 = j1 + 2
        vI_k, vM_k = system_parameters[j1:j2]

        # 化疗仅前四段允许：其余段强制为 0，并屏蔽灵敏度
        allow_M = (k in params.K_M_indices)
        allow_I = (k in params.K_I_indices)

        # 该段时间
        t0 = t_initial + k * t_interval
        tA = t0 + min(tau, 0.5 * t_interval)  # 防御：tau 不超过半段
        t1 = t_initial + (k + 1) * t_interval

        # —— 子段A：脉冲 —— #
        # 等剂量守恒： v_pulse * tau = v * (t_interval)
        scale = (t_interval / (tA - t0))
        uA = np.array([
            vI_k * scale if allow_I else 0.0,
            (vM_k * scale) if allow_M else 0.0
        ], dtype=float)

        deltaA = np.zeros(number_system_parameters, dtype=float)
        if allow_I:
            deltaA[k] = scale
        if allow_M:
            deltaA[k + number_intervals] = scale

        solA = solve_ivp(
            ODE_system, (t0, tA), y0,
            args=(uA, deltaA, params),
            method='LSODA', rtol=1e-6, atol=1e-8, max_step=(tA - t0)
        )

        if (not solA.success) or (not np.isfinite(solA.y).all()):
            return 1e9, np.zeros_like(system_parameters)

        # —— 子段B：清除（无输入） —— #
        yA = solA.y[:, -1]
        uB = np.zeros(2, dtype=float)
        deltaB = np.zeros(number_system_parameters, dtype=float)  # 子段B对 v_k 的敏感性为 0
        solB = solve_ivp(
            ODE_system, (tA, t1), yA,
            args=(uB, deltaB, params),
            method='LSODA', rtol=1e-6, atol=1e-8, max_step=(t1 - tA)
        )
        if (not solB.success) or (not np.isfinite(solB.y).all()):
            return 1e9, np.zeros_like(system_parameters, dtype=float)

        # 更新 y0
        y0 = solB.y[:, -1]

    # Extract state variables from the final state
    x0 = y0[:number_state_variables]

    # Compute the objective value
    objective_value = x0[5]

    # Always compute gradient
    s0 = y0[number_state_variables:]
    # Reshape to (number_state_variables, number_system_parameters) in column-major order
    s0_matrix = y0[number_state_variables:].reshape(
        (number_state_variables, number_system_parameters), order='F'
    )
    # dJ/dx is [0, 0, 1, 0, 1]
    dJdx = np.zeros(number_state_variables, dtype=float)
    dJdx[5] = 1.0
    # Chain rule: dJ/dp = dJ/dx * dx/dp
    # 原梯度顺序是参数顺序 [vI_0..vI_{N-1}, vM_0..vM_{N-1}]
    # 需要重排为优化变量的交错顺序 [vI_0, vM_0, vI_1, vM_1, ...]
    N = number_intervals
    g_p = (dJdx @ s0_matrix).reshape(-1)
    g_x = np.empty_like(g_p)
    g_x[0::2] = g_p[:N]
    g_x[1::2] = g_p[N:]
    return objective_value, g_x