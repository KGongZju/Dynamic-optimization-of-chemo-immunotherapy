import numpy as np
from scipy.integrate import solve_ivp
from params import Params
from ode_system import ODE_system
params = Params()
def Simulation_optimal(system_parameters, penalty_factor, smooth_factor,
                       x_initial_vector, number_intervals,
                       t_initial, t_terminal, t_pre, t_end):
    """
    Simulate the optimal trajectory with three concatenated segments:
    (1) pre-treatment [t_initial, t_initial+t_pre] with vI=vM=0,
    (2) treatment [t_initial+t_pre, t_initial+t_pre+t_terminal-t_initial] with CVP controls,
    (3) post-treatment [.., t_end] with vI=vM=0.
    Robust for any number_intervals.
    """
    # dimensions
    n = 6
    m = 2
    N = number_intervals

    # initial condition: state + sensitivities
    y0 = np.concatenate([x_initial_vector, np.zeros(n * 2 * N)])

    # helper: integrate one segment with fixed controls and delta vector
    def integrate_segment(y0, t0, t1, u_k, delta):
        t_eval = np.linspace(t0, t1, 50)
        sol = solve_ivp(
            ODE_system,
            (t0, t1),
            y0,
            args=(u_k, delta, params),
            method='LSODA',
            rtol=1e-6, atol=1e-8,
            max_step=(t1 - t0) / 20,
            t_eval=t_eval
        )
        return sol

    # containers for concatenated time/state; store only the 6 states for plotting
    T_all = None
    X_all = None

    # utility to append while avoiding duplicated knot point
    def append_solution(sol, keep_first):
        nonlocal T_all, X_all
        t_seg = sol.t
        x_seg = sol.y[:n, :].T
        if keep_first:
            t_use = t_seg
            x_use = x_seg
        else:
            t_use = t_seg[1:]
            x_use = x_seg[1:, :]
        if T_all is None:
            T_all = t_use.copy()
            X_all = x_use.copy()
        else:
            T_all = np.concatenate([T_all, t_use])
            X_all = np.vstack([X_all, x_use])

    # 1) pre-treatment segment (no drug)
    if t_pre > 0:
        sol_pre = integrate_segment(y0, t_initial, t_initial + t_pre, u_k=np.zeros(m), delta=np.zeros(2 * N))
        append_solution(sol_pre, keep_first=True)
        y0 = sol_pre.y[:, -1]

    # 2) treatment segment (CVP, N intervals)
    number_points_interval = np.zeros((N, 2), dtype=int)
    dt = (t_terminal - t_initial) / N
    t_start = t_initial + t_pre
    for k in range(N):
        j1 = 2 * k
        j2 = j1 + 2
        vI_k, vM_k = system_parameters[j1:j2]
        allow_M = (k in params.K_M_indices)
        allow_I = (k in params.K_I_indices)

        t0 = t_start + k * dt
        tA = t0 + min(params.tau_weeks, 0.5 * dt)
        t1 = t0 + dt


        scale = dt / (tA - t0)
        uA = np.array([
            vI_k * scale if allow_I else 0.0,
            (vM_k * scale) if allow_M else 0.0
        ], dtype=float)
        deltaA = np.zeros(2 * N, dtype=float)
        if allow_I: deltaA[k] = scale
        if allow_M: deltaA[k + N] = scale

        t_evalA = np.linspace(t0, tA, 60)
        solA = solve_ivp(ODE_system, (t0, tA), y0, args=(uA, deltaA, params),
                         method='LSODA', rtol=1e-6, atol=1e-8, max_step=(tA - t0) / 20, t_eval=t_evalA)


        yA = solA.y[:, -1]
        uB = np.zeros(2, dtype=float)
        deltaB = np.zeros(2 * N, dtype=float)
        t_evalB = np.linspace(tA, t1, 40)
        solB = solve_ivp(ODE_system, (tA, t1), yA, args=(uB, deltaB, params),
                         method='LSODA', rtol=1e-6, atol=1e-8, max_step=(t1 - tA) / 20, t_eval=t_evalB)


        t_seg = np.concatenate([solA.t, solB.t[1:]])
        x_seg = np.concatenate([solA.y[:n, :], solB.y[:n, 1:]], axis=1).T
        if T_all is None:
            T_all = t_seg.copy();
            X_all = x_seg.copy()
        else:
            T_all = np.concatenate([T_all, t_seg[1:]])
            X_all = np.vstack([X_all, x_seg[1:, :]])


        y0 = solB.y[:, -1]
        before = number_points_interval[k - 1, 1] + 1 if k > 0 else 0
        after = T_all.size
        number_points_interval[k, 0] = before
        number_points_interval[k, 1] = after - 1

    # 3) post-treatment segment (no drug), extend to t_end if needed
    t_after = t_start + (t_terminal - t_initial)
    if t_end > t_after:
        sol_post = integrate_segment(y0, t_after, t_end, u_k=np.zeros(m), delta=np.zeros(2 * N))
        append_solution(sol_post, keep_first=False)
        y0 = sol_post.y[:, -1]

    # masks for safety
    mask_time = np.isfinite(T_all)
    mask_state = np.isfinite(X_all).all(axis=1)
    mask = mask_time & mask_state
    xt_axis = T_all[mask]
    x_optimal = X_all[mask, :]

    return xt_axis, x_optimal, number_points_interval
