

import numpy as np
import pandas as pd
from params import Params
params = Params()

def deal_results(system_parameters_optimal, xt_axis, x_optimal, number_points_interval,
                flag_minmax, flag_time_scaling, flag_terminaltime_fixed,
                number_control_variables, number_intervals,
                number_control_parameters, number_interval_parameters,
                t_initial, t_terminal, t_pre,
                t_initial_original1=None, t_initial_original2=None, t_terminal_original2=None):
    """
    Process optimal control results and prepare arrays for plotting and analysis.
    """
    # Placeholder: J_optimal handling (not implemented, see instructions)
    # if not flag_minmax:
    #     J_optimal = -J_optimal

    # 2. Extract control_parameters
    control_parameters = system_parameters_optimal[: number_control_variables * number_intervals] \
                        .reshape((number_control_variables, number_intervals), order='F')

    # 3. Update t_initial and t_terminal according to flags
    if flag_time_scaling:
        # If time scaling is used, t_initial and t_terminal may be from original problem
        if t_initial_original1 is not None:
            t_initial = t_initial_original1
        if flag_terminaltime_fixed:
            if t_terminal_original2 is not None:
                t_terminal = t_terminal_original2
    # else: t_initial and t_terminal remain as given

    # 4. If flag_time_scaling, compute theta
    if flag_time_scaling:
        num_u = number_control_variables * number_intervals
        theta = system_parameters_optimal[num_u:num_u + number_intervals]
        if not flag_terminaltime_fixed:
            # Rescale theta so sum(theta) = t_terminal - t_initial
            theta = (t_terminal - t_initial) * theta / np.sum(theta)
    else:
        theta = None

    # 5. Compute t_interval
    t_interval = (t_terminal - t_initial) / number_intervals

    # 6. Build control_parameters_stairs: shape (number_intervals+1, number_control_variables)
    control_parameters_stairs = np.concatenate(
        [control_parameters, control_parameters[:, -1][:, None]], axis=1
    ).T  # shape (number_intervals+1, number_control_variables)

    # 7. Build control_parameters_plot: shape (number_intervals, number_control_variables)
    control_parameters_plot = control_parameters.T

    # 8. Construct ut_axis_stairs: time grid for stairs plot (length number_intervals+1)
    ut_axis_stairs = np.zeros(number_intervals + 1)
    ut_axis_stairs[0] = t_initial
    if flag_time_scaling:
        # Cumulative sum of theta
        ut_axis_stairs[1:] = t_initial + np.cumsum(theta)
        # Ensure last value is exactly t_terminal
        ut_axis_stairs[-1] = t_terminal
    else:
        ut_axis_stairs = t_initial + np.arange(number_intervals + 1) * t_interval
    # --- shift control time axis by t_pre so first control step starts at t_pre
    ut_axis_stairs = ut_axis_stairs + t_pre

    # 9. Construct ut_axis_plot: midpoints of ut_axis_stairs
    ut_axis_plot = 0.5 * (ut_axis_stairs[:-1] + ut_axis_stairs[1:])

    # 10. Scale xt_axis if needed


    # 11. If flag_time_scaling, transform segments of xt_axis per theta and number_points_interval
    if flag_time_scaling:
        xt_axis_new = np.zeros_like(xt_axis)
        for i in range(number_intervals):
            i0 = int(number_points_interval[i, 0])
            i1 = int(number_points_interval[i, 1])
            segment = xt_axis[i0:i1 + 1]
            if segment.size > 1:
                segment_scaled = (
                        ut_axis_stairs[i]
                        + (ut_axis_stairs[i + 1] - ut_axis_stairs[i])
                        * (segment - segment[0]) / max(segment[-1] - segment[0], 1e-12)
                )
            else:
                segment_scaled = np.array([ut_axis_stairs[i]])
            xt_axis_new[i0:i1 + 1] = segment_scaled
        xt_axis = xt_axis_new

    # 12. Shift xt_axis by t_initial
    #xt_axis = xt_axis + t_initial

    # 13. If flag_time_scaling and number_interval_parameters > 0: delete intervals shorter than tolerance
    # Placeholder: not implemented, see instructions.
    # --- Package for exports (CSV/JSON) ---
    # a) time-series states in DAYS
    day = xt_axis * 7.0
    Xs = x_optimal[:, 0]
    Xr = x_optimal[:, 1]
    L  = x_optimal[:, 2]
    I  = x_optimal[:, 3]
    M  = x_optimal[:, 4]
    J  = x_optimal[:, 5] if x_optimal.shape[1] > 5 else np.zeros_like(day)
    df_states = pd.DataFrame({
        "day": day,
        "Xs": Xs, "Xr": Xr, "L": L, "I": I, "M": M,
        "X_total": Xs + Xr
    })

    # b) per-interval schedule (days)
    N = number_intervals
    start_weeks = ut_axis_stairs[:-1]
    end_weeks   = ut_axis_stairs[1:]
    mid_weeks   = 0.5 * (start_weeks + end_weeks)
    start_day = start_weeks * 7.0
    end_day   = end_weeks * 7.0
    mid_day   = mid_weeks * 7.0

    # control values per interval (shape: (2, N))
    vI = control_parameters[0, :]
    vM = control_parameters[1, :] if control_parameters.shape[0] > 1 else np.zeros(N)
    dt_weeks = (t_terminal - t_initial) / float(N)
    doseI_week = vI * dt_weeks
    doseM_week = vM * dt_weeks

    # chemo window flag (by index membership)
    try:
        from params import Params
        _p = Params()
        is_chemo = np.array([1 if (k in getattr(_p, "K_M_indices", [])) else 0 for k in range(N)], dtype=int)
    except Exception:
        is_chemo = np.zeros(N, dtype=int)

    df_sched = pd.DataFrame({
        "start_day": start_day,
        "end_day": end_day,
        "mid_day": mid_day,
        "vI": vI,
        "vM": vM,
        "doseI_week": doseI_week,
        "doseM_week": doseM_week,
        "is_chemo": is_chemo,
        "interval": np.arange(N, dtype=int)
    })

    # c) chemo markers (vertical lines) in days — at starts of chemo cycles
    try:
        chemo_markers_days = (ut_axis_stairs[getattr(_p, "K_M_indices", [])] * 7.0).tolist()
    except Exception:
        chemo_markers_days = (start_day[:1].tolist() if len(start_day) else [])

    # d) quick summary (optional)
    summary = {
        "N_intervals": int(N),
        "t_pre_days": float(t_pre * 7.0),
        "t_treatment_days": float((t_terminal - t_initial) * 7.0),
        "total_days": float((day[-1] - day[0]) if len(day) else 0.0),
        "final_X": float((Xs + Xr)[-1]) if len(day) else None,
        "final_L": float(L[-1]) if len(day) else None,
        "cum_dose_I_week": float(np.sum(doseI_week)),
        "cum_dose_M_week": float(np.sum(doseM_week)),
    }
    # Return all arrays as a dictionary
    return {
        "control_parameters_stairs": control_parameters_stairs,
        "control_parameters_plot": control_parameters_plot,
        "ut_axis_stairs": ut_axis_stairs,
        "ut_axis_plot": ut_axis_plot,
        "xt_axis": xt_axis,
        "x_optimal": x_optimal,
        "t_initial": t_initial,
        "t_terminal": t_terminal,
        "t_pre": t_pre,
        # --- added for export/plotting ---
        "timeseries_states": df_states,         # DataFrame: day, Xs, Xr, L, I, M, X_total
        "controls_schedule": df_sched,          # DataFrame: start_day,end_day,mid_day,vI,vM,doseI_week,doseM_week,is_chemo,interval
        "chemo_markers": chemo_markers_days,    # list of days
        "summary": summary
    }