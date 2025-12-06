import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, Bounds
from scipy.sparse import csr_matrix
import time
from nlp_objective import NLP_objective
from params import Params
params = Params()
def NLP_solve(system_parameters_init,
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
              p_guess):
    """
    Solve NLP with smooth penalty via interior-point (like fmincon).
    Returns (system_parameters_optimal, penalty_factor, smooth_factor).
    """
    # Flatten bounds
    lb = u_lower.T.flatten()
    ub = u_upper.T.flatten()

    # initial guess

    x0 = system_parameters_init.copy()

    # --- 让初值满足斜率约束并落在边界内（避免一上来就 infeasible）---
    def _ramp_project(x):
        xv = x.copy()
        # 先按上下界裁剪一次
        xv = np.minimum(np.maximum(xv, lb), ub)
        vI = xv[0::2].copy(); vM = xv[1::2].copy()
        rhoI, rhoM = 0.6, 1.5  # 与 make_constraints 里一致
        for k in range(1, number_intervals):
            d = vI[k] - vI[k-1]
            if d >  rhoI: vI[k] = vI[k-1] + rhoI
            if d < -rhoI: vI[k] = vI[k-1] - rhoI
            d = vM[k] - vM[k-1]
            if d >  rhoM: vM[k] = vM[k-1] + rhoM
            if d < -rhoM: vM[k] = vM[k-1] - rhoM
        xv[0::2] = vI; xv[1::2] = vM
        # 再次确保边界
        xv = np.minimum(np.maximum(xv, lb), ub)
        return xv

    x0 = _ramp_project(x0)

    # 主力求解器：trust-constr（对非线性约束 + ODE 平滑目标更稳）
    method_primary = 'trust-constr'
    solver_options_tc = {
        'gtol': max(tolerance_NLP, 1e-8),
        'xtol': max(tolerance_NLP, 1e-8),
        'maxiter': 1000,
        'initial_tr_radius': 1.0,
        'verbose': 2,
        'finite_diff_rel_step': 1e-6,
    }
    # 备用：SLSQP（只在主力未收敛/不可行时尝试）
    solver_options_slsqp = {'ftol': tolerance_NLP, 'maxiter': 300, 'disp': True, 'eps': 1e-8}

    # penalty initialization
    penalty_times = 1
    penalty_factor = 1.0
    smooth_factor = 0.0

    # define objective wrapper
    def obj_wrap(x, penalty_factor, smooth_factor):
        J, grad = NLP_objective(x, penalty_factor, smooth_factor,
                                x_initial_vector, number_intervals,
                                t_initial, t_terminal)
        grad = np.asarray(grad, dtype=float).reshape(-1)
        if not flag_minmax:   # maximize → minimize by flipping sign
            return -float(J), -grad
        return float(J), grad

    # fun 仅返回标量；jac 单独返回 1D 梯度。这可避免 code 8 的接口歧义。
    def fun_only(x):
        J, _ = NLP_objective(x, penalty_factor, smooth_factor,
                             x_initial_vector, number_intervals,
                             t_initial, t_terminal)
        return float(J)

    def jac_only(x):
        _, g = NLP_objective(x, penalty_factor, smooth_factor,
                             x_initial_vector, number_intervals,
                             t_initial, t_terminal)
        return np.asarray(g, dtype=float).reshape(-1)

    # 零 Hessian（稀疏），trust-constr 会用其内部 BFGS 构造近似
    def hess_zero(_x):
        n = _x.size
        return csr_matrix((n, n))

    # nlp_solve.py 中，minimize 调用前加：
    def make_constraints():
        from scipy.integrate import solve_ivp
        from ode_system import ODE_system
        from params import Params
        p = Params()
        # === 共享缓存：避免同一 x 在目标/约束里重复积分 ===
        _cache = {"key": None, "states": None, "sens": None}

        def _key(x):
            return x.tobytes()

        def _permute_p2x_cols(A):
            """把列从参数顺序 [I0..I{N-1}, M0..M{N-1}] 转为变量交错顺序 [I0,M0,I1,M1,...]."""
            Nloc = number_intervals
            idx = np.empty(2 * Nloc, dtype=int)
            idx[0::2] = np.arange(Nloc)
            idx[1::2] = Nloc + np.arange(Nloc)
            return A[:, idx]

        def _integrate_all(x):
            """
            跨 N 个区间只积分一次，返回：
              states_list: 每个区间末的 y[:6]（长度 N）
              sens_list:   每个区间末的敏感度矩阵 S_end，形状 (6 x 2N)，列为参数顺序 [I0.., M0..]
            """
            kx = _key(x)
            if _cache["key"] == kx:
                return _cache["states"], _cache["sens"]

            dt = (t_terminal - t_initial) / number_intervals
            # y_vec 拼上敏感度展开（你现有的状态布局保持不变：前6是状态，后面是敏感度列展开，Fortran顺序）
            y_vec = np.concatenate([x_initial_vector, np.zeros(6 * 2 * number_intervals)])
            states_list, sens_list = [], []

            t0 = t_initial
            for k in range(number_intervals):
                vI, vM = x[2 * k], x[2 * k + 1]
                allow_I = (k in params.K_I_indices)
                allow_M = (k in params.K_M_indices)

                tA = t0 + min(params.tau_weeks, 0.5 * dt)
                t1 = t0 + dt
                sc = dt / max(tA - t0, 1e-12)

                uA = np.array([vI * sc if allow_I else 0.0,
                               vM * sc if allow_M else 0.0], dtype=float)

                dA = np.zeros(2 * number_intervals)
                if allow_I: dA[k] = sc
                if allow_M: dA[k + number_intervals] = sc

                solA = solve_ivp(ODE_system, (t0, tA), y_vec,
                                 args=(uA, dA, params),
                                 method='LSODA', rtol=1e-6, atol=1e-8, max_step=(tA - t0))
                yA = solA.y[:, -1]

                solB = solve_ivp(ODE_system, (tA, t1), yA,
                                 args=(np.zeros(2), np.zeros_like(dA), params),
                                 method='LSODA', rtol=1e-6, atol=1e-8, max_step=(t1 - tA))
                y_vec = solB.y[:, -1]

                states_list.append(y_vec[:6].copy())
                S_end = y_vec[6:].reshape((6, 2 * number_intervals), order='F')
                sens_list.append(S_end.copy())
                t0 = t1

            _cache["key"] = kx
            _cache["states"] = states_list
            _cache["sens"] = sens_list
            return states_list, sens_list

        # --- 路径约束参数 ---
        tau_max = 0.30     # 当期免疫毒性：K_L*(1 - e^{-M_k}) ≤ tau_max
        L_min   = 0.005    # 每段末 L ≥ L_min
        I_max   = 3.0      # I ≤ I_max
        M_max   = 3.0      # M ≤ M_max

        # --- 轻约束（化疗模式）：仅限制 I、M 上界，去掉 L_min 与当期毒性 ---
        def g_path(x):
            states_list, _ = _integrate_all(x)
            G = []
            I_max = 10.0   # 放宽 I 上限（单位与状态一致）
            M_max = 5.0    # 放宽 M 上限
            for y_end in states_list:
                I, M = y_end[3], y_end[4]
                # 约束顺序： I<=I_max, M<=M_max
                G += [I_max - I, M_max - M]
            return np.array(G, dtype=float)

        def jac_path(x):
            # 用每段末的敏感度矩阵构造解析雅可比（仅对 I、M）
            _, sens_list = _integrate_all(x)
            Nloc = number_intervals
            rows = 2 * Nloc
            J = np.zeros((rows, 2 * Nloc), dtype=float)  # 先按参数顺序

            for k, S_end in enumerate(sens_list):
                r0 = 2 * k
                # d(I_max - I)/dp = - dI/dp
                J[r0, :] = - S_end[3, :]
                # d(M_max - M)/dp = - dM/dp
                J[r0 + 1, :] = - S_end[4, :]

            # 变成优化变量的交错顺序
            return _permute_p2x_cols(J)

        # --- 仅做“总剂量上限”（允许少用甚至不用） ---
        dt = (t_terminal - t_initial) / number_intervals
        maskI = np.zeros(number_intervals, dtype=bool); maskI[params.K_I_indices] = True
        maskM = np.zeros(number_intervals, dtype=bool); maskM[params.K_M_indices] = True

        # Caps are specified directly in week-units (integrated dose), not scaled by dt.
        # S_I = sum(vI_k * dt), S_M = sum(vM_k * dt).
        # Enforce: S_I ≤ D_I_cap_units, S_M ≤ D_M_cap_units.
        D_I_cap = p.D_I_cap_units
        D_M_cap = p.D_M_cap_units

        def g_dose_cap(x):
            vI = x[0::2]; vM = x[1::2]
            S_I = float(np.sum(vI[maskI]) * dt)
            S_M = float(np.sum(vM[maskM]) * dt)
            # g ≥ 0  ⇔  S ≤ cap
            # NOTE on units:
            # - vI, vM are per-week rates on each interval of length dt (weeks).
            # - Integrated dose on an interval is v * dt (week-units).
            # - Total caps D_*_cap are given directly in week-units.
            return np.array([D_I_cap - S_I, D_M_cap - S_M], dtype=float)

        def jac_dose_cap(x):
            Nloc = number_intervals
            dt = (t_terminal - t_initial) / Nloc
            maskI = np.zeros(Nloc, dtype=bool);
            maskI[params.K_I_indices] = True
            maskM = np.zeros(Nloc, dtype=bool);
            maskM[params.K_M_indices] = True

            # 参数顺序的雅可比（2 行 × 2N 列）
            J = np.zeros((2, 2 * Nloc), dtype=float)
            # 行0：cap_I - sum(vI*dt) 的导数 = -dt（只在允许的 I 段）
            for k in range(Nloc):
                if maskI[k]:
                    J[0, k] = -dt
            # 行1：cap_M - sum(vM*dt) 的导数 = -dt（只在允许的 M 段）
            for k in range(Nloc):
                if maskM[k]:
                    J[1, Nloc + k] = -dt

            return _permute_p2x_cols(J)

        # --- ramp 约束：相邻段变化不超过阈值 ---
        rhoI = 0.6
        rhoM = 1.5
        def g_ramp(x):
            vI = x[0::2]; vM = x[1::2]
            g = []
            for k in range(1, number_intervals):
                g += [rhoI - (vI[k] - vI[k-1]), rhoI + (vI[k] - vI[k-1])]
                g += [rhoM - (vM[k] - vM[k-1]), rhoM + (vM[k] - vM[k-1])]
            return np.array(g, dtype=float)

        def jac_ramp(x):
            Nloc = number_intervals
            rows = 4 * (Nloc - 1)
            J = np.zeros((rows, 2 * Nloc), dtype=float)  # 已是变量交错顺序

            r = 0
            for k in range(1, Nloc):
                i_prev = 2 * (k - 1)
                i_curr = 2 * k
                # rhoI - (vI_k - vI_{k-1})
                J[r, i_curr] = -1.0;
                J[r, i_prev] = 1.0;
                r += 1
                # rhoI + (vI_k - vI_{k-1})
                J[r, i_curr] = 1.0;
                J[r, i_prev] = -1.0;
                r += 1

                m_prev = 2 * (k - 1) + 1
                m_curr = 2 * k + 1
                # rhoM - (vM_k - vM_{k-1})
                J[r, m_curr] = -1.0;
                J[r, m_prev] = 1.0;
                r += 1
                # rhoM + (vM_k - vM_{k-1})
                J[r, m_curr] = 1.0;
                J[r, m_prev] = -1.0;
                r += 1

            return J

        # SLSQP 风格（dict）约束
        cons_slsqp = [
            {'type': 'ineq', 'fun': g_path},
            {'type': 'ineq', 'fun': g_dose_cap},
            {'type': 'ineq', 'fun': g_ramp},
        ]
        # trust-constr 风格约束（有限差分雅可比）
        cons_tc = [
            NonlinearConstraint(g_path, 0.0, np.inf, jac=jac_path),
            NonlinearConstraint(g_dose_cap, 0.0, np.inf, jac=jac_dose_cap),
            NonlinearConstraint(g_ramp, 0.0, np.inf, jac=jac_ramp),
        ]

        return cons_slsqp, cons_tc

    cons_slsqp, cons_tc = make_constraints()

    # Optional: quick feasibility check at start
    try:
        g0 = cons_slsqp[0]['fun'](x0)
        print('[DEBUG] min(g_path(x0)) =', np.min(g0))
    except Exception as _e:
        print('[DEBUG] g_path(x0) evaluation failed:', _e)

    bounds_tc = Bounds(lb, ub)

    # --- 第一轮：trust-constr ---
    t_start = time.time()
    res = minimize(fun_only,
                   x0,
                   method=method_primary,
                   jac=jac_only,
                   hess=hess_zero,
                   bounds=bounds_tc,
                   constraints=cons_tc,
                   options=solver_options_tc)

    # ---- Feasibility check & automatic fallback if infeasible ----
    def _min_constraint_residual(x, constraints):
        min_g = float('inf')
        for c in constraints:
            try:
                g = c['fun'](x)
                min_g = min(min_g, float(np.min(g)))
            except Exception:
                pass
        return min_g

    # 简单的约束最小残差检查函数保持不变
    min_g = _min_constraint_residual(res.x, cons_slsqp) if res.success else -np.inf
    if (not res.success) or (min_g < -1e-6):
        print('[INFO] Primary trust-constr 未收敛或不可行；切换 SLSQP 兜底...')
        res = minimize(fun_only,
                       res.x if hasattr(res, 'x') else x0,
                       method='SLSQP',
                       jac=jac_only,
                       bounds=list(zip(lb, ub)),
                       constraints=cons_slsqp,
                       options=solver_options_slsqp)
        min_g = _min_constraint_residual(res.x, cons_slsqp)
        print('[CHECK] min inequality residual after SLSQP =', min_g)

    system_parameters_opt = res.x
    J_opt = res.fun
    J_prev = J_opt

    # smoothing penalty loop
    # error_abs = 10 * tolerance_NLP
    # while error_abs > 2 * tolerance_NLP and penalty_times < 10:
    #     penalty_times += 1
    #
    #     # update for next iteration
    #     x0 = system_parameters_opt.copy()
    #     penalty_factor *= 5
    #     smooth_factor = 2 * tolerance_NLP / ((1 + np.sqrt(5)) * penalty_factor
    #                                          * number_pathconstraints_ineq
    #                                          * (t_terminal - t_initial))
    #     # resolve
    #     res = minimize(lambda x: obj_wrap(x, penalty_factor, smooth_factor),
    #                    x0,
    #                    method=method,
    #                    jac=True,
    #                    bounds=list(zip(lb, ub)),
    #                    options=solver_options)
    #     system_parameters_opt = res.x
    #     J_opt = res.fun
    #     error_abs = abs(J_prev - J_opt)
    #     J_prev = J_opt

    cputime_used = time.time() - t_start

    return system_parameters_opt, penalty_factor, smooth_factor