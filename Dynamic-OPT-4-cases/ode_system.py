import numpy as np
from params import Params
params = Params()
def ODE_system(t, y, u, delta, params):

    n = 6
    P = len(delta)  # =2*N
    Xs, Xr, L, I, M, J = y[:n]
    vI, vM = u
    # Clamp states used inside nonlinear denominators to avoid numerical negatives
    Xs_pos = max(Xs, 0.0)
    Xr_pos = max(Xr, 0.0)
    L_pos  = max(L, 0.0)
    I_pos  = min(max(I, 0.0), 1.0)
    M_pos  = max(M, 0.0)
    X = Xs_pos + Xr_pos

    eps = 1e-8
    # raw fraction of CD8+ T suppressed by checkpoints (affected by I and chemo)
    mu_raw = params.mu0 * (1 - I_pos) * (1 + params.p * (1 - np.exp(-M_pos)))
    # enforce physical range [0,1]
    mu = min(max(mu_raw, 0.0), 1.0)
    Z = L_pos * (1 - mu) / (X + eps)
    DL = params.d * (Z ** params.l) / (params.s + Z ** params.l)
    DM = params.K_xs * (1 - np.exp(-M_pos))

    # derivatives for sensitivity (zero when clipping is active)
    if mu == mu_raw:
        dmu_dI = -params.mu0 * (1 + params.p * (1 - np.exp(-M_pos)))
        dmu_dM = params.mu0 * (1 - I_pos) * params.p * np.exp(-M_pos) * (1.0 if M_pos > 0.0 else 0.0)
    else:
        dmu_dI = 0.0
        dmu_dM = 0.0

    dZ_dXs = -L_pos * (1 - mu) / (X + eps) ** 2
    dZ_dXr = -L_pos * (1 - mu) / (X + eps) ** 2
    dZ_dL  = (1 - mu) / (X + eps)
    dZ_dI  = L_pos * (-dmu_dI) / (X + eps)
    dZ_dM  = L_pos * (-dmu_dM) / (X + eps)

    dDL_dZ = params.d * (params.l * params.s) * (Z ** (params.l - 1)) / (params.s + Z ** params.l) ** 2
    dDL_dXs = dDL_dZ * dZ_dXs
    dDL_dXr = dDL_dZ * dZ_dXr
    dDL_dL = dDL_dZ * dZ_dL
    dDL_dI = dDL_dZ * dZ_dI
    dDL_dM = dDL_dZ * dZ_dM

    dDM_dXs = 0.0
    dDM_dM = params.K_xs * np.exp(-M_pos) if M_pos > 0.0 else 0.0


    # Sensitive tumor: logistic growth minus immune kill and chemo kill
    Phi_comp = (Xs + params.alpha_sr * Xr) / params.X_max
    f1 = params.a_s * Xs * (1 - Phi_comp) - DL * Xs - DM * Xs

    # Resistant tumor: logistic growth minus immune kill
    Psi_comp = (Xr + params.alpha_rs * Xs) / params.X_max
    f2 = params.a_r * Xr * (1 - Psi_comp) - DL * Xr

    T1 = params.j_C * X / (params.k_C + X + eps)
    T2 = params.j_L * DL * X / (params.k_L + DL * X + eps)
    den3 = params.k_M + DM * Xs_pos
    den3_safe = den3 if den3 > eps else eps
    T3 = params.j_M * DM * Xs_pos / den3_safe
    U = T1 + T2 + T3

    f3 = (- params.m_L * L
          + params.r * U * L * (1 - L / params.L_max)
          - params.h_L * params.eta_RL * L
          - params.q * X / (X + params.u_bio + eps) * L
          - params.K_L * (1 - np.exp(-M)) * L)

    f4 = - params.gamma_I * I + vI
    f5 = - params.gamma_M * M + vM

    f6 = (params.w2 * (Xs + Xr)
          + params.w3 * vM
          + params.w4 * vI
          - params.w1 * L)


    dydt = np.zeros(n + P * n)
    dydt[:n] = [f1, f2, f3, f4, f5, f6]



    # ∂Phi_comp/∂Xs = 1/X_max, ∂Phi_comp/∂Xr = alpha_sr/X_max
    df1_dXs = ( params.a_s * (1 - Phi_comp)
               - params.a_s * Xs * (1.0/params.X_max)
               - DL
               - Xs * dDL_dXs
               - DM
               - Xs * dDM_dXs )
    df1_dXr = - params.a_s * Xs * (params.alpha_sr / params.X_max) - Xs * dDL_dXr
    df1_dL  = - Xs * dDL_dL
    df1_dI  = - Xs * dDL_dI
    df1_dM  = - Xs * dDL_dM - Xs * dDM_dM

    # ∂Psi_comp/∂Xs = alpha_rs/X_max, ∂Psi_comp/∂Xr = 1/X_max
    df2_dXs = - params.a_r * Xr * (params.alpha_rs / params.X_max) \
              - Xr * dDL_dXs
    df2_dXr = (params.a_r * (1 - Psi_comp)
               - params.a_r * Xr * (1.0 / params.X_max)
               - DL
               - Xr * dDL_dXr)
    df2_dL  = - Xr * dDL_dL
    df2_dI  = - Xr * dDL_dI
    df2_dM  = - Xr * dDL_dM
    # f3
    dT1_dX = params.j_C * params.k_C / (params.k_C + X + eps) ** 2
    dT1_dXs = dT1_dX
    dT1_dXr = dT1_dX

    Q2 = DL * X
    dT2_dDL = params.j_L * X * params.k_L / (params.k_L + Q2 + eps) ** 2
    dT2_dX = params.j_L * DL * params.k_L / (params.k_L + Q2 + eps) ** 2
    dT2_dXs = dT2_dX * 1 + dT2_dDL * dDL_dXs
    dT2_dXr = dT2_dX * 1 + dT2_dDL * dDL_dXr
    dT2_dL = dT2_dDL * dDL_dL
    dT2_dI = dT2_dDL * dDL_dI
    dT2_dM = dT2_dDL * dDL_dM

    Q3 = DM * Xs_pos
    den3 = params.k_M + Q3
    den3_safe = den3 if den3 > eps else eps
    # dDM/dM with clamp
    dT3_dDM = params.j_M * Xs_pos * params.k_M / (den3_safe ** 2)
    dT3_dXs = params.j_M * params.k_M * DM / (den3_safe ** 2)
    dT3_dM  = dT3_dDM * dDM_dM
    dT3_dL = 0.0
    dT3_dI = 0.0

    dU_dXs = dT1_dXs + dT2_dXs + dT3_dXs
    dU_dXr = dT1_dXr + dT2_dXr
    dU_dL = dT2_dL
    dU_dI = dT2_dI
    dU_dM = dT2_dM + dT3_dM

    df3_dXs = params.r * dU_dXs * L * (1 - L / params.L_max) \
              - params.q * (params.u_bio) * L / (X + params.u_bio + eps) ** 2
    df3_dXr = params.r * dU_dXr * L * (1 - L / params.L_max) \
              - params.q * (params.u_bio) * L / (X + params.u_bio + eps) ** 2
    df3_dL = - params.m_L \
             + params.r * U * (1 - 2 * L / params.L_max) \
             + params.r * dU_dL * L * (1 - L / params.L_max) \
             - params.h_L * params.eta_RL \
             - params.q * X / (X + params.u_bio + eps) \
             - params.K_L * (1 - np.exp(-M))
    df3_dI = params.r * dU_dI * L * (1 - L / params.L_max)
    df3_dM = params.r * dU_dM * L * (1 - L / params.L_max) - params.K_L * np.exp(-M) * L
    # f4
    df4_dI = - params.gamma_I
    # f5
    df5_dM = - params.gamma_M
    # f6
    df6_dXs  = params.w2
    df6_dXr  = params.w2
    df6_dL   = -params.w1
    df6_dI   = 0.0
    df6_dM   = 0.0
    df6_dvI  = params.w4
    df6_dvM  = params.w3


    for j in range(P):
        base = n + j * n
        s = y[base:base + n]
        δ = delta[j]
        ds = np.zeros(n, dtype=float)
        # ds1
        ds[0] = df1_dXs * s[0] + df1_dXr * s[1] + df1_dL * s[2] + df1_dI * s[3] + df1_dM * s[4]
        # ds2
        ds[1] = df2_dXs * s[0] + df2_dXr * s[1] + df2_dL * s[2] + df2_dI * s[3] + df2_dM * s[4]
        # ds3
        ds[2] = df3_dXs * s[0] + df3_dXr * s[1] + df3_dL * s[2] + df3_dI * s[3] + df3_dM * s[4]

        half = P // 2
        ds[3] = df4_dI * s[3] + (δ if j < half else 0)

        ds[4] = df5_dM * s[4] + (δ if j >= half else 0)

        ds6 = ( df6_dXs*s[0]
               + df6_dXr*s[1]
               + df6_dL *s[2]
               + df6_dI *s[3]
               + df6_dM *s[4]
             )
        ds6 += (df6_dvI * δ if j < half else 0) \
             + (df6_dvM * δ if j >= half else 0)
        ds[5] = ds6

        dydt[base:base + n] = ds

    return dydt
