# params.py
class Params:

    alpha_sr   = 0.7
    alpha_rs   = 0.9
    a_s        = 0.01
    a_r        = 0.014   
    X_max      = 1.0
    mu0        = 0.6
    # Maximum chemo‐induced upregulation of checkpoints (Assumption 3: 0.2)
    p          = 0.2   
    d          = 1.0
    l_s        = 1.0
    l          = 2.0
    m_L        = 0.05
    r          = 0.09
    j_C        = 1.0
    k_C        = 0.03
    j_L        = 0.18   
    k_L        = 0.01
    j_M        = 1.0
    k_M        = 0.03
    L_max      = 1.0
    h_L        = 0.005   
    eta_RL     = 0.005
    q          = 0.03    
    u_bio      = 0.03   
    gamma_I    = 0.2428
    gamma_M    = 3.24   
    # gamma_vI and gamma_vM are unused and removed
    K_xs       = 0.6   
    # Chemo‐induced killing effect on CD8+ T cells (Assumption 5: 0.6)
    K_L        = 0.6   
    w1         = 10.0  
    w2         = 3.0   
    w3         = 0.8   
    w4         = 1.8  
    N          = 20

    s = 8.39/100  
    # DM = K_xs * (1 - exp(-M))
    K_M_indices = [0, 1, 2, 3]
    K_I_indices = list(range(10))

    tau_days = 3.0
    tau_weeks = tau_days / 7.0
    D_I_cap_units = 6.0   
    D_M_cap_units = 4.0


def apply_params(**kwa):
    """
    Update Params class attributes at runtime and reload dependent modules
     so that new defaults take effect across the  project.
     """
    import importlib
    # 1) update class attributes
    for k, v in kwa.items():
        setattr(Params, k, v)
    # 2) reload modules that instantiate/use Params() at import time
    import ode_system, nlp_objective, simulation_optimal, draw_results
    importlib.reload(ode_system)
    importlib.reload(nlp_objective)
    importlib.reload(simulation_optimal)
    importlib.reload(draw_results)
