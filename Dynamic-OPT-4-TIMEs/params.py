# params.py
class Params:
    # ==== 冷肿瘤（cold TIME）预设说明 ====
    # 目标：η_LT 介于 0.006 和 0.03 之间，且 CD8+T 的净增殖速度 < 1.2×肿瘤增殖速度
    # 做法：降低Treg与肿瘤抑制（h=0.005, q=0.03为论文拟合值），保持 r≈0.1，
    # 并设 ar > as（例如 ar≈1.4×as）以反映耐药细胞更快的内禀生长。
    # j_L=0.115 以反映由CD8+T诱导的免疫原性死亡对抗原呈递的贡献（偏低，符合冷肿瘤）。
    # ===================================
    # 这里放所有前面定义好的参数……
    alpha_sr   = 0.7
    alpha_rs   = 0.9
    a_s        = 0.01
    a_r        = 0.014   # 令 ar ≈ 1.4*as，体现耐药增殖更快（冷肿瘤更常见）
    X_max      = 1.0
    mu0        = 0.6
    # Maximum chemo‐induced upregulation of checkpoints (Assumption 3: 0.2)
    p          = 0.2   # 化疗诱导PD-L1上调最大幅度
    d          = 1.0
    l_s        = 1.0
    l          = 2.0
    m_L        = 0.05
    r          = 0.09
    j_C        = 1.0
    k_C        = 0.03
    j_L        = 0.18   # CD8+T诱导的免疫原性肿瘤死亡所致抗原呈递速率（匹配论文估计值）
    k_L        = 0.01
    j_M        = 1.0
    k_M        = 0.03
    L_max      = 1.0
    h_L        = 0.005   # 由Treg对CD8+T的抑制率h（匹配论文估计值）
    eta_RL     = 0.005
    q          = 0.03    # 肿瘤对CD8+T抑制强度（匹配论文估计值，冷/热判别用）
    u_bio      = 0.03    # 根据论文设定或补充
    gamma_I    = 0.2428
    gamma_M    = 3.24   # 化疗药物衰减率
    # gamma_vI and gamma_vM are unused and removed
    K_xs       = 0.6   # 化疗对敏感肿瘤细胞杀伤系数
    # Chemo‐induced killing effect on CD8+ T cells (Assumption 5: 0.6)
    K_L        = 0.6   # 化疗对CD8+T杀伤系数
    w1         = 10.0   # 免疫收益权重
    w2         = 3.0   # 肿瘤负担惩罚权重
    w3         = 0.8   # 化疗剂量成本权重
    w4         = 1.8   # ICI 注射成本权重
    N          = 20
    # DL 的分母“陡峭度”系数（s）与抑制项常数 u_bio
    s = 8.39/100  # 可按标定调整
    # 保持 K_xs 不变；DM = K_xs * (1 - exp(-M))
    K_M_indices = [0, 1, 2, 3]
    K_I_indices = list(range(10))
    # 给药窗口（周）：1~2 天通常能出现明显“上-下”
    tau_days = 3.0
    tau_weeks = tau_days / 7.0
    D_I_cap_units = 6.0   # 缺省最多允许 6 个“周单位”
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