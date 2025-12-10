"""
Module for plotting optimal control results using matplotlib.

Provides a function to reproduce MATLAB-style figures for control and state variables
from a results dictionary containing time axes and variable values.
"""

import matplotlib.pyplot as plt
from params import Params
import numpy as np

from matplotlib import font_manager as fm

# --- Choose a CJK font that exists on this machine and set as default ---
def _set_cjk_font():
    candidates = [
        'PingFang SC',            # macOS default Chinese
        'Hiragino Sans GB',       # macOS
        'Songti SC', 'Heiti SC', 'STHeiti',   # macOS legacy
        'Microsoft YaHei',        # Windows
        'SimHei',                 # Windows/Linux common
        'Noto Sans CJK SC',       # Google Noto
        'Source Han Sans SC',     # Adobe/Google
        'Arial Unicode MS'        # Broad coverage
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = [name, 'DejaVu Sans']
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            # Show the minus sign correctly with CJK fonts
            plt.rcParams['axes.unicode_minus'] = False
            # Keep math text (e.g., $CD8^+$) using the default math fonts
            # while normal text uses the chosen CJK font.
            return name
    # If none found, warn once; figures will still save but with placeholders
    print("Warning: No CJK font found. Chinese glyphs may not render correctly.")
    return None

_CJK_FONT = _set_cjk_font()

params = Params()

def draw_results(results_dict):
    """
    Draws figures for control and state variables from the given results dictionary.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing keys:
        'ut_axis_stairs', 'control_parameters_stairs', 'ut_axis_plot',
        'control_parameters_plot', 'xt_axis', 'x_optimal',
        'number_state_variables_original', 'number_control_variables',
        't_initial', 't_terminal'.
    """
    # Extract variables from results_dict
    ut_axis_stairs = results_dict['ut_axis_stairs']
    control_parameters_stairs = results_dict['control_parameters_stairs']
    ut_axis_plot = results_dict['ut_axis_plot']
    control_parameters_plot = results_dict['control_parameters_plot']
    xt_axis = results_dict['xt_axis']
    x_optimal = results_dict['x_optimal']
    
    valid = np.isfinite(xt_axis) & np.all(np.isfinite(x_optimal), axis=1)
    xt_axis = xt_axis[valid]
    x_optimal = x_optimal[valid, :]
    x_optimal = np.clip(x_optimal, 1e-9, None)

    # convert weeks → days for plotting
    xt_days = xt_axis * 7.0

    # Derive counts and time bounds from data if not provided
    number_control_variables = control_parameters_stairs.shape[1]
    number_state_variables_original = x_optimal.shape[1]
    t_initial = results_dict.get('t_initial', ut_axis_stairs[0])
    t_terminal = results_dict.get('t_terminal', ut_axis_stairs[-1])

   
    default_control_labels = (
        ['u(t)'] if number_control_variables == 1
        else [r'vI (ICI 给药速率)', r'vM (化疗给药速率)'][:number_control_variables]
    )
    control_labels = results_dict.get('control_labels', default_control_labels)

    # 状态变量
    default_state_labels_full = [
        r'$X_s$ (敏感肿瘤细胞)',
        r'$X_r$ (耐药肿瘤细胞)',
        r'$L$ (CD8$^+$ T 细胞)',
        r'$I$ (ICI 浓度)',
        r'$M$ (化疗药物浓度)',
        r'$J$ (累计目标)'  
    ]
    
    default_state_labels = default_state_labels_full[:number_state_variables_original]
    state_labels = results_dict.get('state_labels', default_state_labels)

    # Set line width for plots
    plt.rcParams['lines.linewidth'] = 1.5

    # After computing ut_axis_stairs and ut_axis_plot, convert to days for plotting
    ut_days_stairs = ut_axis_stairs * 7.0
    ut_days_plot   = ut_axis_plot   * 7.0

    # Figure 1: staircase plot of control variables
    plt.figure(1)
    for i in range(number_control_variables):
        plt.step(ut_days_stairs, control_parameters_stairs[:, i], where='post')
    plt.xlim([xt_days[0], xt_days[-1]])
    plt.legend(control_labels, title='Controls')
    plt.xlabel('Days')
    plt.ylabel('Control (stairs)')
    plt.grid(True)
    plt.box(True)

    # Figure 2: continuous plot of control variables
    plt.figure(2)
    for i in range(number_control_variables):
        plt.plot(ut_days_plot, control_parameters_plot[:, i])
    plt.xlim([xt_days[0], xt_days[-1]])
    plt.legend(control_labels, title='Controls')
    plt.title('u(t) -- Continuous')
    plt.xlabel('Days')
    plt.ylabel('Control variables')
    plt.grid(True)

    # Figure 3: plot of original state variables
    plt.figure(3)
    for i in range(number_state_variables_original):
        plt.plot(xt_days, x_optimal[:, i], label=state_labels[i])
        
        try:
            plt.annotate(
                state_labels[i],
                xy=(xt_days[-1], x_optimal[-1, i]),
                xytext=(3, 0),
                textcoords='offset points',
                fontsize=9, va='center'
            )
        except Exception:
            pass
    plt.xlim([xt_days[0], xt_days[-1]])
    plt.legend(state_labels, title='States')
    plt.xlabel('Days')
    plt.ylabel('State variables')
    plt.grid(True)

   
    # Mark the four chemo cycles at the START of each cycle using params.K_M_indices
    chemo_idx = getattr(params, 'K_M_indices', [0, 1, 2, 3])
    chemo_times = []
    for idx in chemo_idx:
        if 0 <= idx < len(ut_days_stairs):
            chemo_times.append(ut_days_stairs[idx])  # 用“起点”而不是中点
    for t in chemo_times:
        plt.axvline(t, linestyle='--', linewidth=1.0, alpha=0.8, color='orange')

    # Show all figures
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        plt.tight_layout()
        plt.savefig(f"figure{fig_num}.png")
        plt.close(fig_num)

def draw_results_comparison(results_A, results_B, label_A='Ours', label_B='Reference',
                            show_xs_xr=False, log_tumor=False, save_path='figure_comparison.png'):
    """
    Compare two result dictionaries (A vs B) in a single 3-panel figure:
      (Top) Tumor burden X = Xs + Xr (optionally show Xs, Xr)
      (Mid) Effector T cells L
      (Bottom) Controls vI, vM (stairs)
    Vertical dashed lines mark the 4 chemo cycles indicated by params.K_M_indices.

    Both inputs must be results dicts returned by `deal_results`.
    """
    # --- Extract A ---
    ut_axis_stairs_A = results_A['ut_axis_stairs']
    ut_axis_plot_A   = results_A['ut_axis_plot']
    control_stairs_A = results_A['control_parameters_stairs']   # (N+1, 2)
    control_plot_A   = results_A['control_parameters_plot']     # (N, 2)
    xt_axis_A        = results_A['xt_axis']
    x_optimal_A      = results_A['x_optimal']                   # (T, 6)

    # --- Extract B ---
    ut_axis_stairs_B = results_B['ut_axis_stairs']
    ut_axis_plot_B   = results_B['ut_axis_plot']
    control_stairs_B = results_B['control_parameters_stairs']
    control_plot_B   = results_B['control_parameters_plot']
    xt_axis_B        = results_B['xt_axis']
    x_optimal_B      = results_B['x_optimal']

    # Convert to days
    xt_days_A = xt_axis_A * 7.0
    xt_days_B = xt_axis_B * 7.0
    ut_days_stairs_A = ut_axis_stairs_A * 7.0
    ut_days_stairs_B = ut_axis_stairs_B * 7.0

    # Chemo cycle markers: use K_M_indices on A's stairs (start-of-interval markers)
    try:
        from params import Params
        _p = Params()
        chemo_idx = getattr(_p, 'K_M_indices', [0,1,2,3])
    except Exception:
        chemo_idx = [0,1,2,3]
    chemo_lines_days = []
    for idx in chemo_idx:
        if 0 <= idx < len(ut_days_stairs_A):
            chemo_lines_days.append(ut_days_stairs_A[idx])

    # Build tumor burden
    Xa = x_optimal_A[:, 0] + x_optimal_A[:, 1]
    Xs_a, Xr_a = x_optimal_A[:, 0], x_optimal_A[:, 1]
    La = x_optimal_A[:, 2]

    Xb = x_optimal_B[:, 0] + x_optimal_B[:, 1]
    Xs_b, Xr_b = x_optimal_B[:, 0], x_optimal_B[:, 1]
    Lb = x_optimal_B[:, 2]

    # Controls (stairs) in days
    vI_stairs_A = control_stairs_A[:, 0]
    vM_stairs_A = control_stairs_A[:, 1] if control_stairs_A.shape[1] > 1 else None
    vI_stairs_B = control_stairs_B[:, 0]
    vM_stairs_B = control_stairs_B[:, 1] if control_stairs_B.shape[1] > 1 else None

    # Axis bounds (union)
    x_min = min(xt_days_A[0], xt_days_B[0])
    x_max = max(xt_days_A[-1], xt_days_B[-1])

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # ---- Top: Tumor burden ----
    ax = axes[0]
    if log_tumor:
        ax.semilogy(xt_days_A, Xa, label=f'{label_A}: X=Xs+Xr')
        ax.semilogy(xt_days_B, Xb, label=f'{label_B}: X=Xs+Xr', linestyle='--')
    else:
        ax.plot(xt_days_A, Xa, label=f'{label_A}: X=Xs+Xr')
        ax.plot(xt_days_B, Xb, label=f'{label_B}: X=Xs+Xr', linestyle='--')
    if show_xs_xr:
        ax.plot(xt_days_A, Xs_a, alpha=0.5, label=f'{label_A}: Xs')
        ax.plot(xt_days_A, Xr_a, alpha=0.5, label=f'{label_A}: Xr')
        ax.plot(xt_days_B, Xs_b, alpha=0.5, linestyle='--', label=f'{label_B}: Xs')
        ax.plot(xt_days_B, Xr_b, alpha=0.5, linestyle='--', label=f'{label_B}: Xr')
    for t in chemo_lines_days:
        ax.axvline(t, linestyle='--', linewidth=1.0, alpha=0.8, color='orange')
    ax.set_ylabel('Tumor burden')
    ax.set_xlim([x_min, x_max])
    ax.grid(True)
    ax.legend()

    # ---- Mid: Effector T cells L ----
    ax = axes[1]
    ax.plot(xt_days_A, La, label=f'{label_A}: L')
    ax.plot(xt_days_B, Lb, label=f'{label_B}: L', linestyle='--')
    for t in chemo_lines_days:
        ax.axvline(t, linestyle='--', linewidth=1.0, alpha=0.8, color='orange')
    ax.set_ylabel('Effector T cells (L)')
    ax.grid(True)
    ax.legend()

    # ---- Bottom: Controls (stairs) ----
    ax = axes[2]
    ax.step(ut_days_stairs_A, vI_stairs_A, where='post', label=f'{label_A}: vI')
    if vM_stairs_A is not None:
        ax.step(ut_days_stairs_A, vM_stairs_A, where='post', label=f'{label_A}: vM')

    ax.step(ut_days_stairs_B, vI_stairs_B, where='post', linestyle='--', label=f'{label_B}: vI')
    if vM_stairs_B is not None:
        ax.step(ut_days_stairs_B, vM_stairs_B, where='post', linestyle='--', label=f'{label_B}: vM')

    for t in chemo_lines_days:
        ax.axvline(t, linestyle='--', linewidth=1.0, alpha=0.8, color='orange')

    ax.set_xlim([x_min, x_max])
    ax.set_xlabel('Days')
    ax.set_ylabel('Controls')
    ax.grid(True)
    ax.legend(ncols=2)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path
