# plot_from_exports.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def stairs_from_intervals(start_day, end_day, values):
    """
    根据每段起止(day) + 段值 -> 生成 step(where='post') 需要的 (t_stairs, v_stairs)
    t_stairs: [start0, start1, ..., startN-1, endN-1]  (长度 N+1)
    v_stairs: [v0,     v1,     ..., vN-1,              vN-1] (长度 N+1)
    """
    start_day = np.asarray(start_day)
    end_day   = np.asarray(end_day)
    values    = np.asarray(values)
    assert len(start_day) == len(end_day) == len(values), "interval arrays must match length"

    t_stairs = np.concatenate([start_day, [end_day[-1]]])
    v_stairs = np.concatenate([values, [values[-1]]])
    return t_stairs, v_stairs

def _require_cols(df: pd.DataFrame, must_have):
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def plot_from_exports(prefix="ours_run1", outdir="exports",
                      save_path="figure_from_exports.png",
                      log_tumor=False, shade_chemo=True):
    # ---------- 1) 路径 ----------
    f_states  = os.path.join(outdir, f"{prefix}_timeseries_states.csv")
    f_sched   = os.path.join(outdir, f"{prefix}_controls_schedule.csv")
    f_markers = os.path.join(outdir, f"{prefix}_chemo_markers_days.csv")
    f_summary = os.path.join(outdir, f"{prefix}_summary.json")

    for f in [f_states, f_sched, f_markers, f_summary]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing export file: {f}")

    # ---------- 2) 读取数据（pandas，更健壮） ----------
    states = pd.read_csv(f_states)
    # 统一列名（容错）
    ren = {"t": "day", "days": "day", "time_day": "day"}
    states = states.rename(columns={k: v for k, v in ren.items() if k in states.columns})

    # 自动生成 X_total
    if "X_total" not in states.columns and {"Xs","Xr"}.issubset(states.columns):
        states["X_total"] = states["Xs"] + states["Xr"]

    _require_cols(states, ["day", "X_total", "L"])
    day = states["day"].to_numpy()
    X   = states["X_total"].to_numpy()
    L   = states["L"].to_numpy()
    I   = states["I"].to_numpy() if "I" in states.columns else None
    M   = states["M"].to_numpy() if "M" in states.columns else None

    # schedule: start_day,end_day,mid_day,vI,vM,doseI_week,doseM_week,is_chemo,interval
    sched = pd.read_csv(f_sched)
    # 兼容大小写/不同命名
    sched = sched.rename(columns={
        "Interval":"interval", "v_I":"vI", "v_M":"vM",
        "doseI_w":"doseI_week", "doseM_w":"doseM_week"
    })
    _require_cols(sched, ["start_day","end_day","vI","vM","interval"])
    if "is_chemo" not in sched.columns:
        sched["is_chemo"] = 0

    start_day = sched["start_day"].to_numpy()
    end_day   = sched["end_day"].to_numpy()
    vI        = sched["vI"].to_numpy()
    vM        = sched["vM"].to_numpy()
    is_chemo  = sched["is_chemo"].astype(int).to_numpy()

    # chemo markers
    mk_df = pd.read_csv(f_markers)
    # 允许列名 day / 任意以 day 开头
    day_col = next((c for c in mk_df.columns if c.lower().startswith("day")), mk_df.columns[0])
    chemo_markers = mk_df[day_col].to_numpy()

    # summary（可用于校验横轴是否 ~700 days）
    with open(f_summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # ---------- 3) 生成阶梯数据 ----------
    tI_stairs, vI_stairs = stairs_from_intervals(start_day, end_day, vI)
    tM_stairs, vM_stairs = stairs_from_intervals(start_day, end_day, vM)

    # ---------- 4) 作图 ----------
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # Top: Tumor burden
    ax = axes[0]
    X_plot = np.clip(X, 1e-12, None)
    if log_tumor:
        ax.semilogy(day, X_plot, label=r'$X=X_s+X_r$')
    else:
        ax.plot(day, X_plot, label=r'$X=X_s+X_r$')
    ax.set_ylabel('Tumor burden')
    ax.grid(True)
    ax.legend()

    # 标注 chemo 段（可选）
    if shade_chemo:
        for k in range(len(start_day)):
            if is_chemo[k] == 1:
                ax.axvspan(start_day[k], end_day[k], alpha=0.08)

    for t in chemo_markers:
        ax.axvline(t, linestyle='--', linewidth=1.0, alpha=0.8)

    # Mid: Effector T cells L (+ drugs if present)
    ax = axes[1]
    ax.plot(day, L, label='L (CD8+ T)')
    if I is not None:
        ax.plot(day, I, linestyle=':', label='I')
    if M is not None:
        ax.plot(day, M, linestyle='--', label='M')
    ax.set_ylabel('Effector T / Drugs')
    ax.grid(True)
    ax.legend()
    if shade_chemo:
        for k in range(len(start_day)):
            if is_chemo[k] == 1:
                ax.axvspan(start_day[k], end_day[k], alpha=0.08)
    for t in chemo_markers:
        ax.axvline(t, linestyle='--', linewidth=1.0, alpha=0.8)

    # Bottom: controls
    ax = axes[2]
    ax.step(tI_stairs, vI_stairs, where='post', label='vI (ICI)')
    ax.step(tM_stairs, vM_stairs, where='post', label='vM (Chemo)')
    ax.set_xlabel('Days')
    ax.set_ylabel('Controls')
    ax.grid(True)
    ax.legend(ncols=2)
    for t in chemo_markers:
        ax.axvline(t, linestyle='--', linewidth=1.0, alpha=0.8)

    # x 轴范围：按状态时间序列范围
    ax.set_xlim([day.min(), day.max()])

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {save_path}")
    return save_path

if __name__ == "__main__":
    # 默认从 'exports/ours_run1_*.csv' 读取
    plot_from_exports(prefix="ours_run1", outdir="exports",
                      save_path="figure_from_exports.png",
                      log_tumor=False, shade_chemo=True)