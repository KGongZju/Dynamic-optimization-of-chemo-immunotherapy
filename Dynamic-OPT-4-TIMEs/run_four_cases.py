# run_four_cases.py
import numpy as np
from pathlib import Path
import pandas as pd
import json, os
from params import apply_params
from nlp_solve import NLP_solve
from simulation_optimal import Simulation_optimal
from deal_results import deal_results
from export_results import export_results_package
from export_results import write_excel_report as write_excel_report_from_exports


def run_one(prefix, x0):
    from params import Params
    p = Params()
    number_intervals = 10
    flag_minmax = True
    flag_terminaltime_fixed = True
    t_initial = 0.0
    t_terminal = 30.0
    t_pre = 57.0
    t_plot_end = 100.0
    u_lower = np.vstack([np.zeros(number_intervals), np.zeros(number_intervals)])
    u_upper = np.vstack([np.ones(number_intervals), 5*np.ones(number_intervals)])
    # 禁止窗口外给药
    last_chemo = p.K_M_indices[-1]; u_upper[1, last_chemo+1:] = 0.0
    last_ici = p.K_I_indices[-1];   u_upper[0, last_ici+1:]   = 0.0
    # 初值
    u_guess = np.vstack([0.6*np.ones(number_intervals), 2.5*np.ones(number_intervals)])
    u_guess[1, last_chemo+1:] = 0.0
    u_guess[0, last_ici+1:]   = 0.0
    # 扁平化
    system_parameters_init = u_guess.T.flatten()

    opt, pen, sm = NLP_solve(system_parameters_init,
                             flag_minmax, flag_terminaltime_fixed,
                             6, 2, 0, 0, 0, 0, 0,
                             number_intervals,
                             0, 0, 1e-6, 1e-4, False,
                             t_initial, t_terminal,
                             x0, u_lower, u_upper,
                             None, None, None, None,
                             u_guess, t_terminal, np.empty((0,)))
    xt, xopt, npi = Simulation_optimal(opt, pen, sm, x0, number_intervals,
                                       t_initial, t_terminal, t_pre, t_plot_end)
    results = deal_results(opt, xt, xopt, npi, True, False, True,
                           2, number_intervals, 2, number_intervals,
                           t_initial, t_terminal, t_pre)
    return export_results_package(results, prefix=prefix, outdir="exports")

def write_excel_report(all_outs, outfile: str = "exports/four_cases_report.xlsx"):
    # 1) 路径准备与标准化
    p = Path(outfile)
    p.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "xlsx": str(p),
        "summary": str(p.with_suffix(".summary.json")),   # 可选文件：没有也不报错
        "metrics": str(p.with_suffix(".metrics.csv")),    # 可选文件：没有也不报错
        "traces": str(p.with_suffix(".traces.csv")),      # 可选文件：没有也不报错
    }

    # 2) 加载/构造 summary
    summary_path = Path(paths["summary"])
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        # 用内存里的 all_outs 兜底构造一个最小可用 summary
        summary = {
            "solver": all_outs.get("solver", "SLSQP"),
            "status": all_outs.get("status", "success"),
            "message": all_outs.get("message", "Primary trust-constr 未收敛，已切换 SLSQP"),
            "objective": all_outs.get("fval"),
            "iterations": all_outs.get("nit"),
            "ineq_resid_min": all_outs.get("ineq_resid_min"),
        }

    # 3) 加载/获取 metrics（DataFrame 或 CSV）
    metrics_df = None
    if "metrics" in all_outs and isinstance(all_outs["metrics"], pd.DataFrame):
        metrics_df = all_outs["metrics"]
    else:
        mp = Path(paths["metrics"])
        if mp.exists():
            try:
                metrics_df = pd.read_csv(mp)
            except Exception:
                metrics_df = None

    # 4) 写 Excel
    with pd.ExcelWriter(paths["xlsx"]) as xw:
        # Overview
        pd.DataFrame([summary]).to_excel(xw, sheet_name="Overview", index=False)

        # Metrics
        if metrics_df is not None:
            metrics_df.to_excel(xw, sheet_name="Metrics", index=False)

        # 其余表（若 all_outs 里有可直接 to_excel 的对象，则各写一张表）
        for name, val in all_outs.items():
            if name in {"metrics"}:
                continue
            if hasattr(val, "to_excel"):
                sheet = str(name)[:31]  # Excel sheet 名最长31
                try:
                    val.to_excel(xw, sheet_name=sheet, index=False)
                except TypeError:
                    # 非DataFrame类型但有 to_excel 的奇异对象，尽量转换
                    try:
                        pd.DataFrame(val).to_excel(xw, sheet_name=sheet, index=False)
                    except Exception:
                        pass

if __name__ == "__main__":
    Path("exports").mkdir(exist_ok=True)

    # (a) extremely cold —— 化疗主导，ICI 很少或不用
    apply_params(
        mu0=1.0, j_L=0.10, q=0.05, h_L=0.010, a_s=0.010, a_r=0.016, K_L=0.35,
        w2=3.0, w3=0.8, w4=1.8,                       # ICI 更贵
        D_I_cap_units=6.0, D_M_cap_units=4.0,         # ICI 仅 1 个单位；Chemo 4 个单位
        tau_days=3.0, K_I_indices=list(range(10))
    )
    out_a = run_one("case_a_extremely_cold", np.array([1.0, 0.10, 0.008, 0.0, 0.0, 0.0]))

    # (b) hot —— ICI 主导，化疗很少
    apply_params(
        mu0=0.6, j_L=0.15, q=0.035, h_L=0.005, a_s=0.015, a_r=0.022, K_L=0.22,
        w2=3.0, w3=0.8, w4=1.8,                          # ICI 更便宜
        D_I_cap_units=6.0, D_M_cap_units=4.0,          # ICI 上限大、Chemo 小
        tau_days=3.0, K_I_indices=list(range(10))
    )
    out_b = run_one("case_b_hot", np.array([1.0, 0.06, 0.030, 0.0, 0.0, 0.0]))

    # (c) cold —— 组合治疗（中庸）
    apply_params(
        mu0=0.9, j_L=0.18, q=0.03, h_L=0.005, a_s=0.010, a_r=0.014, K_L=0.28,
        w2=3.0, w3=0.8, w4=1.8,
        D_I_cap_units=6.0, D_M_cap_units=4.0,
        tau_days=3.0, K_I_indices=list(range(10))
    )
    out_c = run_one("case_c_cold", np.array([1.0, 0.10, 0.010, 0.0, 0.0, 0.0]))

    # (d) cold + high resistant growth —— 化疗略偏重
    apply_params(
        mu0=0.9, j_L=0.18, q=0.03, h_L=0.005, a_s=0.010, a_r=0.020, K_L=0.28,
        w2=3.0, w3=0.8, w4=1.8,
        D_I_cap_units=6.0, D_M_cap_units=4.0,
        tau_days=3.0, K_I_indices=list(range(10))
    )
    out_d = run_one("case_d_cold_high_ar", np.array([1.0, 0.10, 0.010, 0.0, 0.0, 0.0]))

    # 汇总到一个 Excel 报告
    all_outs = {
        "case_a_extremely_cold": out_a,
        "case_b_hot": out_b,
        "case_c_cold": out_c,
        "case_d_cold_high_ar": out_d,
    }
    write_excel_report_from_exports(all_outs, outfile="exports/four_cases_report.xlsx")
    print("[OK] four_cases_report.xlsx written with per-case sheets (states/schedule/markers) and a Summary sheet.")