import numpy as np
from params import apply_params

import pandas as pd
import json, os

# ---- unified exporter to CSV/JSON, returns written file paths ----
from typing import Dict, Any

def _to_dataframe(obj, default_columns=None):
    import pandas as pd
    import numpy as np
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj, columns=default_columns if default_columns else None)
    if hasattr(obj, "shape"):
        # numpy array
        return pd.DataFrame(obj, columns=default_columns if default_columns else None)
    raise TypeError("Unsupported data structure for DataFrame export.")


def export_results_package(results: Dict[str, Any], prefix: str, outdir: str) -> Dict[str, str]:
    """Write results to CSV/JSON with a consistent naming scheme and return their file paths.
    Expected keys (if present):
      - 'timeseries_states' (DataFrame/dict/ndarray) -> {prefix}_timeseries_states.csv
      - 'controls_schedule' (DataFrame/dict/ndarray) -> {prefix}_controls_schedule.csv
      - 'chemo_markers'    (list/array/DataFrame)    -> {prefix}_chemo_markers_days.csv (column 'day')
      - 'summary'          (dict)                    -> {prefix}_summary.json
    The function is permissive and will export whatever keys are present.
    """
    import os, json
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)
    paths = {}

    # 1) states
    if "timeseries_states" in results:
        states = results["timeseries_states"]
        # try to preserve common column names if provided
        default_cols = results.get("states_columns", ["day","Xs","Xr","L","I","M","mu","X_total","nLT"])
        df_states = _to_dataframe(states, default_columns=default_cols)
        p_states = os.path.join(outdir, f"{prefix}_timeseries_states.csv")
        df_states.to_csv(p_states, index=False)
        paths["timeseries_states"] = p_states

    # 2) controls
    if "controls_schedule" in results:
        controls = results["controls_schedule"]
        default_cols = results.get("controls_columns", ["interval","vI","vM","doseI_week","doseM_week"])
        df_ctrl = _to_dataframe(controls, default_columns=default_cols)
        p_ctrl = os.path.join(outdir, f"{prefix}_controls_schedule.csv")
        df_ctrl.to_csv(p_ctrl, index=False)
        paths["controls_schedule"] = p_ctrl

    # 3) chemo markers (days)
    if "chemo_markers" in results:
        mk = results["chemo_markers"]
        if isinstance(mk, pd.DataFrame):
            df_mk = mk[[c for c in mk.columns if c.lower().startswith("day") or c.lower()=="day"]]
            if df_mk.shape[1] == 0:
                df_mk = mk.copy()
        else:
            df_mk = pd.DataFrame({"day": mk})
        p_mk = os.path.join(outdir, f"{prefix}_chemo_markers_days.csv")
        df_mk.to_csv(p_mk, index=False)
        paths["chemo_markers"] = p_mk

    # 4) summary (dict)
    if "summary" in results and isinstance(results["summary"], dict):
        p_js = os.path.join(outdir, f"{prefix}_summary.json")
        with open(p_js, "w", encoding="utf-8") as f:
            json.dump(results["summary"], f, ensure_ascii=False, indent=2)
        paths["summary"] = p_js
    else:
        # create a minimal summary from states if available
        try:
            if "timeseries_states" in results:
                df_states = _to_dataframe(results["timeseries_states"])
                last = df_states.iloc[-1].to_dict()
                summary = {
                    "N_intervals": int(results.get("N_intervals", 10)),
                    "t_pre_days": float(results.get("t_pre_days", 0.0)),
                    "t_treatment_days": float(results.get("t_treatment_days", 0.0)),
                    "t_post_days": float(results.get("t_post_days", 0.0)),
                    "total_days": float(results.get("total_days", df_states.shape[0] if "day" not in df_states else df_states["day"].max())),
                    "final_X": float(last.get("X_total", last.get("X", 0.0))),
                    "final_L": float(last.get("L", 0.0)),
                }
                p_js = os.path.join(outdir, f"{prefix}_summary.json")
                with open(p_js, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                paths["summary"] = p_js
        except Exception as e:
            print(f"[WARN] Failed to auto-create summary from states: {e}")

    return paths

def run_one(prefix, x0):
    from deal_results import deal_results
    results = deal_results(x0)
    return export_results_package(results, prefix=prefix, outdir="exports")

def write_excel_report(case_outputs: dict, outfile="exports/four_cases_report.xlsx"):
    """Aggregate all four cases into a single Excel workbook.
    case_outputs: {case_name: {"timeseries_states": path, "controls_schedule": path, "chemo_markers": path, "summary": path}}
    """
    # Read summaries for the overview sheet
    summary_rows = []
    for name, paths in case_outputs.items():
        with open(paths["summary"], "r", encoding="utf-8") as f:
            s = json.load(f)
        summary_rows.append({
            "case": name,
            "N_intervals": s.get("N_intervals"),
            "t_pre_days": s.get("t_pre_days"),
            "t_treatment_days": s.get("t_treatment_days"),
            "t_post_days": s.get("t_post_days"),
            "total_days": s.get("total_days"),
            "cum_dose_I_week": s.get("cum_dose_I_week"),
            "cum_dose_M_week": s.get("cum_dose_M_week"),
            "final_X": s.get("final_X"),
            "final_L": s.get("final_L"),
            "min_X": s.get("min_X"),
            "day_at_min_X": s.get("day_at_min_X"),
            "min_L": s.get("min_L"),
            "day_at_min_L": s.get("day_at_min_L"),
        })

    df_summary = pd.DataFrame(summary_rows)

    # Build controls side-by-side comparison (by interval)
    controls_comp = None
    for name, paths in case_outputs.items():
        dfc = pd.read_csv(paths["controls_schedule"])  # expects columns: interval,vI,vM,doseI_week,doseM_week
        cols = ["interval", "vI", "vM", "doseI_week", "doseM_week"]
        dfc = dfc[cols].copy()
        dfc.columns = ["interval",
                       f"{name}_vI", f"{name}_vM",
                       f"{name}_doseI_w", f"{name}_doseM_w"]
        controls_comp = dfc if controls_comp is None else controls_comp.merge(dfc, on="interval", how="outer")
    if controls_comp is not None:
        controls_comp = controls_comp.sort_values("interval")

    # Write all sheets into one workbook
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with pd.ExcelWriter(outfile) as writer:
        # Summary
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        try:
            writer.sheets["Summary"].freeze_panes(1, 1)
        except Exception:
            pass

        # Per-case detailed sheets
        for name, paths in case_outputs.items():
            df_states = pd.read_csv(paths["timeseries_states"])   # day, Xs, Xr, L, I, M, mu, X_total, nLT
            df_sched  = pd.read_csv(paths["controls_schedule"])  # schedule per-interval
            df_marks  = pd.read_csv(paths["chemo_markers"])      # chemo markers

            sheet_states = f"{name}_states"[:31]
            sheet_sched  = f"{name}_schedule"[:31]
            sheet_marks  = f"{name}_markers"[:31]

            df_states.to_excel(writer, sheet_name=sheet_states, index=False)
            df_sched.to_excel(writer, sheet_name=sheet_sched, index=False)
            df_marks.to_excel(writer, sheet_name=sheet_marks, index=False)

            for sn in (sheet_states, sheet_sched, sheet_marks):
                try:
                    writer.sheets[sn].freeze_panes(1, 1)
                except Exception:
                    pass

        # Controls comparison
        if controls_comp is not None:
            controls_comp.to_excel(writer, sheet_name="Controls_Comparison", index=False)
            try:
                writer.sheets["Controls_Comparison"].freeze_panes(1, 1)
            except Exception:
                pass

    print(f"[OK] Excel report written: {outfile}")

if __name__ == "__main__":
    # (a) extremely cold —— 化疗主导，ICI 很少或不用
    apply_params(
        mu0=1.0, j_L=0.10, q=0.05, h_L=0.010, a_s=0.010, a_r=0.016, K_L=0.35,
        w2=3.0, w3=0.8, w4=2.8,                       # ICI 更贵
        D_I_cap_units=1.0, D_M_cap_units=4.0,         # ICI 仅 1 个单位；Chemo 4 个单位
        tau_days=3.0, K_I_indices=list(range(10))
    )
    out_a = run_one("case_a_extremely_cold", np.array([1.0, 0.10, 0.008, 0.0, 0.0, 0.0]))

    # (b) hot —— ICI 主导，化疗很少
    apply_params(
        mu0=0.6, j_L=0.25, q=0.015, h_L=0.002, a_s=0.010, a_r=0.012, K_L=0.22,
        w2=3.0, w3=0.8, w4=0.6,                        # ICI 更便宜
        D_I_cap_units=24.0, D_M_cap_units=3.0,          # ICI 上限大、Chemo 小
        tau_days=3.0, K_I_indices=list(range(10))
    )
    out_b = run_one("case_b_hot", np.array([1.0, 0.06, 0.030, 0.0, 0.0, 0.0]))

    # (c) cold —— 组合治疗（中庸）
    apply_params(
        mu0=0.9, j_L=0.18, q=0.03, h_L=0.005, a_s=0.010, a_r=0.014, K_L=0.28,
        w2=3.0, w3=0.8, w4=1.6,
        D_I_cap_units=4.0, D_M_cap_units=2.5,
        tau_days=3.0, K_I_indices=list(range(10))
    )
    out_c = run_one("case_c_cold", np.array([1.0, 0.10, 0.010, 0.0, 0.0, 0.0]))

    # (d) cold + high resistant growth —— 化疗略偏重
    apply_params(
        mu0=0.9, j_L=0.18, q=0.03, h_L=0.005, a_s=0.010, a_r=0.020, K_L=0.28,
        w2=3.0, w3=0.8, w4=2.0,
        D_I_cap_units=3.0, D_M_cap_units=4.0,
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
    write_excel_report(all_outs, outfile="exports/four_cases_report.xlsx")