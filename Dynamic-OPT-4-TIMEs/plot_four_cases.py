import numpy as np
import pandas as pd
import json, os
from export_results import export_results_package
from params import apply_params
import matplotlib.pyplot as plt


# --- Helper functions for plotting from Excel report ---
def _sheet_exists(xls: pd.ExcelFile, sheet: str) -> bool:
    try:
        return sheet in xls.sheet_names
    except Exception:
        return False

def load_case_frames(report_path: str, case: str):
    """Return (states_df, sched_df, markers_df) for a given case name from the Excel report.
    Accepts `<case>_schedule` or `<case>_controls` as the schedule sheet name; `<case>_markers` or `<case>_marker` for markers (optional).
    """
    xls = pd.ExcelFile(report_path)

    # 1) states sheet (must exist)
    s_states = f"{case}_states"[:31]
    if s_states not in xls.sheet_names:
        raise FileNotFoundError(f"Sheet '{s_states}' not found in {report_path}. Available: {xls.sheet_names}")

    # 2) schedule sheet can be either `<case>_schedule` or `<case>_controls`
    sched_candidates = [f"{case}_schedule"[:31], f"{case}_controls"[:31]]
    s_sched = next((cand for cand in sched_candidates if cand in xls.sheet_names), None)
    if s_sched is None:
        raise FileNotFoundError(
            f"No schedule sheet for case '{case}'. Tried {sched_candidates}. Available: {xls.sheet_names}"
        )

    # 3) markers sheet is optional; accept `<case>_markers` or `<case>_marker`
    mark_candidates = [f"{case}_markers"[:31], f"{case}_marker"[:31]]
    s_marks = next((cand for cand in mark_candidates if cand in xls.sheet_names), None)

    # Read sheets
    states_df = pd.read_excel(xls, sheet_name=s_states)
    sched_df  = pd.read_excel(xls, sheet_name=s_sched)
    markers_df = pd.read_excel(xls, sheet_name=s_marks) if s_marks else pd.DataFrame()

    # Robust columns for states
    if "day" not in states_df.columns:
        for alt in ("t", "days", "time_day"):
            if alt in states_df.columns:
                states_df = states_df.rename(columns={alt: "day"})
                break
    if "X_total" not in states_df.columns and {"Xs", "Xr"}.issubset(states_df.columns):
        states_df["X_total"] = states_df["Xs"] + states_df["Xr"]

    # Robust columns for schedule
    ren = {"Interval": "interval", "v_I": "vI", "v_M": "vM", "doseI_w": "doseI_week", "doseM_w": "doseM_week"}
    sched_df = sched_df.rename(columns=ren)
    for c in ("interval", "vI", "vM"):
        if c not in sched_df.columns:
            raise KeyError(f"'{s_sched}' is missing required column: {c}")
    if "doseI_week" not in sched_df.columns:
        sched_df["doseI_week"] = 0.0
    if "doseM_week" not in sched_df.columns:
        sched_df["doseM_week"] = 0.0
    sched_df["interval"] = sched_df["interval"].astype(int)

    return states_df, sched_df, markers_df

def plot_case_overview(states_df: pd.DataFrame, sched_df: pd.DataFrame, case: str, outdir: str) -> str:
    """Make a 3-panel figure for one case: tumor load, lymphocytes, and controls."""
    os.makedirs(outdir, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)

    # Panel 1: Tumor burden
    ax = axs[0]
    x = states_df.get("day", pd.Series(range(len(states_df))))
    if "X_total" in states_df:
        ax.plot(x, states_df["X_total"], label="X_total")
    if {"Xs", "Xr"}.issubset(states_df.columns):
        ax.plot(x, states_df["Xs"], linestyle=":", label="Xs")
        ax.plot(x, states_df["Xr"], linestyle="--", label="Xr")
    ax.set_ylabel("Tumor load")
    ax.set_title(f"{case}: Tumor dynamics")
    ax.legend(loc="best")

    # Panel 2: Lymphocytes
    ax = axs[1]
    if "L" in states_df:
        ax.plot(x, states_df["L"], label="L")
    if "I" in states_df:
        ax.plot(x, states_df["I"], linestyle=":", label="I")
    if "M" in states_df:
        ax.plot(x, states_df["M"], linestyle="--", label="M")
    ax.set_ylabel("Immune-related")
    ax.set_title(f"{case}: L/I/M")
    ax.legend(loc="best")

    # Panel 3: Controls per interval
    ax = axs[2]
    k = sched_df["interval"].astype(int)
    ax.step(k, sched_df["vI"], where="post", label="vI")
    ax.step(k, sched_df["vM"], where="post", label="vM")
    ax.set_xlabel("Interval")
    ax.set_ylabel("Control")
    ax.set_title(f"{case}: Controls")
    ax.legend(loc="best")

    out_png = os.path.join(outdir, f"{case}_overview.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png

def plot_four_cases_grids(report_path: str, cases: list, outdir: str) -> tuple[str, str]:
    """Create 2x2 grids across cases for X_total and L."""
    os.makedirs(outdir, exist_ok=True)

    # X_total grid
    fig1, ax1 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    # L grid
    fig2, ax2 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for idx, case in enumerate(cases[:4]):
        r, c = divmod(idx, 2)
        states_df, sched_df, _ = load_case_frames(report_path, case)
        x = states_df.get("day", pd.Series(range(len(states_df))))

        # X_total panel
        ax = ax1[r, c]
        if "X_total" in states_df:
            ax.plot(x, states_df["X_total"])
        elif {"Xs", "Xr"}.issubset(states_df.columns):
            ax.plot(x, states_df["Xs"] + states_df["Xr"])  # fallback
        else:
            ax.text(0.5, 0.5, "No tumor columns", transform=ax.transAxes, ha="center")
        ax.set_title(case)
        ax.set_xlabel("Day")
        ax.set_ylabel("X_total")

        # L panel
        ax = ax2[r, c]
        if "L" in states_df:
            ax.plot(x, states_df["L"])
        else:
            ax.text(0.5, 0.5, "No L column", transform=ax.transAxes, ha="center")
        ax.set_title(case)
        ax.set_xlabel("Day")
        ax.set_ylabel("L")

    out1 = os.path.join(outdir, "grid_X_total.png")
    out2 = os.path.join(outdir, "grid_L.png")
    fig1.savefig(out1, dpi=200)
    fig2.savefig(out2, dpi=200)
    plt.close(fig1); plt.close(fig2)
    return out1, out2

def plot_from_report(report_path: str, outdir: str = "exports/plots"):
    """Read four_cases_report.xlsx and generate per-case and grid plots."""
    # Try to get cases from Summary/Overview; otherwise infer from sheet names
    cases = []
    for sum_sheet in ("Summary", "Overview"):
        try:
            df_sum = pd.read_excel(report_path, sheet_name=sum_sheet)
            if "case" in df_sum.columns:
                cases = [str(x) for x in df_sum["case"].dropna().tolist()]
                break
        except Exception:
            pass
    if not cases:
        try:
            xls = pd.ExcelFile(report_path)
            cases = sorted({s[:-7] for s in xls.sheet_names if s.endswith("_states")})
        except FileNotFoundError:
            print(f"[ERROR] Report not found: {report_path}")
            return []
        except Exception as e:
            print(f"[ERROR] Failed to read report '{report_path}': {e}")
            return []
    if not cases:
        print(
            f"[WARN] No cases detected in '{report_path}'. "
            f"Please run 'run_four_cases.py' first to generate the Excel with a 'Summary' sheet "
            f"(containing a 'case' column) or per-case '*_states' sheets."
        )
        return []
    print(f"[INFO] Detected cases: {cases}")

    os.makedirs(outdir, exist_ok=True)
    saved = []
    for case in cases:
        try:
            states_df, sched_df, _ = load_case_frames(report_path, case)
            saved.append(plot_case_overview(states_df, sched_df, case, outdir))
        except Exception as e:
            print(f"[WARN] skip {case}: {e}")

    # 2x2 grids (first four cases)
    if cases:
        plot_four_cases_grids(report_path, cases, outdir)

    print(f"[OK] Plots saved to: {outdir} (count={len(saved)})")
    return saved

def run_one(prefix, x0):
    # ... (rest of run_one code) ...
    results = ...  # suppose results is obtained here
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
    with pd.ExcelWriter(outfile, engine="xlsxwriter") as writer:
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
    import sys, os
    report = sys.argv[1] if len(sys.argv) > 1 else "exports/four_cases_report.xlsx"
    saved = plot_from_report(report_path=report, outdir="exports/plots")
    try:
        files = os.listdir("exports/plots") if os.path.isdir("exports/plots") else []
        print(f"[CHECK] files in exports/plots: {files}")
    except Exception as e:
        print(f"[WARN] Could not list exports/plots: {e}")