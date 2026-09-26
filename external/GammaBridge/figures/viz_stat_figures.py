"""Statistical figures for the paper (richer version).

- fig_smartstart_curve.pdf  — 2 panels: PSNR/SSIM vs L_in (smart vs naive) + ratio-var vs L_in.
- fig_ablation_nfe.pdf       — 2 panels: PSNR vs NFE + SSIM vs NFE, 7 NFE points, 5 variants.
- fig_target_L_curve.pdf     — 3 panels: ratio mean, ratio variance, KS p-value vs L_out.

All data are read from eval/results/ CSVs (see scripts/eval_naive_smart.py,
scripts/eval_finer_nfe.sh, scripts/eval_target_L.py).
"""

from __future__ import annotations
import csv
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# --- global style ------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "lines.markersize": 3.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

_ROOT   = Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "paper"
RES     = _ROOT / "eval" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IEEE column widths
COL_W = 3.5   # single column
DBL_W = 7.16  # double column (figure*)


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _mean_std(rows, key):
    xs = [float(r[key]) for r in rows]
    return statistics.mean(xs), statistics.pstdev(xs)


def _save(fig, stem):
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=200)
        print(f"[saved] {out}")
    plt.close(fig)


# =============================================================================
# Figure A: smart-start vs naive-start, plus ratio-variance physics check
# =============================================================================
def fig_smartstart_curve():
    rows = _read_csv(RES / "naive_smart_start.csv")
    L_ins = sorted({float(r["L_in"]) for r in rows})

    smart_psnr = []; smart_ssim = []; smart_rvar = []
    naive_psnr = []
    input_psnr = []
    for L in L_ins:
        s = [r for r in rows if float(r["L_in"]) == L and r["mode"] == "smart"]
        n = [r for r in rows if float(r["L_in"]) == L and r["mode"] == "naive"]
        smart_psnr.append(_mean_std(s, "psnr_denoised"))
        smart_ssim.append(_mean_std(s, "ssim_denoised"))
        smart_rvar.append(_mean_std(s, "ratio_var"))
        naive_psnr.append(_mean_std(n, "psnr_denoised"))
        input_psnr.append(_mean_std(s, "psnr_noisy"))

    L = np.array(L_ins)
    def col(p, i): return np.array([r[i] for r in p])

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(DBL_W, 2.3))

    # ---- (a) PSNR/SSIM vs L_in ------------------------------------------
    c_smart = "#1f4e79"
    c_naive = "#8b1a1a"
    c_input = "#7d7d7d"
    c_ssim  = "#c66a11"

    ax1.errorbar(L, col(smart_psnr, 0), yerr=col(smart_psnr, 1),
                 fmt="o-", color=c_smart, capsize=1.5, elinewidth=0.6,
                 label="smart-start (PSNR)", zorder=4)
    ax1.errorbar(L, col(naive_psnr, 0), yerr=col(naive_psnr, 1),
                 fmt="s--", color=c_naive, capsize=1.5, elinewidth=0.6,
                 label="naive-start (PSNR)", zorder=3)
    ax1.plot(L, col(input_psnr, 0), ":", color=c_input, linewidth=0.9,
             label="input PSNR", zorder=2)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(L); ax1.set_xticklabels([f"{int(x)}" for x in L])
    ax1.set_xlabel(r"input look number $L_{\mathrm{in}}$")
    ax1.set_ylabel("PSNR (dB)", color=c_smart)
    ax1.tick_params(axis="y", colors=c_smart)
    ax1.set_ylim(8, 30)
    ax1.grid(True, which="major", alpha=0.25, linewidth=0.4)

    ax2 = ax1.twinx()
    ax2.plot(L, col(smart_ssim, 0), "^:", color=c_ssim,
             markersize=3.2, linewidth=0.9, label="smart-start (SSIM)")
    ax2.set_ylabel("SSIM", color=c_ssim)
    ax2.tick_params(axis="y", colors=c_ssim)
    ax2.set_ylim(0.05, 0.90)
    ax2.spines["right"].set_visible(True)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    # legend below the axes to avoid overlapping with the naive-start plateau
    ax1.legend(lines1 + lines2, labs1 + labs2,
               loc="lower center", bbox_to_anchor=(0.5, -0.55),
               handlelength=1.5, ncol=4, columnspacing=1.4,
               labelspacing=0.35, borderaxespad=0.1)
    ax1.set_title(r"(a) reconstruction quality vs. $L_{\mathrm{in}}$")

    # ---- (b) ratio variance vs L_in (physics check) --------------------
    theo = 1.0 / L
    ax3.plot(L, theo, "-", color=c_input, linewidth=1.1,
             label=r"theory $1/L_{\mathrm{in}}$")
    ax3.errorbar(L, col(smart_rvar, 0), yerr=col(smart_rvar, 1),
                 fmt="o-", color=c_smart, capsize=1.5, elinewidth=0.6,
                 label="empirical (smart-start)")
    ax3.set_xscale("log", base=2)
    ax3.set_yscale("log")
    ax3.set_xticks(L); ax3.set_xticklabels([f"{int(x)}" for x in L])
    ax3.set_xlabel(r"input look number $L_{\mathrm{in}}$")
    ax3.set_ylabel(r"ratio variance $\mathrm{Var}(\xobs / \hat x_0)$"
                   .replace("\\xobs", "x_{\\mathrm{obs}}"))
    ax3.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax3.legend(loc="lower left", handlelength=1.6)
    ax3.set_title(r"(b) statistical fidelity to $\Gamma(L_{\mathrm{in}}, L_{\mathrm{in}})$")

    fig.subplots_adjust(wspace=0.28, bottom=0.28)
    _save(fig, "fig_smartstart_curve")


# =============================================================================
# Figure B: ablation, PSNR/SSIM vs NFE with fine sweep
# =============================================================================
_ABL_CKPTS = [
    ("Full model",                    "combo_v1_L1/ckpt0015000", "#1f4e79", "o", "-"),
    (r"$-$ closed-form posterior",    "abl_naive_post",          "#c66a11", "s", "-"),
    (r"$-$ cond$_{x_1}$ $-$ cons.",   "abl_no_cond",             "#8b1a1a", "^", "-"),
    (r"$-$ log schedule",             "abl_linear_sched",        "#357a3a", "D", "-"),
    (r"$-$ log-residual",             "abl_direct_pred",         "#666666", "v", "-"),
]
_NFE_TO_NS = {1: 2, 2: 3, 3: 4, 5: 6, 10: 11, 15: 16, 25: 26}


def _load_abl_metric(stem, metric):
    xs, ys = [], []
    for nfe, ns in _NFE_TO_NS.items():
        p = RES / f"{stem}_ns{ns}.csv"
        if not p.exists(): continue
        rows = _read_csv(p)
        m, _ = _mean_std(rows, metric)
        xs.append(nfe); ys.append(m)
    return np.array(xs), np.array(ys)


def fig_ablation_nfe():
    fig, (axP, axS) = plt.subplots(1, 2, figsize=(DBL_W, 2.4))
    for label, stem, color, marker, ls in _ABL_CKPTS:
        xs, ps = _load_abl_metric(stem, "psnr_denoised")
        _, ss  = _load_abl_metric(stem, "ssim_denoised")
        style = dict(markersize=3.8, linewidth=1.1)
        axP.plot(xs, ps, marker=marker, ls=ls, color=color, label=label, **style)
        axS.plot(xs, ss, marker=marker, ls=ls, color=color, **style)

    for ax, ylab, title in [
        (axP, "PSNR (dB)",   "(a) PSNR"),
        (axS, "SSIM",        "(b) SSIM"),
    ]:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 15, 25])
        ax.set_xticklabels(["1", "2", "3", "5", "10", "15", "25"])
        ax.set_xlabel("NFE (number of reverse steps)")
        ax.set_ylabel(ylab)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.4)
        ax.set_title(title)

    axP.set_ylim(4, 26)
    axS.set_ylim(0, 0.68)

    axP.axvline(5, color="#dddddd", linewidth=0.6, zorder=0)
    axS.axvline(5, color="#dddddd", linewidth=0.6, zorder=0)

    # legend below the two panels, outside the axes
    handles, labels = axP.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05), handlelength=1.6,
               columnspacing=1.4)

    fig.subplots_adjust(wspace=0.28, bottom=0.24)
    _save(fig, "fig_ablation_nfe")


# =============================================================================
# Figure C: target-L: mean, variance, KS-p vs L_out
# =============================================================================
def fig_target_L_curve():
    rows = _read_csv(RES / "target_L_sweep.csv")
    L_outs = sorted({float(r["L_out"]) for r in rows})

    r_mean, r_var, ksp = [], [], []
    r_mean_std, r_var_std = [], []
    for L in L_outs:
        subset = [r for r in rows if float(r["L_out"]) == L]
        m, s = _mean_std(subset, "ratio_mean"); r_mean.append(m); r_mean_std.append(s)
        m, s = _mean_std(subset, "ratio_var");  r_var.append(m);  r_var_std.append(s)
        m, _ = _mean_std(subset, "ratio_ks_p"); ksp.append(max(m, 1e-40))
    L_outs = np.array(L_outs)
    r_mean = np.array(r_mean); r_var = np.array(r_var); ksp = np.array(ksp)
    r_mean_std = np.array(r_mean_std); r_var_std = np.array(r_var_std)

    # theoretical curves for variance (both couplings)
    beta_theo = (L_outs - 1) / (L_outs + 1)       # stochastic Beta-coupled ratio
    fwd_theo  = 1.0 / L_outs                       # decoupled forward marginal

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(DBL_W, 2.2))

    c_emp = "#1f4e79"
    c_th1 = "#7d7d7d"
    c_th2 = "#c66a11"

    # (a) ratio mean
    ax1.axhline(1.0, color=c_th1, linewidth=0.7, linestyle="--", label=r"target $\bar r{=}1$")
    ax1.errorbar(L_outs, r_mean, yerr=r_mean_std, fmt="o-", color=c_emp,
                 capsize=1.5, elinewidth=0.6, label="empirical", zorder=3)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"target look $L_{\mathrm{out}}$")
    ax1.set_ylabel(r"ratio mean $\bar r$")
    ax1.set_ylim(0.72, 1.05)
    ax1.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax1.legend(loc="lower right", handlelength=1.6)
    ax1.set_title(r"(a) mean of $r = x_{\mathrm{obs}} / x_{t^\star}$")

    # (b) ratio variance
    ax2.plot(L_outs, beta_theo, "-", color=c_th1, linewidth=1.1,
             label=r"stochastic (Beta)")
    ax2.plot(L_outs, fwd_theo,  ":", color=c_th2, linewidth=1.0,
             label=r"decoupled $1/L_{\mathrm{out}}$")
    ax2.errorbar(L_outs, r_var, yerr=r_var_std, fmt="o-", color=c_emp,
                 capsize=1.5, elinewidth=0.6, label="empirical (OT-ODE)", zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"target look $L_{\mathrm{out}}$")
    ax2.set_ylabel(r"ratio variance $\mathrm{Var}(r)$")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax2.legend(loc="lower right", handlelength=1.6)
    ax2.set_title(r"(b) variance of $r$")

    # (c) KS p-value
    ax3.axhline(0.05, color=c_th1, linewidth=0.7, linestyle="--", label=r"$p{=}0.05$")
    ax3.plot(L_outs, ksp, "o-", color=c_emp, label="empirical KS $p$", zorder=3)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel(r"target look $L_{\mathrm{out}}$")
    ax3.set_ylabel(r"KS $p$-value vs. $\Gamma$")
    ax3.set_ylim(1e-40, 1.5)
    ax3.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax3.legend(loc="lower right", handlelength=1.6)
    ax3.set_title(r"(c) marginal fit to $\Gamma(1,1)$")

    fig.subplots_adjust(wspace=0.32)
    _save(fig, "fig_target_L_curve")


# =============================================================================
# Combo 2: unified "runtime look-number control" figure — merges smart-start
# (input axis, top row) with target-L (output axis, bottom row).
# =============================================================================
def fig_lookcontrol_combo():
    # smart-start data
    rows_ss = _read_csv(RES / "naive_smart_start.csv")
    L_ins = sorted({float(r["L_in"]) for r in rows_ss})
    smart_p, smart_s, smart_v = [], [], []
    naive_p = []
    for L in L_ins:
        s = [r for r in rows_ss if float(r["L_in"]) == L and r["mode"] == "smart"]
        n = [r for r in rows_ss if float(r["L_in"]) == L and r["mode"] == "naive"]
        smart_p.append(_mean_std(s, "psnr_denoised"))
        smart_s.append(_mean_std(s, "ssim_denoised"))
        smart_v.append(_mean_std(s, "ratio_var"))
        naive_p.append(_mean_std(n, "psnr_denoised"))
    L_in = np.array(L_ins)
    def col(p, i): return np.array([r[i] for r in p])

    # target-L data
    rows_tL = _read_csv(RES / "target_L_sweep.csv")
    L_outs = sorted({float(r["L_out"]) for r in rows_tL})
    r_mean, r_var, ksp, r_mean_s, r_var_s = [], [], [], [], []
    for L in L_outs:
        subset = [r for r in rows_tL if float(r["L_out"]) == L]
        m, s = _mean_std(subset, "ratio_mean"); r_mean.append(m); r_mean_s.append(s)
        m, s = _mean_std(subset, "ratio_var");  r_var.append(m);  r_var_s.append(s)
        m, _ = _mean_std(subset, "ratio_ks_p"); ksp.append(max(m, 1e-40))
    L_out = np.array(L_outs)
    r_mean = np.array(r_mean); r_var = np.array(r_var); ksp = np.array(ksp)
    r_mean_s = np.array(r_mean_s); r_var_s = np.array(r_var_s)
    beta_theo = (L_out - 1) / (L_out + 1)
    fwd_theo  = 1.0 / L_out

    c_emp   = "#1f4e79"
    c_naive = "#8b1a1a"
    c_ssim  = "#c66a11"
    c_th1   = "#7d7d7d"
    c_th2   = "#c66a11"

    fig, axes = plt.subplots(2, 3, figsize=(DBL_W, 4.0))
    (ax_p, ax_s, ax_v) = axes[0]
    (ax_m, ax_var, ax_k) = axes[1]

    def _panel_letter(ax, letter):
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold",
                ha="left", va="top")

    # ---- Row 1: input axis (L_in) --------------------------------------
    # (a) PSNR smart vs naive
    ax_p.errorbar(L_in, col(smart_p, 0), yerr=col(smart_p, 1),
                  fmt="o-", color=c_emp, capsize=1.5, elinewidth=0.6,
                  label="smart-start")
    ax_p.errorbar(L_in, col(naive_p, 0), yerr=col(naive_p, 1),
                  fmt="s--", color=c_naive, capsize=1.5, elinewidth=0.6,
                  label="naive-start")
    ax_p.set_xscale("log", base=2)
    ax_p.set_xticks(L_in); ax_p.set_xticklabels([str(int(x)) for x in L_in])
    ax_p.set_xlabel(r"input look $L_{\mathrm{in}}$")
    ax_p.set_ylabel("PSNR (dB)")
    ax_p.set_ylim(8, 30)
    ax_p.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax_p.legend(loc="center right", handlelength=1.5,
                bbox_to_anchor=(0.98, 0.55))
    _panel_letter(ax_p, "(a)")

    # (b) SSIM
    ax_s.errorbar(L_in, col(smart_s, 0), yerr=col(smart_s, 1),
                  fmt="^-", color=c_ssim, capsize=1.5, elinewidth=0.6,
                  label="smart-start")
    ax_s.set_xscale("log", base=2)
    ax_s.set_xticks(L_in); ax_s.set_xticklabels([str(int(x)) for x in L_in])
    ax_s.set_xlabel(r"input look $L_{\mathrm{in}}$")
    ax_s.set_ylabel("SSIM")
    ax_s.set_ylim(0.3, 0.9)
    ax_s.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax_s.legend(loc="lower right", handlelength=1.5)
    _panel_letter(ax_s, "(b)")

    # (c) ratio variance vs L_in — physics check
    theo = 1.0 / L_in
    ax_v.plot(L_in, theo, "-", color=c_th1, linewidth=1.1,
              label=r"theory $1/L_{\mathrm{in}}$")
    ax_v.errorbar(L_in, col(smart_v, 0), yerr=col(smart_v, 1),
                  fmt="o-", color=c_emp, capsize=1.5, elinewidth=0.6,
                  label="empirical")
    ax_v.set_xscale("log", base=2); ax_v.set_yscale("log")
    ax_v.set_xticks(L_in); ax_v.set_xticklabels([str(int(x)) for x in L_in])
    ax_v.set_xlabel(r"input look $L_{\mathrm{in}}$")
    ax_v.set_ylabel(r"ratio variance $\mathrm{Var}(r)$")
    ax_v.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax_v.legend(loc="lower left", handlelength=1.5)
    _panel_letter(ax_v, "(c)")

    # ---- Row 2: output axis (L_out) ------------------------------------
    # (d) ratio mean
    ax_m.axhline(1.0, color=c_th1, linewidth=0.7, linestyle="--",
                 label=r"target $\bar r{=}1$")
    ax_m.errorbar(L_out, r_mean, yerr=r_mean_s, fmt="o-", color=c_emp,
                  capsize=1.5, elinewidth=0.6, label="empirical")
    ax_m.set_xscale("log")
    ax_m.set_xlabel(r"target look $L_{\mathrm{out}}$")
    ax_m.set_ylabel(r"ratio mean $\bar r$")
    ax_m.set_ylim(0.72, 1.05)
    ax_m.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax_m.legend(loc="lower right", handlelength=1.5)
    _panel_letter(ax_m, "(d)")

    # (e) ratio variance with two theoretical curves
    ax_var.plot(L_out, beta_theo, "-", color=c_th1, linewidth=1.1,
                label="stochastic (Beta)")
    ax_var.plot(L_out, fwd_theo, ":", color=c_th2, linewidth=1.0,
                label=r"decoupled $1/L_{\mathrm{out}}$")
    ax_var.errorbar(L_out, r_var, yerr=r_var_s, fmt="o-", color=c_emp,
                    capsize=1.5, elinewidth=0.6, label="empirical")
    ax_var.set_xscale("log")
    ax_var.set_xlabel(r"target look $L_{\mathrm{out}}$")
    ax_var.set_ylabel(r"ratio variance $\mathrm{Var}(r)$")
    ax_var.set_ylim(0, 1.15)
    ax_var.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax_var.legend(loc="lower right", handlelength=1.5)
    _panel_letter(ax_var, "(e)")

    # (f) KS p-value
    ax_k.axhline(0.05, color=c_th1, linewidth=0.7, linestyle="--", label=r"$p{=}0.05$")
    ax_k.plot(L_out, ksp, "o-", color=c_emp, label=r"empirical")
    ax_k.set_xscale("log"); ax_k.set_yscale("log")
    ax_k.set_xlabel(r"target look $L_{\mathrm{out}}$")
    ax_k.set_ylabel(r"KS $p$-value")
    ax_k.set_ylim(1e-40, 1.5)
    ax_k.grid(True, which="major", alpha=0.25, linewidth=0.4)
    ax_k.legend(loc="lower right", handlelength=1.5)
    _panel_letter(ax_k, "(f)")

    fig.subplots_adjust(wspace=0.32, hspace=0.42, left=0.06, right=0.985,
                        top=0.97, bottom=0.09)
    _save(fig, "fig_lookcontrol_combo")


if __name__ == "__main__":
    fig_smartstart_curve()
    fig_ablation_nfe()
    fig_target_L_curve()
    fig_lookcontrol_combo()
