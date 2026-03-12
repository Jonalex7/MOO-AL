from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter1d

from settings import (
    AGGREGATED_DIR,
    CASE_STUDIES,
    GROUP_2D,
    GROUP_HD,
    CASE_TITLES,
    STRATEGY_COLORS,
    strategy_label,
)


# ---------------------------------------------------------------------------
# Load aggregated artifacts and expose legacy variables expected by the
# original plotting blocks below.
# ---------------------------------------------------------------------------
rel_path = AGGREGATED_DIR / "relative_error_dict.pkl"
cfg_path = AGGREGATED_DIR / "config_results_dict.pkl"

if not rel_path.is_file():
    raise FileNotFoundError(f"Missing artifact: {rel_path}")
if not cfg_path.is_file():
    raise FileNotFoundError(f"Missing artifact: {cfg_path}")

with open(rel_path, "rb") as f_id:
    relative_error_dict_raw = pickle.load(f_id)
with open(cfg_path, "rb") as f_id:
    config_results_dict = pickle.load(f_id)

# Convert to legacy shape used by original figures:
# relative_error_dict[case][strategy][run] -> list of errors
relative_error_dict = {}
for case, strategy_dict in relative_error_dict_raw.items():
    case_dict = {}
    for strategy, run_dict in strategy_dict.items():
        run_legacy = {}
        for run_name, payload in run_dict.items():
            if isinstance(payload, dict):
                if "abs_relative_error" in payload:
                    run_legacy[run_name] = payload["abs_relative_error"]
                elif "relative_error" in payload:
                    run_legacy[run_name] = np.abs(
                        np.asarray(payload["relative_error"], dtype=float)
                    ).tolist()
                else:
                    run_legacy[run_name] = []
            else:
                run_legacy[run_name] = payload
        case_dict[strategy] = run_legacy
    relative_error_dict[case] = case_dict

# Legacy names kept intentionally to preserve your original figure code.
group_2D = GROUP_2D
group_HD = GROUP_HD
casestudy = [case for case in CASE_STUDIES if case in relative_error_dict]
custom_titles = [CASE_TITLES.get(case, case) for case in casestudy]
strategy_colors = dict(STRATEGY_COLORS)
font_size = 8
linewidth = 0.6
cm = 1 / 2.54
sigma = 1.5
doe = 10

# Order used across your existing figures.
_strategy_display_order = [
    "moo_reliability",
    "moo_knee",
    "moo_compromise",
    "moo_eps_ew",
    "erf",
    "reif",
    "reif2",
    "eff",
    "u",
    "portfolio",
    "eier",
]
custom_legend = [strategy_label(s) for s in _strategy_display_order]

SUMMARY_TXT_PATH = AGGREGATED_DIR / "Figures" / "postprocess_figures_summary.txt"
REPORT_LINES = []


def report(message=""):
    text = str(message)
    print(text)
    REPORT_LINES.append(text)


def flush_report_summary(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_id:
        f_id.write("\n".join(REPORT_LINES).rstrip() + "\n")
    print(f"[save][SUMMARY] {path}")


report("[start] figures postprocess")
report(f"  relative_error_dict : {rel_path}")
report(f"  config_results_dict : {cfg_path}")
report(f"  cases available     : {len(casestudy)}")

# ---------------------------------------------------------------------------
# Figure controls (inputs)
# ---------------------------------------------------------------------------
FIGURES_DIR = AGGREGATED_DIR / "Figures"
SHOW_FIGURES = False
FIGURE_EXPORTS = {
    "F01": "pf_evolution_single_row.pdf",
    "F02": "pf_evolution_2rows_softmedians.pdf",
    "F03": "aggregated_N_failed_exp.pdf",
    "F04": "distrib_epsilons_mean_1threshold.pdf",
    "F05": "pf_evolution_highdim.pdf",
    "F06": "sample_effic_highdim.pdf",
}
FIGURE_ENABLED = {
    "F01": False,   # disabled by request
    "F02": True,
    "F03": True,
    "F04": True,
    "F05": True,
    "F06": True,
}
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def finalize_figure(fig, figure_id):
    if not FIGURE_ENABLED.get(figure_id, True):
        report(f"[skip][{figure_id}] Figure disabled by toggle")
        plt.close(fig)
        return
    output_path = FIGURES_DIR / FIGURE_EXPORTS[figure_id]
    fig.savefig(output_path)
    report(f"[save][{figure_id}] {output_path}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def parse_experiment_number(run_key):
    """
    Parse experiment id from run folder names like:
      - '1_10_2026_...'
      - 'moo_eps_ew_1_10_2025_...'
    Returns int when parse succeeds, otherwise returns the original key.
    """
    text = str(run_key)
    tokens = text.split("_")

    # Find first adjacent numeric pair: batch_id, exp_id
    for idx in range(len(tokens) - 1):
        if tokens[idx].isdigit() and tokens[idx + 1].isdigit():
            return int(tokens[idx + 1])

    # Fallback: first numeric token
    for token in tokens:
        if token.isdigit():
            return int(token)

    return text


def _fmt_sci_compact(x):
    mantissa, exponent = f"{float(x):.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _case_console_name(case):
    title = CASE_TITLES.get(case, case)
    return title.replace("$", "")

# Let us adapt the code 

# --- Parameters ------------------------------------------------------------
captured_ls = 5         # number of best strategies to capture (top-k)
threshold_factor = 1.0   # scale threshold up (>1) or down (<1). Set threshold_factor = 1.2 if you want the threshold to be 20% looser than that 3rd-minimum.
required_consecutive = 3  # consecutive iterations below threshold for ranking/efficiency
eier_reference_strategy = "eier"
eier_reference_marker = "."
excluded_from_threshold = {eier_reference_strategy}
excluded_from_f03 = {eier_reference_strategy}
debug_case_for_print = None  # e.g., "four_branch_7"
# We calculate threshold for the specified ls

report("[config] threshold/ranking conditions")
report(f"  captured_ls          : {captured_ls}")
report(f"  threshold_factor     : {threshold_factor}")
report(f"  required_consecutive : {required_consecutive}")
report(f"  excluded threshold   : [{eier_reference_strategy}]")
report("")

# --- Step 2: per-strategy global minima, then top-k per case --------------

threshold_dict = {}  # [case] -> { "topk_minima": [...], "threshold_delta_pf": ..., "threshold_entry": ... }

for case in casestudy:
    if case not in relative_error_dict:
        continue

    # decide max length for this case
    if case in group_2D:
        case_max_len = 200
    else:
        case_max_len = 500

    per_strategy_minima = []  # one entry per strategy: its global minimum on median evolution

    # --- compute global minimum of the median evolution per strategy -------
    for strategy, exp_dict in relative_error_dict[case].items():
        if strategy in excluded_from_threshold:
            continue
        if not exp_dict:
            continue

        max_len_available = max(len(arr) for arr in exp_dict.values())
        cap_len = min(max_len_available, case_max_len)

        n_runs = len(exp_dict)
        rel_diff_mat = np.full((n_runs, cap_len), np.nan, dtype=float)

        # rows = experiments
        for row_idx, (exp_num, rel_diff) in enumerate(sorted(exp_dict.items())):
            this_len = min(len(rel_diff), cap_len)
            rel_diff_mat[row_idx, :this_len] = rel_diff[:this_len]

        # median evolution across runs (ignoring NaNs)
        median_raw = np.median(rel_diff_mat, axis=0)   #CHECK MEAN NOT MEDIAN
        # median_evolution = gaussian_filter1d(median_raw, sigma=sigma)
        median_evolution = median_raw

        finite_mask = np.isfinite(median_evolution)
        if not np.any(finite_mask):
            continue

        finite_indices = np.where(finite_mask)[0]
        finite_values = median_evolution[finite_mask]

        local_argmin = np.argmin(finite_values)
        it_idx = int(finite_indices[local_argmin])

        delta_pf_min = float(median_evolution[it_idx])
        n_acquired_samples = int(doe + it_idx)

        per_strategy_minima.append({
            "strategy": strategy,
            "delta_pf_min": delta_pf_min,
            "iteration_idx": it_idx,
            "n_acquired_samples": n_acquired_samples,
        })

    if not per_strategy_minima:
        report(f"[warn] no valid minima for case '{case}'.")
        continue

    # sort strategies by their global minimum δPf
    per_strategy_minima_sorted = sorted(per_strategy_minima,
                                        key=lambda d: d["delta_pf_min"])

    # number of strategies we can actually capture
    top_k = min(captured_ls, len(per_strategy_minima_sorted))
    topk = []

    for rank in range(top_k):
        entry = per_strategy_minima_sorted[rank].copy()
        entry["rank"] = rank + 1  # 1 = best
        topk.append(entry)

    # threshold rank index (0-based): captured_ls-1, but capped by available top_k-1
    threshold_rank_index = min(captured_ls - 1, top_k - 1)
    threshold_entry = topk[threshold_rank_index]

    # apply factor to adjust strictness
    base_threshold = threshold_entry["delta_pf_min"]
    threshold_delta_pf = threshold_factor * base_threshold

    threshold_dict[case] = {
        "topk_minima": topk,                    # up to captured_ls strategies
        "captured_ls": captured_ls,
        "threshold_factor": threshold_factor,
        "threshold_delta_pf": threshold_delta_pf,
        "threshold_entry": threshold_entry      # the strategy that defines the base threshold
    }

target_epsilon = {}

for case, data in threshold_dict.items():
    # This is your primary (hard) threshold
    base_eps = data["threshold_delta_pf"]
    
    # Define the three levels: [100%, 50%, 10%]
    # We sort them descending so the lines appear in order on the plot
    target_epsilon[case] = sorted([
        base_eps, 
        # base_eps * 3.0, 
        # base_eps * 5.0
    ], reverse=True)

# Optional: Print to verify the generated levels
for case, levels in target_epsilon.items():
    formatted_levels = [_fmt_sci_compact(l) for l in levels]
    report(f"threshold[{case}] = {formatted_levels[0]} "
           f"(defined by {strategy_label(threshold_dict[case]['threshold_entry']['strategy'])})")
report("")


# labels in the desired order
strategies_order = [
    'moo_reliability','moo_knee', 'moo_compromise',
                    'moo_eps_ew',
                    'erf', 'reif', 'reif2', 'eff', 'u', 'portfolio', 'eier'
                    ]

plt.rcParams.update({
    'font.size': font_size,
    'legend.fontsize': font_size,
    'legend.title_fontsize': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
})

remaining_cases = casestudy[:-1]

# ---------------------------------------------------------------------------
# Figure F01: Pf evolution (single row, non-high-dimensional cases)
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(
    1, len(remaining_cases),
    figsize=(19*cm, 4.0*cm),
    sharey=True, sharex=False
)
if len(remaining_cases) == 1:
    axs = [axs]

epsilon_color = '#4d4d4d'

for i, case in enumerate(remaining_cases):
    
    ax = axs[i]
    ax.set_title(f"{custom_titles[i]}", fontsize=font_size)

    if i == 0:
        ax.set_ylabel(r"$\delta P_\mathrm{f}$")

    # Determine max length based on the case group
    case_max_len = 200 if case in group_2D else 500

    for strategy in strategies_order:
        # Check if strategy exists for this case in our pre-calculated dict
        if strategy not in relative_error_dict.get(case, {}):
            continue
        
        # Get all experiments for this specific case and strategy
        exp_data = relative_error_dict[case][strategy]
        if not exp_data:
            continue

        # Convert the dictionary of arrays into a list of arrays
        relative_diffs_all_exp = list(exp_data.values())

        # --- Handle incomplete runs by padding with NaNs ---
        max_len_available = max(len(diff) for diff in relative_diffs_all_exp)
        max_len = min(max_len_available, case_max_len)

        n_runs = len(relative_diffs_all_exp)
        rel_diff_mat = np.full((n_runs, max_len), np.nan, dtype=float)

        for r, diff in enumerate(relative_diffs_all_exp):
            this_len = min(len(diff), max_len)
            rel_diff_mat[r, :this_len] = diff[:this_len]

        # Calculate statistics
        p2_5  = np.nanpercentile(rel_diff_mat,  2.5, axis=0)
        p50   = np.nanpercentile(rel_diff_mat, 50.0, axis=0)
        p97_5 = np.nanpercentile(rel_diff_mat, 97.5, axis=0)

        # Gaussian smoothing
        p2_5_smooth   = gaussian_filter1d(p2_5,  sigma=sigma)
        p50_smooth    = gaussian_filter1d(p50,   sigma=sigma)
        p97_5_smooth  = gaussian_filter1d(p97_5, sigma=sigma)
        
        # p2_5_smooth   = p2_5
        # p50_smooth    = p50
        # p97_5_smooth  = p97_5

        # X-axis
        steps = np.arange(doe, doe + max_len)
        pretty_label = custom_legend[strategies_order.index(strategy)]

        ax.plot(
            steps, p50_smooth,
            label=pretty_label,
            color=strategy_colors[strategy],
            linewidth=linewidth
        )
        ax.fill_between(
            steps, p2_5_smooth, p97_5_smooth,
            color=strategy_colors[strategy],
            alpha=0.1
        )

    # Target epsilon lines
    # for eps in target_epsilon.get(case, []):
    #     ax.axhline(y=eps, color=epsilon_color, linestyle='--', linewidth=0.5, alpha=0.8)

    for j, eps in enumerate(target_epsilon[case]):
        # We only assign a label to the VERY FIRST line of the FIRST subplot
        line_label = r"$\delta P_{\mathrm{f,target}}$" if (i == 0 and j == 0) else None
        ax.axhline(
            y=eps,
            color=epsilon_color,
            linestyle='--',
            linewidth=0.5,
            alpha=0.8,
            label=line_label  # Assign label here
        )
    if i == 5:
        # loc='upper right' or 'lower left' depending on your data flow
        ax.legend(handles=[plt.Line2D([0], [0], color=epsilon_color, linestyle='--', linewidth=0.5)],
                  labels=[r"$\delta P_{\mathrm{F,target}}$"],
                  loc='lower center', 
                #   frameon=False, 
                  fontsize=font_size - 1)
    
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1)
    ax.grid(True, which="both", linewidth=0.01, alpha=0.3)

# X-limits and ticks per case group
for ax, case in zip(axs, casestudy):
    if case in group_2D:
        ax.set_xlim(10, 200)
        ax.set_xticks([10, 100, 200])
    else:
        ax.set_xlim(10, 500)
        ax.set_xticks([10, 300, 500])
    ax.tick_params(width=0.3)

# --- Figure-level legend -----------------------------------------------------
handles = []
for strat_key, label in zip(strategies_order, custom_legend):
    # Only append the strategy markers
    handles.append(
        plt.Line2D(
            [0], [0],
            color=strategy_colors[strat_key],
            marker='s', markersize=5,
            linestyle='', label=label
        )
    )

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=5,                # 10 strategies / 5 columns = 2 rows
    fontsize=font_size,
    columnspacing=1.0,
    handletextpad=0.4,
    handlelength=1.5,
    bbox_to_anchor=(0.5, -0.38), # Adjust vertical position if needed
)

fig.text(
    0.5, -0.07,
    "Number of acquired samples",
    ha='center',
    va='center',
    fontsize=font_size
)

fig.subplots_adjust(wspace=0.2, hspace=1.2)

for ax in axs:
    ax.tick_params(width=0.3, which='minor')
    ax.tick_params(width=0.3, which='major')
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
finalize_figure(fig, "F01")

# ---------------------------------------------------------------------------
# Figure F02: Pf evolution (two-row grouped layout)
# ---------------------------------------------------------------------------
epsilon_color = '#4d4d4d'

plt.rcParams.update({
    'font.size': font_size,
    'legend.fontsize': font_size,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
})

remaining_cases = casestudy[:-1]

# Define Strategy Groups
row1_strats = ['moo_reliability', 'moo_knee', 'moo_compromise', 'moo_eps_ew', 'portfolio']
row2_strats = ['erf', 'reif', 'reif2', 'eff', 'u']
all_strats = row1_strats + row2_strats + [eier_reference_strategy]
strat_to_label = dict(zip(strategies_order, custom_legend))

fig, axs = plt.subplots(
    2, len(remaining_cases),
    figsize=(17.5*cm, 10*cm), 
    sharey=True, sharex=False,
)

for col_idx, case in enumerate(remaining_cases):
    case_max_len = 200 if case in group_2D else 500
    
    # 1. PRE-CALCULATE ENVELOPE (Min-Max of smoothed MEANs across all strategies)
    all_means_in_case = []
    for strategy in all_strats:
        if strategy not in relative_error_dict.get(case, {}): continue
        
        exp_data = relative_error_dict[case][strategy]
        relative_diffs_all_exp = list(exp_data.values())
        max_len = min(max(len(diff) for diff in relative_diffs_all_exp), case_max_len)
        
        # Build matrix
        n_runs = len(relative_diffs_all_exp)
        rel_diff_mat = np.full((n_runs, max_len), np.nan)
        for r, diff in enumerate(relative_diffs_all_exp):
            this_len = min(len(diff), max_len)
            rel_diff_mat[r, :this_len] = diff[:this_len]
        
        # Calculate median and apply Smoothing
        median_val = np.nanmedian(rel_diff_mat, axis=0)
        smoothed_median = gaussian_filter1d(median_val, sigma=sigma)
        all_means_in_case.append(smoothed_median)

        # # --- PRINTING LOGIC ---
        # # Observe the first (after DoE) and last available mean values
        # final_val = smoothed_median[-1]
        # initial_val = smoothed_median[0]
        # print(f"{case[:15]:<15} | {strategy[:15]:<15} | {initial_val:.2e}      | {final_val:.2e}")

    # Determine global min/max bounds across smoothed means
    max_steps_case = max(len(m) for m in all_means_in_case)
    comp_mat = np.full((len(all_means_in_case), max_steps_case), np.nan)
    for i, m in enumerate(all_means_in_case):
        comp_mat[i, :len(m)] = m
    
    global_min_mean = np.nanmin(comp_mat, axis=0)
    global_max_mean = np.nanmax(comp_mat, axis=0)
    ref_steps = np.arange(doe, doe + max_steps_case)

    # Reference strategy (EIER): plotted in both rows, excluded from row groups.
    eier_steps = None
    eier_median_s = None
    if eier_reference_strategy in relative_error_dict.get(case, {}):
        eier_data = relative_error_dict[case][eier_reference_strategy]
        eier_runs = list(eier_data.values())
        if eier_runs:
            eier_max_len = min(max(len(diff) for diff in eier_runs), case_max_len)
            eier_mat = np.full((len(eier_runs), eier_max_len), np.nan)
            for r, diff in enumerate(eier_runs):
                this_len = min(len(diff), eier_max_len)
                eier_mat[r, :this_len] = diff[:this_len]
            eier_median_raw = np.nanpercentile(eier_mat, 50.0, axis=0)
            eier_median_s = gaussian_filter1d(eier_median_raw, sigma=sigma)
            eier_steps = np.arange(doe, doe + eier_max_len)

    # 2. PLOT ROWS
    for row_idx, current_group in enumerate([row1_strats, row2_strats]):
        ax = axs[row_idx, col_idx]
        
        # Individual Strategies
        for strategy in current_group:
            if strategy not in relative_error_dict.get(case, {}): continue
            
            exp_data = relative_error_dict[case][strategy]
            relative_diffs_all_exp = list(exp_data.values())
            max_len = min(max(len(diff) for diff in relative_diffs_all_exp), case_max_len)

            rel_diff_mat = np.full((len(relative_diffs_all_exp), max_len), np.nan)
            for r, diff in enumerate(relative_diffs_all_exp):
                this_len = min(len(diff), max_len)
                rel_diff_mat[r, :this_len] = diff[:this_len]

            # Statistics: Mean and Percentiles
            # mean_raw = np.nanmean(rel_diff_mat, axis=0)
            median_raw = np.nanpercentile(rel_diff_mat, 50.0,axis=0)
            p2_5_raw = np.nanpercentile(rel_diff_mat, 2.5, axis=0)
            p97_5_raw = np.nanpercentile(rel_diff_mat, 97.5, axis=0)

            # Apply Smoothing
            median_s = gaussian_filter1d(median_raw, sigma=sigma)
            p2_5_s = gaussian_filter1d(p2_5_raw, sigma=sigma)
            p97_5_s = gaussian_filter1d(p97_5_raw, sigma=sigma)

            steps = np.arange(doe, doe + max_len)
            ax.plot(steps, median_s, color=strategy_colors[strategy], linewidth=linewidth, zorder=2, alpha=0.9)
            ax.fill_between(steps, p2_5_s, p97_5_s, color=strategy_colors[strategy], alpha=0.1, zorder=1)

        # EIER as a common reference curve in both rows.
        if eier_steps is not None and eier_median_s is not None:
            ax.plot(
                eier_steps,
                eier_median_s,
                color=strategy_colors[eier_reference_strategy],
                linewidth=linewidth,
                marker=eier_reference_marker,
                markersize=3.0,
                markevery=max(1, len(eier_steps) // 10),
                markerfacecolor='white',
                markeredgewidth=0.6,
                zorder=20,
                alpha=0.9,
            )

        # Plot Min-Max Mean Reference Lines ON TOP (Higher zorder)
        ax.plot(ref_steps, global_min_mean, color='black', dashes=(3, 3), linewidth=linewidth-0.1, alpha=0.8, zorder=10)
        ax.plot(ref_steps, global_max_mean, color='black', dashes=(3, 3), linewidth=linewidth-0.1, alpha=0.8, zorder=10)

        # 3. FORMATTING
        if row_idx == 0: ax.set_title(f"{custom_titles[col_idx]}", fontsize=font_size)
        if col_idx == 0: ax.set_ylabel(r"$\delta P_\mathrm{F}$")
        
        # Stringer threshold only (the minimum value in target_epsilon)
        strictest_eps = min(target_epsilon[case])
        ax.axhline(y=strictest_eps, color=epsilon_color, linestyle=':', linewidth=0.8, alpha=0.7)
        
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1)
        ax.grid(True, which="both", linewidth=0.01, alpha=0.3)

        if case in group_2D:
            ax.set_xlim(10, 200)
            ax.set_xticks([10, 100, 200])
        else:
            ax.set_xlim(10, 500)
            ax.set_xticks([10, 250, 500])

# --- Legend Alignment & Figure Text ---
fig.text(0.5, 0.575, "Number of acquired samples", ha='center', fontsize=font_size)
fig.text(0.5, 0.125, "Number of acquired samples", ha='center', fontsize=font_size)

def get_handles(strats):
    return [plt.Line2D([0], [0], color=strategy_colors[s], marker='s', 
            markersize=4, linestyle='', label=strat_to_label[s]) for s in strats]

h1 = get_handles(row1_strats)
h2 = get_handles(row2_strats)

h3 = []
h3.append(plt.Line2D([0], [0], color=epsilon_color, ls=':', lw=0.8, label=r'$\delta P_{\mathrm{F,target}}$'))
h3.append(plt.Line2D([0], [0], color='black', ls='--', lw=0.8, label='min-max medians'))
h3.append(
    plt.Line2D(
        [0],
        [0],
        color=strategy_colors[eier_reference_strategy],
        marker=eier_reference_marker,
        markersize=4,
        lw=linewidth + 0.1,
        label=strategy_label(eier_reference_strategy),
        markerfacecolor='white',
        markeredgewidth=0.8
    )
)

# Standardize spacing for visual balance
common_params = {'loc': "lower center", 'fontsize': font_size, 'frameon': True, 'handlelength': 1.0}
fig.legend(handles=h1, ncol=len(h1), bbox_to_anchor=(0.5, 0.499), columnspacing=0.8, **common_params)
fig.legend(handles=h2, ncol=len(h2), bbox_to_anchor=(0.4, 0.047), columnspacing=0.9, **common_params)
fig.legend(handles=h3, ncol=len(h3), bbox_to_anchor=(0.78, 0.047), columnspacing=0.9, **common_params)

for ax in axs.flat:
    ax.tick_params(width=0.3, which='minor')
    ax.tick_params(width=0.3, which='major')
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
      
plt.subplots_adjust(
    left=0.08, 
    right=0.95, 
    top=0.95, 
    bottom=0.20, # Increases space at the very bottom
    hspace=0.51, # Increases space between row 1 and row 2
    wspace=0.25
)
finalize_figure(fig, "F02")

# then we create the ranking per seed
from typing import Optional
ranking_metadata = {
    "required_consecutive": required_consecutive
}
seed_ranking_dict = {}

def find_first_hit_index(rel_diff: np.ndarray,
                         threshold: float,
                         required_consecutive: int) -> Optional[int]:
    """
    Returns the first index i such that:
        rel_diff[i : i + required_consecutive] <= threshold
    for all elements in that window.
    If no such i exists, returns None.
    """
    if rel_diff.size == 0:
        return None

    is_below = rel_diff <= threshold
    max_start = rel_diff.size - required_consecutive

    if max_start < 0:
        return None

    for i in range(max_start + 1):
        # Check if all values in this window are True
        if np.all(is_below[i : i + required_consecutive]):
            return i

    return None

# --- Ranking all seeds per case based on threshold crossing ---------------
for case in casestudy:
    if case not in relative_error_dict:
        continue
    if case not in threshold_dict:
        report(f"[warn] no threshold found for case '{case}', skipping.")
        continue

    threshold = threshold_dict[case]["threshold_delta_pf"]

    # Decide max length for this case
    if case in group_2D:
        case_max_len = 200
        no_hit_value = 201     # put non-hit seeds at the end
    else:
        case_max_len = 500
        no_hit_value = 501     # analogous idea for HD cases

    seed_entries = []

    for strategy, exp_dict in relative_error_dict[case].items():
        for exp_num, rel_diff in exp_dict.items():
            rel_diff = np.asarray(rel_diff, dtype=float)
            if rel_diff.size == 0:
                continue

            cap_len = min(rel_diff.size, case_max_len)
            rel_diff = rel_diff[:cap_len]

            # 1) best δPf
            finite_mask = np.isfinite(rel_diff)
            if not np.any(finite_mask):
                continue

            finite_values = rel_diff[finite_mask]
            best_delta_pf = float(np.min(finite_values))
            best_idx_local = int(np.where(rel_diff == best_delta_pf)[0][0])
            best_samples = int(doe + best_idx_local)

            # 2) first hit index with required_consecutive below threshold
            first_hit_idx = find_first_hit_index(
                rel_diff, threshold=threshold, required_consecutive=required_consecutive
            )

            if first_hit_idx is not None:
                reached = True
                first_hit_samples = int(doe + first_hit_idx + (required_consecutive - 1))
                delta_at_hit = float(rel_diff[first_hit_idx])
            else:
                reached = False
                first_hit_samples = no_hit_value   # 201 (2D) / 501 (HD)
                delta_at_hit = None

            seed_entries.append({
                "case": case,
                "strategy": strategy,
                "exp_num": parse_experiment_number(exp_num),

                "reached_threshold": reached,
                "first_hit_idx": first_hit_idx,
                "first_hit_samples": first_hit_samples,
                "delta_at_hit": delta_at_hit,

                "best_delta_pf": best_delta_pf,
                "best_idx": best_idx_local,
                "best_samples": best_samples,

                "threshold_delta_pf": threshold,
            })

    if not seed_entries:
        report(f"[warn] no seed data for case '{case}'.")
        continue

    # --- Define NEW ranking rule: by first_hit_samples ---------------------
    # Priority:
    # 1) Smaller first_hit_samples (non-hit seeds already have large value)
    # 2) Among equal first_hit_samples: smaller delta_at_hit is better
    # 3) Among equal above: smaller best_delta_pf then earlier best_samples

    def seed_sort_key(entry):
        # Treat None delta_at_hit as +inf so non-hits go to the bottom on tie
        delta = entry["delta_at_hit"]
        if delta is None or not np.isfinite(delta):
            delta = np.inf
        return (
            entry["first_hit_samples"],
            delta,
            entry["best_delta_pf"],
            entry["best_samples"],
        )

    pointwise_entries = [
        entry for entry in seed_entries if entry["strategy"] != eier_reference_strategy
    ]
    reference_entries = [
        entry for entry in seed_entries if entry["strategy"] == eier_reference_strategy
    ]

    # Rank point-wise strategies only.
    pointwise_sorted = sorted(pointwise_entries, key=seed_sort_key)
    for rank_idx, entry in enumerate(pointwise_sorted, start=1):
        entry["global_rank"] = rank_idx

    # Rank EIER against point-wise ranks without changing point-wise ordering.
    for entry in reference_entries:
        current_key = seed_sort_key(entry)
        better_count = sum(seed_sort_key(other) < current_key for other in pointwise_sorted)
        entry["global_rank"] = better_count + 1

    seed_entries_sorted = pointwise_sorted + sorted(reference_entries, key=seed_sort_key)

    seed_ranking_dict[case] = {
        "threshold_delta_pf": threshold,
        "required_consecutive": required_consecutive,
        "seeds": seed_entries_sorted,
    }

strategy_rankings_dict = {}  # [case] -> list of {strategy, avg_rank, ...}

for case in casestudy:
    if case not in seed_ranking_dict:
        continue
    
    seeds = seed_ranking_dict[case]["seeds"]

    # Collect ranks per strategy
    ranks_by_strategy = {}
    for s in seeds:
        strat = s["strategy"]
        ranks_by_strategy.setdefault(strat, []).append(s["global_rank"])

    # Compute average rank and number of seeds
    avg_list = []
    for strat, ranks in ranks_by_strategy.items():
        median_rank = float(np.mean(ranks))
        avg_list.append({
            "strategy": strat,
            "median_rank": median_rank,
            "n_seeds": len(ranks)
        })

    # Sort strategies by median_rank (lower = better)
    avg_list_sorted = sorted(avg_list, key=lambda d: d["median_rank"])

    strategy_rankings_dict[case] = avg_list_sorted
    # Create a mapping for display names
display_names = {
    'moo_reliability': 'MOO-R', 'moo_knee': 'MOO-K', 
    'moo_compromise': 'MOO-C', 'moo_eps_ew': 'MOO-LD',
    'erf': 'ERF', 'reif': 'REIF', 'reif2': 'REIF2', 
    'eff': 'EFF', 'u': 'U', 'portfolio': 'Portfolio', 'eier': 'EIER'
}

if debug_case_for_print and debug_case_for_print in seed_ranking_dict:
    case = debug_case_for_print
    seeds = seed_ranking_dict[case]["seeds"]
    
    # Dictionary to store sample counts per strategy
    samples_by_strat = {}
    for s in seeds:
        strat = s["strategy"]
        samples_by_strat.setdefault(strat, []).append(s["first_hit_samples"])
    
    report(f"{'Strategy':<15} | {'Avg. Rank':<10} | {'Avg. Samples':<12}")
    report("-" * 45)
    
    # Loop through the already sorted strategy_rankings_dict
    for entry in strategy_rankings_dict[case]:
        strat_key = entry['strategy']
        avg_rank = entry['median_rank']
        avg_samples = np.mean(samples_by_strat[strat_key])
        
        display = display_names.get(strat_key, strat_key)
        report(f"{display:<15} | {avg_rank:<10.2f} | {avg_samples:<12.1f}")
    report("")

# ---------------------------------------------------------------------------
# Figure F03: Number of failed experiments at t_max (stacked bars)
# ---------------------------------------------------------------------------
# --- 1. Data Aggregation ---
strategies_f03 = [s for s in strategies_order if s not in excluded_from_f03]

# Initialize a dictionary to store counts: counts[case][strategy]
exceeded_counts = {case: {strat: 0 for strat in strategies_f03} for case in casestudy}

for case in casestudy:
    if case not in seed_ranking_dict:
        continue
    seeds = seed_ranking_dict[case]["seeds"]
    for s in seeds:
        # Check if the budget was exceeded (threshold not reached)
        if not s["reached_threshold"]:
            strat = s["strategy"]
            if strat in exceeded_counts[case]:
                exceeded_counts[case][strat] += 1

# --- 2. Plotting Configuration ---
# X-axis labels: Display names for strategies
x_labels = [custom_legend[strategies_order.index(s)] for s in strategies_f03]
n_strategies = len(strategies_f03)

fig, ax = plt.subplots(figsize=(17.5*cm, 7*cm))

# Starting point for stacking
bottom = np.zeros(n_strategies)

# Use a colormap for the different limit-state functions
colors = plt.cm.viridis(np.linspace(0, 1, len(casestudy)))

# --- 3. Build the Stacked Bar Chart ---
for i, case in enumerate(casestudy):
    # Extract the list of counts for this case across all strategies
    counts = [exceeded_counts[case][strat] for strat in strategies_f03]
    
    # Label for the legend
    case_label = custom_titles[i] if i < len(custom_titles) else case
    
    # Plot the bar segment
    ax.bar(x_labels, counts, bottom=bottom, label=case_label, color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)
    
    # Update the 'bottom' for the next stack
    bottom += np.array(counts)

# --- 4. Formatting ---
ax.set_ylabel(
    r"Experiments exceeding $\delta P_{\mathrm{F,target}}$ at $t_{\max}$",
    fontsize=font_size,
)
ax.set_xlabel("Acquisition strategy", fontsize=font_size)
# ax.set_title("Reliability of Convergence: Seeds Failing to Reach $\delta P_{F,target}$", fontsize=font_size, pad=15)

# Add a horizontal grid for readability
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# Adjust legend
ax.legend(title="Limit-state Function", loc='upper left', bbox_to_anchor=(1, 1), frameon=True)

# Rotate x-labels for better fit
plt.xticks(rotation=0, ha='center')

plt.tight_layout()
# plt.savefig('aggregated_N_failed_exp.pdf')
finalize_figure(fig, "F03")

from matplotlib.legend_handler import HandlerLine2D

class HandlerVerticalLine(HandlerLine2D):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # We define a line that goes from bottom-center to top-center of the legend box
        # instead of the default left-to-right
        line = plt.Line2D([width/2, width/2], [0, height],
                          color=orig_handle.get_color(),
                          linestyle=orig_handle.get_linestyle(),
                          linewidth=orig_handle.get_linewidth())
        line.set_transform(trans)
        return [line]


# ---------------------------------------------------------------------------
# Figure F04: Distribution of required samples to hit threshold
# ---------------------------------------------------------------------------
# --- Configuration ---
remaining_cases = casestudy[:-1]
n_rows = 1 
n_cols = len(remaining_cases)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(17.5*cm, 6.5*cm), sharex=False, sharey=True)

if n_cols == 1: axs = [axs]

for i, case in enumerate(remaining_cases):
    ax = axs[i]
    
    # Position Case Titles
    ax.text(0.5, 1.25, f"{custom_titles[i]}", transform=ax.transAxes,
            ha='center', va='center', fontsize=font_size)

    limit = 201 if case in group_2D else 501
    mid_limit = 100 if case in group_2D else 250
    stability_threshold = target_epsilon[case][-1]

    # --- 1. GET THE CUSTOM SORTING ORDER FOR THIS CASE ---
    # We take the list of dicts and extract just the strategy names
    # Reverse it so the best (rank 1) is at the top of the y-axis
    case_ranking = strategy_rankings_dict.get(case, [])
    sorted_order_names = [item['strategy'] for item in case_ranking][::-1]

    # --- 2. PLOT FOLLOWING THE RANKING ORDER ---
    for idx, strategy in enumerate(sorted_order_names):
        # Extract data from the relative_error_dict
        data = []
        if strategy in relative_error_dict.get(case, {}):
            exp_results = relative_error_dict[case][strategy]
            for rel_diff_evolution in exp_results.values():
                consecutive_count = 0
                found = False
                for k, diff in enumerate(rel_diff_evolution):
                    if diff < stability_threshold:
                        consecutive_count += 1
                        if consecutive_count >= required_consecutive:
                            data.append((k + 1) + doe)
                            found = True
                            break
                    else:
                        consecutive_count = 0
                if not found: data.append(limit)

        if not data:
            continue

        # Compute stats
        mean_val = np.mean(data)
        median_val = np.median(data)
        p25, p75 = np.percentile(data, [2.5, 97.5])

        # # --- DUAL-COLOR BACKGROUND LOGIC ---
        # if strategy in ['moo_reliability', 'moo_eps_ew']:
        #     ax.axhspan(idx - 0.5, idx + 0.5, color='blue', alpha=0.05, zorder=0, lw=0)
        # elif strategy in ['moo_knee', 'moo_compromise']:
        #     ax.axhspan(idx - 0.5, idx + 0.5, color='orange', alpha=0.07, zorder=0, lw=0)

        if strategy in ['moo_reliability', 'moo_eps_ew']:
            ax.axhspan(idx - 0.5, idx + 0.5, color='black', alpha=0.2, zorder=0, lw=0)
        elif strategy in ['moo_knee', 'moo_compromise']:
            ax.axhspan(idx - 0.5, idx + 0.5, color='black', alpha=0.07, zorder=0, lw=0)

        color = strategy_colors[strategy]
        is_moo = 'moo' in strategy.lower()
        st_alpha = 0.9
        z_ord = 5 if is_moo else 2

        # Plot Horizontal IQR Line
        if strategy == eier_reference_strategy:
            # Use Line2D for EIER so plot caps match the legend rendering.
            eier_line, = ax.plot(
                [p25, p75],
                [idx, idx],
                color='white',
                linestyle='-',
                linewidth=0.4,
                alpha=st_alpha,
                zorder=z_ord,
            )
            eier_line.set_solid_capstyle('butt')
            eier_line.set_path_effects(
                [
                    pe.Stroke(
                        linewidth=1.2, foreground='black', capstyle='projecting', joinstyle='miter'
                    ),
                    pe.Normal(),
                ]
            )
        else:
            ax.hlines(
                y=idx,
                xmin=p25,
                xmax=p75,
                color=color,
                linestyle='-',
                linewidth=1.0,
                alpha=st_alpha,
                zorder=z_ord,
            )
        
        # Vertical Line: MEAN (Solid)
        ax.vlines(x=mean_val, ymin=idx-0.32, ymax=idx+0.32, 
                  color=color, linestyle='-', linewidth=0.7, alpha=st_alpha, zorder=z_ord+1)
        
        # Vertical Line: MEDIAN (Dashed)
        ax.vlines(x=median_val, ymin=idx-0.32, ymax=idx+0.32, 
                  color='black', linestyle='--', linewidth=0.6, alpha=st_alpha, zorder=z_ord+2)

    # Formatting
    mantissa = f"{stability_threshold:.0e}".split('e')[0]
    exponent = int(f"{stability_threshold:.0e}".split('e')[1])
    ax.set_title(rf"$\delta P_{{\mathrm{{F,target}}}} = {mantissa} \cdot 10^{{{exponent}}}$", 
                 fontsize=font_size-1, pad=10)

    ax.grid(True, axis='x', linewidth=0.2, alpha=0.3)
    ax.set_xlim(10, limit)
    ax.set_xticks([10, mid_limit, limit])
    ax.set_xticklabels(['10', f'{mid_limit}', f'>{limit-1}'])
    ax.set_yticks([])

# --- Legend and Labels ---
handles = []
handles2 = []
for strat_key in strategies_order:
    if strat_key == eier_reference_strategy:
        continue
    label = custom_legend[strategies_order.index(strat_key)]
    handles.append(plt.Line2D([0], [0], color=strategy_colors[strat_key], lw=2, label=label))

# Define Mean and Median handles specifically to map them to the handler
mean_handle = plt.Line2D([0], [0], color='black', linestyle='-', lw=1.0, label='Mean')
median_handle = plt.Line2D([0], [0], color='black', linestyle='--', lw=0.6, label='Median')
eier_handle = plt.Line2D(
    [0, 1],
    [0, 0],
    color='white',
    linestyle='-',
    linewidth=0.4,
    label='EIER',
)
eier_handle.set_solid_capstyle('butt')
eier_handle.set_path_effects(
    [pe.Stroke(linewidth=1.2, foreground='black', capstyle='projecting', joinstyle='miter'), pe.Normal()]
)
iqr_patch = plt.Line2D(
    [0],
    [0],
    color='black',
    linestyle='-',
    linewidth=1.5,
    label='2.5th-97.5th percentile',
)
handles2.extend([eier_handle, iqr_patch, mean_handle, median_handle])

# fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=font_size, 
#            bbox_to_anchor=(0.5, 0.03), columnspacing=0.8, handlelength=1.5)
fig.legend(
    handles=handles, 
    loc="lower center", 
    ncol=5, 
    fontsize=font_size, 
    bbox_to_anchor=(0.28, 0.03), 
    columnspacing=0.8, 
    handlelength=1.0, # Reduced to make vertical lines look centered
)

# fig.legend(handles=h1, ncol=len(h1), bbox_to_anchor=(0.5, 0.499), columnspacing=0.8, **common_params)
fig.legend(handles=handles2, ncol=len(handles2), bbox_to_anchor=(0.736, 0.055), columnspacing=0.9, **common_params,
               numpoints=2,
               handler_map={mean_handle: HandlerVerticalLine(), 
                 median_handle: HandlerVerticalLine()})

fig.text(0.5, 0.22, "Number of acquired samples", ha='center', fontsize=font_size)

plt.subplots_adjust(
    left=0.05, 
    right=0.95, 
    top=0.82, 
    bottom=0.35, # Increases space at the very bottom
    hspace=0.51, # Increases space between row 1 and row 2
    wspace=0.25
)

# Global spine cleanup
for ax in (axs if isinstance(axs, np.ndarray) else [axs]):
    ax.tick_params(width=0.3, which='minor')
    ax.tick_params(width=0.3, which='major')
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

finalize_figure(fig, "F04")

# I want to isolate high_dim for the ranking table
all_strategies = [
    'moo_reliability', 'moo_knee', 'moo_compromise',
    'moo_eps_ew', 'eff', 'u', 'erf', 'reif', 'reif2', 'portfolio', 'eier'
]

all_limit_states = [
    'four_branch_6', 'four_branch_7',
    'hat', 'himmelblau',
    'nonlinear_oscillator', '2dof_oscillator', #'high_dimensional'
]

# DataFrames: rows = strategies, columns = limit states
rank_df    = pd.DataFrame(index=all_strategies, columns=all_limit_states, dtype=float)
samples_df = pd.DataFrame(index=all_strategies, columns=all_limit_states, dtype=float)

for case in all_limit_states:
    if case not in seed_ranking_dict:
        report(f"[warn] no seed ranking data for case '{case}', skipping.")
        continue

    seeds = seed_ranking_dict[case]["seeds"]

    # Collect per-strategy data for this case
    per_strat = {}  # strategy -> { "ranks": [...], "samples": [...] }

    for s in seeds:
        strat = s["strategy"]
        if strat not in all_strategies:
            continue

        per_strat.setdefault(strat, {"ranks": [], "samples": []})
        per_strat[strat]["ranks"].append(s["global_rank"])
        # first_hit_samples already encodes non-hits as 201 / 501
        per_strat[strat]["samples"].append(s["first_hit_samples"])

    # Fill the DataFrames with means
    for strat, data in per_strat.items():
        avg_rank    = float(np.mean(data["ranks"]))
        avg_samples = float(np.mean(data["samples"]))

        rank_df.loc[strat, case]    = avg_rank
        samples_df.loc[strat, case] = avg_samples

# Global average rank across selected limit states (row-wise mean)
rank_df["global_avg_rank"] = rank_df.mean(axis=1, skipna=True)

# Global average first_hit_samples across selected limit states
samples_df["global_avg_samples"] = samples_df.mean(axis=1, skipna=True)

# Build final table: ranks per case + global averages + avg samples
final_df = rank_df.copy()
final_df["global_avg_samples"] = samples_df["global_avg_samples"]

# Reorder columns: global_avg_rank first, then each case, then avg samples
ordered_cols =  all_limit_states + ["global_avg_rank"]+ ["global_avg_samples"]
final_df = final_df[ordered_cols]

# Sort strategies by global_avg_rank (lower rank = better)
final_df_sorted = final_df.sort_values("global_avg_rank")

report("Average seed rank per strategy and limit state:")
table_case_cols = {case: _case_console_name(case) for case in all_limit_states}
table_case_cols["global_avg_rank"] = "Avg. Seed Rank (down)"
table_case_cols["global_avg_samples"] = "Avg. Samples"

final_df_print = final_df_sorted.rename(columns=table_case_cols).copy()
final_df_print.index = [strategy_label(s) for s in final_df_print.index]
final_df_print.index.name = "Strategy"
final_df_print = final_df_print.apply(
    lambda col: col.map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
)
report(final_df_print.to_string())
report("")

# Sample-count tables per case, ordered as in the global average rank table
ordered_strategies_by_global_rank = list(final_df_sorted.index)

for case in casestudy:
    if case not in seed_ranking_dict:
        report(f"[warn] no seed ranking data for case '{case}', skipping sample-count table.")
        continue

    report(f"{'='*60}")
    report(f"CASE STUDY: {_case_console_name(case)}")
    report(f"{'='*60}")

    case_threshold = threshold_dict.get(case, {}).get("threshold_delta_pf", None)
    if case_threshold is not None:
        report(f"Target Delta Pf: {_fmt_sci_compact(case_threshold)}")
    else:
        report("Target Delta Pf: N/A")

    report(f"{'Strategy':<20} | {'Mean':>7} | {'Median':>7} | (2.5% - 97.5%)")
    report("-" * 60)

    seeds = seed_ranking_dict[case]["seeds"]
    per_strat_samples = {}
    for s in seeds:
        strat = s["strategy"]
        if strat not in all_strategies:
            continue
        per_strat_samples.setdefault(strat, []).append(float(s["first_hit_samples"]))

    for strat in ordered_strategies_by_global_rank:
        values = per_strat_samples.get(strat, [])
        label = strategy_label(strat)
        if len(values) == 0:
            report(f"{label:<20} | {'N/A':>7} | {'N/A':>7} | (N/A - N/A)")
            continue

        arr = np.asarray(values, dtype=float)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        p2_5, p97_5 = np.percentile(arr, [2.5, 97.5])
        report(
            f"{label:<20} | {mean_val:>7.2f} | {median_val:>7.2f} | "
            f"({p2_5:>6.2f} - {p97_5:>6.2f})"
        )
    report("")

# Additional ranking table for the high-dimensional case only
high_dim_case = "high_dimensional"
if high_dim_case in seed_ranking_dict:
    seeds_hd = seed_ranking_dict[high_dim_case]["seeds"]
    per_strat_hd = {}

    for s in seeds_hd:
        strat = s["strategy"]
        if strat not in all_strategies:
            continue
        per_strat_hd.setdefault(strat, {"ranks": [], "samples": []})
        per_strat_hd[strat]["ranks"].append(float(s["global_rank"]))
        per_strat_hd[strat]["samples"].append(float(s["first_hit_samples"]))

    hd_rows = []
    for strat, vals in per_strat_hd.items():
        ranks_arr = np.asarray(vals["ranks"], dtype=float)
        samples_arr = np.asarray(vals["samples"], dtype=float)
        p2_5, p97_5 = np.percentile(samples_arr, [2.5, 97.5])
        median_val = float(np.median(samples_arr))

        hd_rows.append(
            {
                "Ranked Strategy": strategy_label(strat),
                "Avg. Seed Rank (down)": float(np.mean(ranks_arr)),
                "Avg. Samples": float(np.mean(samples_arr)),
                "Median (2.5th, 97.5th)": f"{median_val:.1f} ({p2_5:.1f}, {p97_5:.1f})",
            }
        )

    hd_rank_df = pd.DataFrame(hd_rows).sort_values("Avg. Seed Rank (down)", ascending=True)
    hd_rank_df.insert(0, "Rank", np.arange(1, len(hd_rank_df) + 1))

    hd_rank_df["Avg. Seed Rank (down)"] = hd_rank_df["Avg. Seed Rank (down)"].map(lambda v: f"{v:.2f}")
    hd_rank_df["Avg. Samples"] = hd_rank_df["Avg. Samples"].map(lambda v: f"{v:.2f}")

    report(f"Ranking summary for case '{high_dim_case}':")
    report(hd_rank_df.to_string(index=False))
    report("")
else:
    report(f"[warn] no seed ranking data for case '{high_dim_case}', skipping high-dimensional ranking table.")

# ---------------------------------------------------------------------------
# Figure F05: Pf evolution (high-dimensional case)
# ---------------------------------------------------------------------------
# --- Setup Target ---
target_case = casestudy[-1]

# --- Plot Config ---
plt.rcParams.update({
    'font.size': font_size,
    'legend.fontsize': font_size,
    'legend.title_fontsize': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
})

fig, ax = plt.subplots(figsize=(8.75*cm, 7.6*cm))
epsilon_color = '#4d4d4d'
case_max_len = 500
hd_pointwise_strats = [s for s in strategies_order if s != eier_reference_strategy]
hd_all_strats = hd_pointwise_strats + [eier_reference_strategy]

all_medians_in_case = []
eier_steps = None
eier_median_s = None

# --- 1. Data Processing & Plotting Strategies ---
for strategy in hd_all_strats:
    if strategy not in relative_error_dict.get(target_case, {}):
        continue

    exp_data = relative_error_dict[target_case][strategy]
    if not exp_data:
        continue

    relative_diffs_all_exp = list(exp_data.values())
    max_len_available = max(len(diff) for diff in relative_diffs_all_exp)
    max_len = min(max_len_available, case_max_len)

    rel_diff_mat = np.full((len(relative_diffs_all_exp), max_len), np.nan, dtype=float)
    for r, diff in enumerate(relative_diffs_all_exp):
        this_len = min(len(diff), max_len)
        rel_diff_mat[r, :this_len] = diff[:this_len]

    p2_5 = np.nanpercentile(rel_diff_mat, 2.5, axis=0)
    p50 = np.nanpercentile(rel_diff_mat, 50.0, axis=0)
    p97_5 = np.nanpercentile(rel_diff_mat, 97.5, axis=0)

    p2_5_s = gaussian_filter1d(p2_5, sigma=sigma)
    p50_s = gaussian_filter1d(p50, sigma=sigma)
    p97_5_s = gaussian_filter1d(p97_5, sigma=sigma)
    all_medians_in_case.append(p50_s)

    steps = np.arange(doe, doe + max_len)
    if strategy == eier_reference_strategy:
        eier_steps = steps
        eier_median_s = p50_s
        continue

    ax.plot(
        steps,
        p50_s,
        color=strategy_colors[strategy],
        linewidth=linewidth,
        zorder=5,
        alpha=0.9,
    )
    ax.fill_between(
        steps,
        p2_5_s,
        p97_5_s,
        color=strategy_colors[strategy],
        alpha=0.1,
        zorder=4,
    )

# EIER as reference curve (same visual language as Figure F02)
if eier_steps is not None and eier_median_s is not None:
    ax.plot(
        eier_steps,
        eier_median_s,
        color=strategy_colors[eier_reference_strategy],
        linewidth=linewidth,
        marker=eier_reference_marker,
        markersize=4.0,
        markevery=max(1, len(eier_steps) // 10),
        markerfacecolor='white',
        markeredgewidth=0.6,
        zorder=20,
        alpha=0.9,
    )

# --- 2. Calculate and Plot Global Min-Max Reference ---
if all_medians_in_case:
    max_steps_case = max(len(m) for m in all_medians_in_case)
    comp_mat = np.full((len(all_medians_in_case), max_steps_case), np.nan)
    for i, m in enumerate(all_medians_in_case):
        comp_mat[i, :len(m)] = m

    global_min_mean = np.nanmin(comp_mat, axis=0)
    global_max_mean = np.nanmax(comp_mat, axis=0)
    ref_steps = np.arange(doe, doe + max_steps_case)

    ax.plot(
        ref_steps,
        global_min_mean,
        color='black',
        dashes=(3, 3),
        linewidth=linewidth,
        alpha=0.8,
        zorder=10,
    )
    ax.plot(
        ref_steps,
        global_max_mean,
        color='black',
        dashes=(3, 3),
        linewidth=linewidth,
        alpha=0.8,
        zorder=10,
    )

# --- 3. Threshold and Formatting ---
strictest_eps = min(target_epsilon[target_case])
ax.axhline(
    y=strictest_eps,
    color=epsilon_color,
    linestyle=':',
    linewidth=0.8,
    alpha=0.7,
    zorder=2,
)

ax.set_ylabel(r"$\delta P_\mathrm{F}$")
ax.set_xlabel("Number of acquired samples")
ax.set_yscale('log')
ax.set_ylim(1e-3, 1)
ax.set_xlim(doe, case_max_len)
ax.set_xticks([10, 100, 200, 300, 400, 500])
ax.grid(True, which="both", linewidth=0.01, alpha=0.3)

# --- 4. Legends (point-wise + reference handles) ---
handles_pointwise = []
for strat_key in hd_pointwise_strats:
    handles_pointwise.append(
        plt.Line2D(
            [0],
            [0],
            color=strategy_colors[strat_key],
            marker='s',
            markersize=4,
            linestyle='',
            label=strategy_label(strat_key),
        )
    )

handles_ref = []
handles_ref.append(
    plt.Line2D(
        [0],
        [0],
        color=epsilon_color,
        ls=':',
        lw=0.8,
        label=r'$\delta P_{\mathrm{F,target}}$',
    )
)
handles_ref.append(
    plt.Line2D(
        [0],
        [0],
        color='black',
        ls='--',
        lw=0.8,
        label='min-max medians',
    )
)
handles_ref.append(
    plt.Line2D(
        [0],
        [0],
        color=strategy_colors[eier_reference_strategy],
        marker=eier_reference_marker,
        markersize=4,
        lw=linewidth + 0.1,
        label=strategy_label(eier_reference_strategy),
        markerfacecolor='white',
        markeredgewidth=0.8,
    )
)

common_params_hd = {'loc': "lower center", 'fontsize': font_size, 'frameon': True, 'handlelength': 1.0}
fig.legend(handles=handles_pointwise, ncol=5, bbox_to_anchor=(0.53, 0.00), columnspacing=0.6, **common_params_hd)
fig.legend(handles=handles_ref, ncol=3, bbox_to_anchor=(0.52, 0.125), columnspacing=0.8, **common_params_hd)

plt.subplots_adjust(
    left=0.15,
    right=0.96,
    top=0.95,
    bottom=0.34,
    hspace=0.51,
    wspace=0.25,
)

finalize_figure(fig, "F05")

# ---------------------------------------------------------------------------
# Figure F06: Sample-efficiency distribution (high-dimensional case)
# ---------------------------------------------------------------------------
# --- Configuration for High-Dimensional Case ---
case = casestudy[-1]
limit = 501
doe = 10
stability_threshold = target_epsilon[case][-1]

fig, ax = plt.subplots(figsize=(8.75*cm, 7.5*cm))

# --- 1. Get Ranking and Sorting ---
case_ranking = strategy_rankings_dict.get(case, [])
sorted_order_names = [item['strategy'] for item in case_ranking][::-1]
ytick_labels = []

# --- 2. Plotting ---
for strategy in sorted_order_names:
    data = []
    if strategy in relative_error_dict.get(case, {}):
        exp_results = relative_error_dict[case][strategy]
        for rel_diff_evolution in exp_results.values():
            consecutive_count = 0
            found = False
            for k, diff in enumerate(rel_diff_evolution):
                if diff < stability_threshold:
                    consecutive_count += 1
                    if consecutive_count >= required_consecutive:
                        data.append((k + 1) + doe)
                        found = True
                        break
                else:
                    consecutive_count = 0
            if not found: 
                data.append(limit)

    if not data:
        continue

    idx = len(ytick_labels)
    pretty_label = custom_legend[strategies_order.index(strategy)]
    ytick_labels.append(pretty_label)

    mean_val = np.mean(data)
    median_val = np.median(data)
    p25, p75 = np.percentile(data, [2.5, 97.5])

    if strategy in ['moo_reliability', 'moo_eps_ew']:
        ax.axhspan(idx - 0.5, idx + 0.5, color='black', alpha=0.2, zorder=0, lw=0)
    elif strategy in ['moo_knee', 'moo_compromise']:
        ax.axhspan(idx - 0.5, idx + 0.5, color='black', alpha=0.07, zorder=0, lw=0)

    color = strategy_colors[strategy]
    is_moo = 'moo' in strategy.lower()
    st_alpha = 0.9
    z_ord = 5 if is_moo else 2

    if strategy == eier_reference_strategy:
        eier_line, = ax.plot(
            [p25, p75],
            [idx, idx],
            color='white',
            linestyle='-',
            linewidth=0.4,
            alpha=st_alpha,
            zorder=z_ord,
        )
        eier_line.set_solid_capstyle('butt')
        eier_line.set_path_effects(
            [
                pe.Stroke(
                    linewidth=1.5, foreground='black', capstyle='projecting', joinstyle='miter'
                ),
                pe.Normal(),
            ]
        )
    else:
        ax.hlines(
            y=idx,
            xmin=p25,
            xmax=p75,
            color=color,
            linestyle='-',
            linewidth=1.5,
            alpha=st_alpha,
            zorder=z_ord,
        )

    ax.vlines(
        x=mean_val,
        ymin=idx - 0.32,
        ymax=idx + 0.32,
        color=color,
        linestyle='-',
        linewidth=0.7,
        alpha=st_alpha,
        zorder=z_ord + 1,
    )

    ax.vlines(
        x=median_val,
        ymin=idx - 0.32,
        ymax=idx + 0.32,
        color='black',
        linestyle='--',
        linewidth=0.6,
        alpha=st_alpha,
        zorder=z_ord + 2,
    )

# --- 3. Formatting ---
mantissa = f"{stability_threshold:.0e}".split('e')[0]
exponent = int(f"{stability_threshold:.0e}".split('e')[1])

ax.set_title(
    rf"$\delta P_{{\mathrm{{F,target}}}} = {mantissa} \cdot 10^{{{exponent}}}$",
    fontsize=font_size,
    pad=1,
)

ax.set_yticks(range(len(ytick_labels)))
ax.set_yticklabels(ytick_labels, fontsize=font_size)

ax.set_xlim(10, limit)
ax.set_xticks([10, 100, 200, 300, 400, limit])
ax.set_xticklabels(['10', '100', '200', '300', '400', f'>{limit-1}'])

ax.grid(True, axis='x', linewidth=0.2, alpha=0.3)

# Global spine cleanup
ax.tick_params(width=0.3, which='both')
for spine in ax.spines.values():
    spine.set_linewidth(0.3)

mean_handle = plt.Line2D([0], [0], color='black', linestyle='-', lw=1.0, label='Mean')
median_handle = plt.Line2D([0], [0], color='black', linestyle='--', lw=0.6, label='Median')
eier_handle = plt.Line2D(
    [0, 1],
    [0, 0],
    color='white',
    linestyle='-',
    linewidth=0.4,
    label='EIER',
)

iqr_patch = plt.Line2D(
    [0],
    [0],
    color='black',
    linestyle='-',
    linewidth=1.5,
    label='2.5th-97.5th percentile',
)

handles2 = [iqr_patch, mean_handle, median_handle]

fig.legend(
    handles=handles2,
    ncol=len(handles2),
    bbox_to_anchor=(0.57, 0.01),
    columnspacing=0.9,
    **common_params,
    numpoints=2,
    handler_map={
        mean_handle: HandlerVerticalLine(),
        median_handle: HandlerVerticalLine(),
    }
)

fig.text(0.55, 0.12, "Number of acquired samples", ha='center', fontsize=font_size)

plt.subplots_adjust(
    left=0.2,
    right=0.95,
    top=0.94,
    bottom=0.22,
    hspace=0.51,
    wspace=0.25,
)
finalize_figure(fig, "F06")
flush_report_summary(SUMMARY_TXT_PATH)
