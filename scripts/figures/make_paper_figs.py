"""Draw the paper's bar figures from the runs of scripts/bash/run_all_methods.sh.
Reads only the runs (--runs) and data/query_tiers.csv; writes only to --out.

Figures (one PNG each, under --out):
  main_solverate_90s     solve rate within 90 s: GE, GE-PS, CFLGE, PBGE, RL   (fig:main-solverate)
  main_time_90s          mean time to first solution, 90 s cap, same methods
  ge_variants_solverate  solve rate, 300 s: GE, GE-RS, GE-PS, GE-RL            (fig:ge-variants)
  ge_variants_steps      mean Go-steps to solve (unsolved = 41)
  cflge_variants_solverate / cflge_variants_steps: GE, CFLGE, CFLGE-Frozen, CFLGE-Finetune
  rl_variants_solverate / rl_variants_time: GE, RL, RL-FT (RL-FT scored at 90 s)
"CFLGE" is CFLGE-Scratch, as in the paper.

Scoring, per query (5 seeds):
  solve rate   solved within 300 s (RL-FT: 90 s)
  Go-steps     the Go-step of the solve, 41 when unsolved
  time         solve time capped at 90 s, 90 when unsolved
  90 s rate    solved within 90 s
A run that crashed (log with no result line and no [TIMEOUT]) counts as unsolved.
Tiers follow the paper: the 56 queries sorted by the paper's GE mean time,
18 Short / 19 Medium / 19 Long (data/query_tiers.csv). Bars are tier
means; whiskers are the interquartile range of the per-query values.

Usage:
  python scripts/figures/make_paper_figs.py --runs results/runs/all --out results/figs/all
Also writes tier_means.csv and per_query.csv next to the figures.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TIERS = ("Short", "Medium", "Long")
TIER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "query_tiers.csv")


def load_tiers(path):
    """{query: tier} from data/query_tiers.csv."""
    return {r["query"]: r["tier"] for r in csv.DictReader(open(path))}

BUDGET_S, CAP_S, UNSOLVED_STEPS = 300, 90, 41
RESULT = re.compile(r"time=([\d.]+)s solution_found=(\w+) bins_discovered=\d+ solve_step=(\S+)")

# label -> (variant dir under --runs, layout)
METHODS = {
    "GE":             ("goexplore_vanilla", "gess"),
    "GE-RS":          ("goexplore_random_selection", "gess"),
    "GE-PS":          ("goexplore_policy_selection", "gess"),
    "GE-RL":          ("goexplore_only_rl_policy", "gess"),
    "CFLGE":          ("cflge_scratch", "gess"),
    "CFLGE-Frozen":   ("cflge_frozen", "gess"),
    "CFLGE-Finetune": ("cflge_finetune", "gess"),
    "PBGE":           ("goexplore_pbge", "gess"),
    "RL":             ("rl_inference", "rl"),
    "RL-FT":          ("rl_finetune", "rl_ft"),
}
COLORS = {"GE": "#a3b6dc", "GE-RS": "#7a9cc6", "GE-PS": "#4470b0", "GE-RL": "#2c4a7c",
          "CFLGE": "#dfa27e", "CFLGE-Frozen": "#c97b5b", "CFLGE-Finetune": "#9c5a40",
          "PBGE": "#2b3f75", "RL": "#a8c8a8", "RL-FT": "#5f9a6a"}


def seed_runs(d, layout):
    """{query: [(solved_300, solve_time_s or None, go_step or None), ...]} and # crashed runs."""
    out, crashed = {}, 0
    if layout == "gess":
        for log in glob.glob(os.path.join(d, "**", "seed*", "seed*.log"), recursive=True):
            q = os.path.relpath(os.path.dirname(os.path.dirname(log)), d)
            text = open(log, errors="ignore").read()
            m = RESULT.findall(text)
            if m:
                t, found, step = float(m[-1][0]), m[-1][1] == "True", m[-1][2]
                ok = found and t <= BUDGET_S
                out.setdefault(q, []).append((ok, t if ok else None, int(step) if ok else None))
            else:  # [TIMEOUT] or crash: unsolved
                crashed += "[TIMEOUT]" not in text
                out.setdefault(q, []).append((False, None, None))
    elif layout == "rl":
        p = os.path.join(d, "results.csv")
        for r in (csv.DictReader(open(p)) if os.path.exists(p) else []):
            walls = [int(w) for w in (r.get(f"solve{i}_wall", "") for i in range(1, 6)) if w]
            walls = sorted(w for w in walls if w <= BUDGET_S)
            out[r["query"]] = [(True, w, None) for w in walls] + [(False, None, None)] * (5 - len(walls))
    else:  # rl_ft: one result.json per seed
        for p in glob.glob(os.path.join(d, "**", "seed*", "result.json"), recursive=True):
            q = os.path.relpath(os.path.dirname(os.path.dirname(p)), d)
            try:
                r = json.load(open(p))
            except (ValueError, OSError):
                crashed += 1
                r = {}
            t = r.get("solve_time_sec")
            ok = bool(r.get("solved")) and t is not None and t <= CAP_S  # RL-FT is scored at 90 s
            out.setdefault(q, []).append((ok, t if ok else None, None))
    return out, crashed


def per_query(runs):
    """{query: dict(solve, solve90, steps, time)}"""
    res = {}
    for q, v in runs.items():
        res[q] = dict(
            solve=statistics.mean(ok for ok, _, _ in v),
            solve90=statistics.mean(ok and t <= CAP_S for ok, t, _ in v),
            steps=statistics.mean(s if ok and s is not None else UNSOLVED_STEPS for ok, _, s in v),
            time=statistics.mean(min(t, CAP_S) if ok else CAP_S for ok, t, _ in v))
    return res


def figure(path, title, labels, metric, data, tier, ylabel, ymax, fmt):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    n = len(labels)
    w = 0.8 / n
    rows = []
    for i, lab in enumerate(labels):
        for j, t in enumerate(TIERS):
            x = j + (i - (n - 1) / 2) * w
            vals = [data[lab][q][metric] for q in data.get(lab, {}) if tier.get(q) == t]
            if not vals:
                ax.text(x, ymax * 0.01, "n/a", ha="center", va="bottom", fontsize=6, rotation=90, color="#777")
                continue
            mu = statistics.mean(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            ax.bar(x, mu, w * 0.92, color=COLORS[lab], label=lab if j == 0 else None)
            ax.errorbar(x, mu, yerr=[[max(mu - q1, 0)], [max(q3 - mu, 0)]], fmt="none",
                        ecolor="#333", elinewidth=0.7, capsize=0)
            ax.text(x, mu + ymax * 0.015, fmt(mu), ha="center", va="bottom", fontsize=7, fontweight="bold")
            rows.append((os.path.basename(path)[:-4], lab, t, metric, round(mu, 4), len(vals)))
    ax.set_xticks(range(3), [f"{t}\n({sum(1 for q in tier if tier[q] == t)} queries)" for t in TIERS])
    ax.set_ylim(0, ymax * 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=n, frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="the OUT dir of run_all_methods.sh")
    ap.add_argument("--out", required=True, help="figure dir, e.g. results/figs/all")
    ap.add_argument("--tiers", default=TIER_FILE, help="query -> tier table")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    tier = load_tiers(args.tiers)

    data = {}
    for lab, (sub, layout) in METHODS.items():
        d = os.path.join(args.runs, sub)
        if not os.path.isdir(d):
            print(f"{lab:15s} missing ({d}): drawn as n/a")
            continue
        runs, crashed = seed_runs(d, layout)
        runs = {q: v for q, v in runs.items() if q in tier}
        short = [q for q, v in runs.items() if len(v) != 5]
        print(f"{lab:15s} {len(runs):2d} queries, {crashed} crashed runs (unsolved)"
              + (f", WARNING {len(short)} queries without 5 seeds" if short else ""))
        data[lab] = per_query(runs)

    f2, f1 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.1f}")
    main5 = ["GE", "GE-PS", "CFLGE", "PBGE", "RL"]
    ge4 = ["GE", "GE-RS", "GE-PS", "GE-RL"]
    cf4 = ["GE", "CFLGE", "CFLGE-Frozen", "CFLGE-Finetune"]
    rl3 = ["GE", "RL", "RL-FT"]
    figs = [
        ("main_solverate_90s", "Solve rate (90s budget)", main5, "solve90", "Solve rate", 1.0, f2),
        ("main_time_90s", "Mean time to first solution (90s cap)", main5, "time", "Time (s)", 90, f1),
        ("ge_variants_solverate", "Go-Explore variants: solve rate", ge4, "solve", "Solve rate", 1.0, f2),
        ("ge_variants_steps", "Go-Explore variants: Go-steps to solve", ge4, "steps", "Go-steps", 41, f1),
        ("cflge_variants_solverate", "CFLGE variants: solve rate", cf4, "solve", "Solve rate", 1.0, f2),
        ("cflge_variants_steps", "CFLGE variants: Go-steps to solve", cf4, "steps", "Go-steps", 41, f1),
        ("rl_variants_solverate", "RL variants: solve rate", rl3, "solve", "Solve rate", 1.0, f2),
        ("rl_variants_time", "RL variants: time to solve (90s cap)", rl3, "time", "Time (s)", 90, f1),
    ]
    rows = []
    for name, title, labels, metric, ylabel, ymax, fmt in figs:
        rows += figure(os.path.join(out, name + ".png"), title, labels, metric, data, tier, ylabel, ymax, fmt)
    with open(os.path.join(out, "tier_means.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("figure", "method", "tier", "metric", "mean", "n_queries")] + rows)
    with open(os.path.join(out, "per_query.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("method", "query", "tier", "solve", "solve90", "steps", "time"))
        for lab, d in data.items():
            for q in sorted(d, key=lambda q: (TIERS.index(tier[q]), q)):
                w.writerow((lab, q, tier[q]) + tuple(round(d[q][k], 4) for k in ("solve", "solve90", "steps", "time")))
    print(f"wrote {len(figs)} figures, tier_means.csv and per_query.csv to {out}")


if __name__ == "__main__":
    main()
