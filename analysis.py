"""
For every model, quantizer and bit-width b it reports:
  delta_cf   cross-fitted transfer regret (F1 points): choose the FP32 winner and the bit-b winner on
             the dev split using some seeds (S), score both at bit b on the test split with the other
             seeds (E), average over all seed splits. Choosing and scoring on different seeds and
             different data removes the winner's-curse bias of comparing best against best.
  delta_naive  the biased version (choose and score on the same test means), for comparison.
  r_obs, r_corr  Pearson correlation of the configurations' mean F1 in FP32 vs at bit b, raw and
             corrected for seed noise with Spearman's (1904) correction (split-half reliability).
  tau        Kendall tau between the two rankings (raw, descriptive).
  var_ratio  median over configurations of seed variance at b / seed variance in FP32.
  CI90, CI95 two-level bootstrap: resample passages (paired across all runs) and seeds.
  decision   TOST-style rule with margin delta_min: "equivalent" if CI90 lies inside (-d, d);
             "shifted" if delta_cf > d and CI90 lower bound > 0; otherwise "inconclusive".

    python analysis.py --scores <out_dir>/scores --out <out_dir>/analysis
"""
import argparse
import glob
import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats


def load_scores(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv.gz")))
    if not files:
        raise SystemExit(f"no score files in {folder}")
    df = pd.concat([pd.read_csv(f, dtype={"bits": str, "qid": str, "pid": str}) for f in files])
    df["config"] = "lr=" + df.lr.map("{:g}".format) + ",wd=" + df.wd.map("{:g}".format)
    return df


def build_arrays(d, levels, seeds, configs):
    """Per-passage F1 sums for one model x quantizer: SUM[split] has shape (C, K, L, P)."""
    out = {}
    for split in ("dev", "test"):
        s = d[d.split == split]
        pids = sorted(s.pid.unique())
        pidx = {p: i for i, p in enumerate(pids)}
        SUM = np.zeros((len(configs), len(seeds), len(levels), len(pids)))
        CNT = np.zeros(len(pids))
        ref = s[(s.config == configs[0]) & (s.seed == seeds[0]) & (s.level == levels[0])]
        for p, n in ref.groupby("pid").size().items():
            CNT[pidx[p]] = n
        g = s.groupby(["config", "seed", "level", "pid"]).f1.sum()
        for (c, k, l, p), v in g.items():
            SUM[configs.index(c), seeds.index(k), levels.index(l), pidx[p]] = v
        out[split] = (SUM, CNT)
    return out


def means(arr, w=None):
    SUM, CNT = arr
    if w is None:
        return SUM.sum(-1) / CNT.sum()
    return (SUM @ w) / (CNT @ w)


def eval_size(K):
    return max(1, K // 2)


def crossfit(dev, test, lvl):
    """dev, test: (C, K, L) mean F1. Level 0 is FP32."""
    K = dev.shape[1]
    vals = []
    for S in itertools.combinations(range(K), K - eval_size(K)):
        E = [k for k in range(K) if k not in S]
        c_fp = int(np.argmax(dev[:, list(S), 0].mean(1)))
        c_b = int(np.argmax(dev[:, list(S), lvl].mean(1)))
        vals.append(test[c_b, E, lvl].mean() - test[c_fp, E, lvl].mean())
    return float(np.mean(vals))


def naive(test, lvl):
    m = test.mean(1)
    return float(m[:, lvl].max() - m[int(np.argmax(m[:, 0])), lvl])


def split_half_reliability(test, lvl):
    """Average split-half correlation of configuration means across seeds, Spearman-Brown stepped up."""
    K = test.shape[1]
    rs = []
    for A in itertools.combinations(range(K), K // 2):
        B = [k for k in range(K) if k not in A]
        a, b = test[:, list(A), lvl].mean(1), test[:, B, lvl].mean(1)
        if a.std() > 0 and b.std() > 0:
            rs.append(np.corrcoef(a, b)[0, 1])
    if not rs:
        return float("nan")
    r = float(np.mean(rs))
    return 2 * r / (1 + r) if r > -1 else float("nan")


def summarize(dev, test, lvl):
    a, b = test.mean(1)[:, 0], test.mean(1)[:, lvl]
    r_obs = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    rel0, relb = split_half_reliability(test, 0), split_half_reliability(test, lvl)
    r_corr = r_obs / math.sqrt(rel0 * relb) if rel0 > 0 and relb > 0 else float("nan")
    tau = float(stats.kendalltau(a, b).statistic) if len(a) > 1 else float("nan")
    v0, vb = test[:, :, 0].var(1, ddof=1), test[:, :, lvl].var(1, ddof=1)
    var_ratio = float(np.median(vb / np.where(v0 > 0, v0, np.nan)))
    return {"delta_cf": crossfit(dev, test, lvl), "delta_naive": naive(test, lvl),
            "f1_fp32_best": float(a.max()), "f1_b_mean": float(b.mean()),
            "r_obs": r_obs, "rel_fp32": rel0, "rel_b": relb, "r_corr": r_corr,
            "tau": tau, "var_ratio": var_ratio}


def bootstrap(arrs, lvl, B, rng):
    """Two-level bootstrap of delta_cf: passages resampled jointly for all runs, then seeds."""
    K = arrs["dev"][0].shape[1]
    out = np.empty(B)
    for i in range(B):
        w_dev = rng.multinomial(arrs["dev"][1].size, np.full(arrs["dev"][1].size, 1 / arrs["dev"][1].size))
        w_test = rng.multinomial(arrs["test"][1].size, np.full(arrs["test"][1].size, 1 / arrs["test"][1].size))
        dev, test = means(arrs["dev"], w_dev), means(arrs["test"], w_test)
        ks = rng.integers(0, K, K)
        out[i] = crossfit(dev[:, ks], test[:, ks], lvl)
    return out


def decide(est, lo90, hi90, d):
    if lo90 > -d and hi90 < d:
        return "equivalent"
    if est > d and lo90 > 0:
        return "shifted"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta-min", type=float, default=1.0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = load_scores(args.scores)
    rows, mean_rows = [], []
    for model in sorted(df.model.unique()):
        dm = df[df.model == model]
        # keep only seeds present for every configuration (complete design)
        cfg_seeds = dm[dm.quant == "none"].groupby("config").seed.apply(set)
        seeds = sorted(set.intersection(*cfg_seeds))
        configs = sorted(cfg_seeds.index)
        dm = dm[dm.seed.isin(seeds)]
        for quant in sorted(q for q in dm.quant.unique() if q != "none"):
            d = dm[dm.quant.isin(["none", quant])].copy()
            d["level"] = np.where(d.quant == "none", "32", d.bits)
            levels = ["32"] + sorted((l for l in d.level.unique() if l != "32"),
                                     key=lambda s: (-int(s.split("g")[0]), s))
            arrs = build_arrays(d, levels, seeds, configs)
            dev, test = means(arrs["dev"]), means(arrs["test"])
            for ci, c in enumerate(configs):
                for k, s in enumerate(seeds):
                    for li, l in enumerate(levels):
                        mean_rows.append({"model": model, "quant": quant, "bits": l, "config": c, "seed": s,
                                          "f1_dev": dev[ci, k, li], "f1_test": test[ci, k, li]})
            for li, l in enumerate(levels[1:], start=1):
                res = summarize(dev, test, li)
                bs = bootstrap(arrs, li, args.boot, rng)
                lo90, hi90 = np.percentile(bs, [5, 95])
                lo95, hi95 = np.percentile(bs, [2.5, 97.5])
                se = float(bs.std(ddof=1))
                k_needed = math.ceil(len(seeds) * (1.645 * se / (args.delta_min / 2)) ** 2) if se > 0 else len(seeds)
                rows.append({"model": model, "quant": quant, "bits": l, "K_seeds": len(seeds),
                             "n_configs": len(configs), **res, "ci90_lo": lo90, "ci90_hi": hi90,
                             "ci95_lo": lo95, "ci95_hi": hi95, "boot_se": se,
                             "seeds_for_halfwidth_d/2": k_needed,
                             "decision": decide(res["delta_cf"], lo90, hi90, args.delta_min)})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.DataFrame(mean_rows).to_csv(os.path.join(args.out, "config_means.csv"), index=False)
    with open(os.path.join(args.out, "settings.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    plot(summary, args.delta_min, os.path.join(args.out, "fig_delta_cf.png"))
    cols = ["model", "quant", "bits", "K_seeds", "delta_cf", "ci90_lo", "ci90_hi", "delta_naive", "r_corr",
            "var_ratio", "decision"]
    print(summary[cols].round(2).to_string(index=False))


def plot(summary, d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(summary.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        s = summary[summary.model == m]
        levels = list(dict.fromkeys(s.bits))
        x = np.arange(len(levels))
        ax.axhspan(-d, d, color="0.9", zorder=0)
        ax.axhline(0, color="0.5", lw=0.8)
        for j, q in enumerate(sorted(s.quant.unique())):
            sq = s[s.quant == q].set_index("bits").reindex(levels)
            off = (j - 0.5) * 0.18
            ax.errorbar(x + off, sq.delta_cf, yerr=[sq.delta_cf - sq.ci90_lo, sq.ci90_hi - sq.delta_cf],
                        fmt="o", capsize=3, label=f"{q.upper()} cross-fitted")
            ax.scatter(x + off, sq.delta_naive, marker="x", color="0.3", zorder=3,
                       label="naive (biased)" if j == 0 else None)
        ax.set_xticks(x, ["INT" + l.replace("g", " g") for l in levels])
        ax.set_title(m)
        ax.set_xlabel("weight bit-width")
    axes[0][0].set_ylabel("transfer regret Δ (F1 points)")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
