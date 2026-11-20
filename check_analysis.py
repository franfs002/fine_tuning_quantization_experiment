"""Check the estimators on simulated scores where the true answer is known.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MU32 = np.array([78.0, 79.0, 80.0, 79.5, 77.0, 78.5])  # true FP32 F1 of 6 configurations


def simulate(folder, mu4, seeds=5, passages=300, q_per_p=3, rng=None):
    configs = [(lr, wd) for lr in (1e-5, 3e-5, 5e-5) for wd in (0.0, 0.1)]
    diff = {sp: rng.normal(0, 1.2, passages) for sp in ("dev", "test")}   # passage difficulty (logit)
    for ci, (lr, wd) in enumerate(configs):
        for s in range(1, seeds + 1):
            seed_eff = rng.normal(0, 0.8)                                  # same run, all precisions
            rows = []
            for quant, bits, mu in (("none", "32", MU32), ("rtn", "8", MU32 - 0.2), ("rtn", "4", mu4)):
                m = mu[ci] + seed_eff + (rng.normal(0, 0.3) if bits != "32" else 0)
                base = np.log(m / (100 - m))
                for sp in ("dev", "test"):
                    p = 1 / (1 + np.exp(-(base + diff[sp])))
                    for pi in range(passages):
                        hits = rng.random(q_per_p) < p[pi]
                        for qi, h in enumerate(hits):
                            rows.append((lr, wd, s, quant, bits, sp, f"{sp}{pi}_{qi}", f"{sp}p{pi}", 100.0 * h, 100.0 * h))
            df = pd.DataFrame(rows, columns=["lr", "wd", "seed", "quant", "bits", "split", "qid", "pid", "f1", "em"])
            df.insert(0, "model", "sim")
            df.to_csv(os.path.join(folder, f"sim__lr{lr:g}__wd{wd:g}__s{s}.csv.gz"), index=False)


def main():
    rng = np.random.default_rng(7)
    for name, mu4, truth in (("A (no shift)", MU32 - 3.0, 0.0),
                             ("B (planted shift)", MU32 - np.array([3, 3, 6, 1.5, 3, 3]), 4.5)):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "scores"))
            simulate(os.path.join(tmp, "scores"), mu4, rng=rng)
            subprocess.run([sys.executable, os.path.join(HERE, "analysis.py"), "--scores", os.path.join(tmp, "scores"),
                            "--out", os.path.join(tmp, "a"), "--boot", "500"], check=True, stdout=subprocess.DEVNULL)
            s = pd.read_csv(os.path.join(tmp, "a", "summary.csv"), dtype={"bits": str})
            r = s[s.bits == "4"].iloc[0]
            print(f"{name}: true regret {truth:.1f} | cross-fitted {r.delta_cf:.2f} "
                  f"[{r.ci90_lo:.2f}, {r.ci90_hi:.2f}] | naive {r.delta_naive:.2f} | decision: {r.decision}")


if __name__ == "__main__":
    main()
