#!/usr/bin/env python3
"""Document-level banding sweep.

Recomputes, for four banding configurations and three exact thresholds, the raw LSH
deleted set and classifies every deleted document against the exact truth of the same
corpus: unsupported (no neighbour at J>=tau), hijacked (has a neighbour but is the
representative its exact component keeps), and correct (deleted by both). Also records
missed exact deletions, pair recall and the banding design threshold.

Outputs:
  results/banding_sweep_rows.jsonl   one row per (corpus, seed, config, tau)
  results/banding_sweep.json         aggregate (mean over seeds, per cell)

Usage:
  python3 banding_sweep.py --corpora c4,fw,wet,mix --seeds 1-5
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
C03 = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, C03)
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import lsh as L  # noqa: E402
import main_compare as MC  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")
ROWS = os.path.join(OUT, "banding_sweep_rows.jsonl")
CONFIGS = [(14, 8), (20, 5), (10, 10), (26, 11)]
TAUS = [0.7, 0.8, 0.9]
H_MAIN = 112
H_2611 = 286


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def append_row(row):
    row.setdefault("ts_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(ROWS, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def design_threshold(b, r):
    """Similarity at which the S-curve reaches 0.5."""
    return float((1.0 - 0.5 ** (1.0 / b)) ** (1.0 / r))


def run(corpus, seeds, threads):
    n = MC.CORPORA[corpus]
    C.set_threads(threads)
    off, flat, nw, meta = MC.load_shingles(corpus)
    valid = np.diff(off) > 0
    truth = {}
    for tau in TAUS:
        q = np.load(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"))
        ti, tj = np.asarray(q["i"]), np.asarray(q["j"])
        rem = np.asarray(q["removed"]).astype(bool)
        deg = np.zeros(n, np.int64)
        np.add.at(deg, ti, 1)
        np.add.at(deg, tj, 1)
        truth[tau] = dict(ti=ti, tj=tj, rem=rem, deg=(deg > 0), n_pairs=len(ti))
        log(corpus, "truth", tau, "pairs", len(ti), "removed", int(rem.sum()))
    for seed in seeds:
        t0 = time.time()
        sig = L.signatures(off, flat, H_MAIN, seed, "dt")
        t_sig = time.time() - t0
        log(corpus, "seed", seed, "sig %.1fs" % t_sig)
        for (b, r) in CONFIGS:
            t1 = time.time()
            if (b, r) == (26, 11):
                sig_use = L.signatures(off, flat, H_2611, seed, "dt")
                sdraw = "286-permutation draw, seed %d (independent of the 112 draw)" % seed
            else:
                sig_use = sig
                sdraw = "first 112 permutations of the seed-%d draw" % seed
            K = L.band_keys(sig_use, b, r)
            codes, ci, cj, tot = L.lsh_pairs(K, valid, 400_000_000)
            del K
            root = L.uf_components(n, ci, cj)
            rem_lsh = L.removed_set(root) & valid
            del codes, root
            t_band = time.time() - t1
            for tau in TAUS:
                T = truth[tau]
                hit = L.truth_hits(sig_use, T["ti"], T["tj"], b, r).astype(np.float64)
                unsupported = rem_lsh & ~T["deg"]
                hijacked = rem_lsh & T["deg"] & ~T["rem"]
                both = rem_lsh & T["rem"]
                missed = T["rem"] & ~rem_lsh
                row = dict(
                    kind="banding_sweep", corpus=corpus, seed=seed, config=f"{b}x{r}",
                    tau=tau, bands=b, rows=r, h=b * r,
                    sig_draw=sdraw,
                    design_threshold=round(design_threshold(b, r), 4),
                    exact_pairs=int(T["n_pairs"]),
                    exact_removed=int(T["rem"].sum()),
                    deleted=int(rem_lsh.sum()),
                    unsupported=int(unsupported.sum()),
                    hijacked=int(hijacked.sum()),
                    both_deleted=int(both.sum()),
                    missed=int(missed.sum()),
                    unsupported_frac=float(unsupported.sum() / max(1, rem_lsh.sum())),
                    unsupported_frac_input=float(unsupported.sum() / n),
                    missed_frac_exact=float(missed.sum() / max(1, T["rem"].sum())),
                    pair_recall=float(hit.mean()) if len(hit) else None,
                    cand_pairs_total=int(tot),
                    sec_banding=round(t_band, 2),
                    sec_sig=round(t_sig, 2),
                    n=int(n),
                    provenance="independently rerun (this run)",
                )
                append_row(row)
                log(corpus, seed, f"{b}x{r}", tau, "del", row["deleted"],
                    "unsup", row["unsupported"], "%.3f" % row["unsupported_frac"],
                    "missed", row["missed"])
            del rem_lsh
        del sig


def aggregate():
    import collections
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["corpus"], r["config"], r["tau"])].append(r)
    out = {}
    for (corpus, cfg, tau), lst in sorted(by.items()):
        def mean(k):
            v = [m[k] for m in lst if m.get(k) is not None]
            return float(np.mean(v)) if v else None
        def sd(k):
            v = [m[k] for m in lst if m.get(k) is not None]
            return float(np.std(v, ddof=1)) if len(v) > 1 else None
        out.setdefault(corpus, {}).setdefault(cfg, {})[str(tau)] = dict(
            seeds=len(lst), design_threshold=lst[0]["design_threshold"],
            deleted_mean=mean("deleted"), deleted_sd=sd("deleted"),
            exact_removed=int(lst[0]["exact_removed"]),
            unsupported_mean=mean("unsupported"), unsupported_sd=sd("unsupported"),
            unsupported_frac_mean=mean("unsupported_frac"),
            unsupported_frac_sd=sd("unsupported_frac"),
            unsupported_frac_input_mean=mean("unsupported_frac_input"),
            hijacked_mean=mean("hijacked"),
            missed_mean=mean("missed"), missed_sd=sd("missed"),
            missed_frac_exact_mean=mean("missed_frac_exact"),
            pair_recall_mean=mean("pair_recall"), pair_recall_sd=sd("pair_recall"),
            sec_banding_mean=mean("sec_banding"),
        )
    with open(os.path.join(OUT, "banding_sweep.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("wrote banding_sweep.json", len(out), "corpora")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--corpora", default="c4")
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "run":
        x, y = a.seeds.split("-")
        seeds = list(range(int(x), int(y) + 1))
        for corpus in a.corpora.split(","):
            run(corpus, seeds, a.threads)
    else:
        aggregate()


if __name__ == "__main__":
    main()
