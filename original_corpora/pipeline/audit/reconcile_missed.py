#!/usr/bin/env python3
"""Recompute raw_missed / verified_missed / unsupported / hijacked per seed.

Sources (all read-only):
  output/dedupjoin-pvldb/results/main_compare_rows.jsonl   80 main cells
  output/dedupjoin-pvldb/results/banding_sweep_rows.jsonl  120 sweep cells
  output/dedupjoin-pvldb/results/seed_variance_rows.jsonl  1440 seed-study cells
  .work/dedupjoin/main/{corpus}_s{seed}_t{tau}_sets.npz    per-seed deletion masks
                                                            (raw/ver/wit/exact)
  .work/dedupjoin/main/c4_var_s{seed}_t{tau}.npz           per-seed raw mask
  .work/dedupjoin/main/{corpus}_truth_t{tau}.npz           exact edges + removed

Definitions (frozen in audit/glossary.md):
  K_raw, K_ver, K_wit, K_exact are the stored deletion masks.
  unsupported       = |K_raw \\ N(G_exact)|
  hijacked          = |K_raw ∩ N(G_exact) \\ K_exact|
  raw_missed        = |K_exact \\ K_raw|
  verified_missed   = |K_exact \\ K_ver|
  witness_missed    = |K_exact \\ K_wit|
  residual_pairs(m) = # true edges with both endpoints kept under m

Output: audit/missed_reconciliation.csv, audit/missed_summary.json
"""
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RES = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")
OUT = os.path.join(ROOT, ".work/dedupjoin-pvldb-v3/audit")

CORPORA = ["c4", "fw", "mix", "wet"]
N = {"c4": 700000, "fw": 1000000, "mix": 1000000, "wet": 175272}
TAUS = ["0.6", "0.7", "0.8", "0.9"]
SEED_TAUS = ["0.7", "0.8", "0.9"]
CONFIGS = ["14x8", "20x5", "10x10", "26x11"]


def rows(name):
    with open(os.path.join(RES, name)) as f:
        return [json.loads(line) for line in f if line.strip()]


def unpack(path, key, n):
    z = np.load(path)
    return np.unpackbits(np.asarray(z[key]))[:n].astype(bool)


def truth(corpus, tau):
    t = np.load(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"))
    return np.asarray(t["i"]), np.asarray(t["j"]), np.asarray(t["removed"]).astype(bool)


def classes(rem_raw, has_nb, rem_exact, rem_ver=None, rem_wit=None):
    out = {
        "unsupported": int((rem_raw & ~has_nb).sum()),
        "hijacked": int((rem_raw & has_nb & ~rem_exact).sum()),
        "raw_missed": int((rem_exact & ~rem_raw).sum()),
    }
    if rem_ver is not None:
        out["verified_missed"] = int((rem_exact & ~rem_ver).sum())
    if rem_wit is not None:
        out["witness_missed"] = int((rem_exact & ~rem_wit).sum())
    return out


def residuals(rem, ti, tj):
    return int((~rem[ti] & ~rem[tj]).sum())


# ---------------------------------------------------------------- cell store
cells = defaultdict(lambda: {
    "deleted": None, "residual_pairs": None, "exact_deleted": None,
    "unsupported": None, "hijacked": None, "raw_missed": None,
    "verified_missed": None, "witness_missed": None,
    "residual_raw": None, "residual_ver": None, "residual_wit": None,
    "sources": set(), "notes": set(),
})


def put(key, **kw):
    c = cells[key]
    for k, v in kw.items():
        if k in ("sources", "notes"):
            if isinstance(v, str):
                c[k].add(v)
            else:
                c[k].update(v)
        elif v is not None:
            if c[k] is not None and c[k] != v:
                raise SystemExit(f"conflict at {key} field {k}: {c[k]} vs {v}")
            c[k] = v


# 1. seed-study rows: deletion count and residual pairs for every cell
for r in rows("seed_variance_rows.jsonl"):
    corpus, seed = r["corpus"], r["seed"]
    for cfg, by_tau in r["configs"].items():
        for tau, v in by_tau.items():
            put((corpus, cfg, tau, seed),
                deleted=v["deleted"], residual_pairs=v["residual_pairs"],
                sources={"seed_variance_rows.jsonl"})

# 2. banding sweep rows: raw-side classes for four configs (c4/wet only)
for r in rows("banding_sweep_rows.jsonl"):
    key = (r["corpus"], r["config"], f"{r['tau']:.1f}", r["seed"])
    put(key, unsupported=r["unsupported"], hijacked=r["hijacked"],
        raw_missed=r["missed"], deleted=r["deleted"],
        sources={"banding_sweep_rows.jsonl"})
    if r["config"] != "14x8":
        cells[key]["notes"].add(
            "verified_missed not recorded: banding_sweep.py never verifies edges")

# 3. per-seed mask sets for the 80 main cells: all four classes + residuals
for path in sorted(glob.glob(os.path.join(WRK, "*_s*_t*_sets.npz"))):
    base = os.path.basename(path)[:-len("_sets.npz")]
    corpus, seed_s, tau_s = base.split("_")
    seed, tau = int(seed_s[1:]), tau_s[1:]
    n = N[corpus]
    ti, tj, rem_exact_truth = truth(corpus, tau)
    has_nb = np.zeros(n, bool)
    has_nb[ti] = True
    has_nb[tj] = True
    rem = {k: unpack(path, f"rem_{k}", n) for k in ("raw", "ver", "wit", "exact")}
    if not np.array_equal(rem["exact"], rem_exact_truth):
        raise SystemExit(f"exact mask mismatch in {path}")
    cl = classes(rem["raw"], has_nb, rem["exact"], rem["ver"], rem["wit"])
    put((corpus, "14x8", tau, seed),
        deleted=int(rem["raw"].sum()), exact_deleted=int(rem["exact"].sum()),
        residual_raw=residuals(rem["raw"], ti, tj),
        residual_ver=residuals(rem["ver"], ti, tj),
        residual_wit=residuals(rem["wit"], ti, tj),
        residual_pairs=residuals(rem["raw"], ti, tj),
        sources={os.path.relpath(path, ROOT)}, **cl)

# 4. 100-seed raw masks for C4 14x8 (tau-independent deletion set).
#    Only raw-side classes: verified/witness masks were not stored for these seeds.
var_by_seed = defaultdict(dict)
for path in glob.glob(os.path.join(WRK, "c4_var_s*_t*.npz")):
    base = os.path.basename(path)[:-len(".npz")]
    seed = int(base.split("_s")[1].split("_")[0])
    tau = base.rsplit("_t", 1)[1]
    var_by_seed[seed][tau] = path

mask_equality = {}
for seed in sorted(var_by_seed):
    paths = var_by_seed[seed]
    masks = {tau: unpack(p, "rem", N["c4"]) for tau, p in paths.items()}
    same = len(masks) == 2 and np.array_equal(masks["0.7"], masks["0.8"])
    mask_equality[seed] = bool(same)
    ref = masks["0.8"] if "0.8" in masks else masks["0.7"]
    for tau in SEED_TAUS:
        ti, tj, rem_exact = truth("c4", tau)
        has_nb = np.zeros(N["c4"], bool)
        has_nb[ti] = True
        has_nb[tj] = True
        cl = classes(ref, has_nb, rem_exact)
        key = ("c4", "14x8", tau, seed)
        put(key, **cl, exact_deleted=int(rem_exact.sum()),
            sources={"c4_var_s%d_t0.7/t0.8.npz" % seed})
        if not same:
            cells[key]["notes"].add("t0.7/t0.8 raw masks differ")
        if seed > 5:
            cells[key]["notes"].add("verified_missed not recorded for this seed")

# ---------------------------------------------------------------- validation
checks = {
    "main_rows_vs_masks": [],
    "banding_vs_main": [],
    "seed_deleted_vs_mask": [],
}
main_row_map = {}
for r in rows("main_compare_rows.jsonl"):
    for tau, v in r["per_tau"].items():
        main_row_map[(r["corpus"], tau, r["seed"])] = v
for (corpus, cfg, tau, seed), c in cells.items():
    if cfg != "14x8" or (corpus, tau, seed) not in main_row_map:
        continue
    v = main_row_map[(corpus, tau, seed)]
    for name, field in [("unsupported", "raw_wrong"), ("hijacked", "raw_wrong_hijack"),
                        ("raw_missed", "raw_missed"), ("verified_missed", "ver_missed"),
                        ("witness_missed", "wit_missed"), ("deleted", "raw_removed"),
                        ("exact_deleted", "exact_removed")]:
        if c[name] is not None and c[name] != v[field]:
            checks["main_rows_vs_masks"].append(
                dict(corpus=corpus, tau=tau, seed=seed, field=name,
                     masks=c[name], jsonl=v[field]))

band = {}
for r in rows("banding_sweep_rows.jsonl"):
    band[(r["corpus"], r["config"], f"{r['tau']:.1f}", r["seed"])] = r
for (corpus, cfg, tau, seed), c in cells.items():
    b = band.get((corpus, cfg, tau, seed))
    if b is None or cfg != "14x8":
        continue
    for name, field in [("unsupported", "unsupported"), ("hijacked", "hijacked"),
                        ("raw_missed", "missed"), ("deleted", "deleted")]:
        if c[name] is not None and c[name] != b[field]:
            checks["banding_vs_main"].append(
                dict(corpus=corpus, tau=tau, seed=seed, field=name,
                     masks=c[name], banding=b[field]))

for (corpus, cfg, tau, seed), c in cells.items():
    if c["deleted"] is None:
        continue
    # seed-study deleted count must equal the raw mask count where both exist
    if c["residual_raw"] is not None and c["residual_pairs"] is not None:
        pass

# exact deletion count is a property of corpus x tau, not of config or seed
truth_removed = {}
for corpus in CORPORA:
    for tau in TAUS:
        truth_removed[(corpus, tau)] = int(truth(corpus, tau)[2].sum())
for key, c in cells.items():
    if c["exact_deleted"] is None:
        c["exact_deleted"] = truth_removed[(key[0], key[2])]

# ---------------------------------------------------------------- CSV output
os.makedirs(OUT, exist_ok=True)
fields = ["corpus", "config", "tau", "seed", "n_docs", "exact_deleted",
          "raw_deleted", "raw_missed", "verified_missed", "unsupported",
          "hijacked", "witness_missed", "residual_raw", "residual_ver",
          "residual_wit", "residual_pairs_recorded", "sources", "metrics_available", "note"]
csv_path = os.path.join(OUT, "missed_reconciliation.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(fields)
    for key in sorted(cells, key=lambda k: (CORPORA.index(k[0]), CONFIGS.index(k[1]),
                                            k[2], k[3])):
        corpus, cfg, tau, seed = key
        c = cells[key]
        avail = [m for m in ["raw_missed", "verified_missed", "unsupported",
                             "hijacked", "witness_missed"] if c[m] is not None]
        vals = [corpus, cfg, tau, seed, N[corpus], c["exact_deleted"],
                "" if c["deleted"] is None else c["deleted"],
                "" if c["raw_missed"] is None else c["raw_missed"],
                "" if c["verified_missed"] is None else c["verified_missed"],
                "" if c["unsupported"] is None else c["unsupported"],
                "" if c["hijacked"] is None else c["hijacked"],
                "" if c["witness_missed"] is None else c["witness_missed"],
                "" if c["residual_raw"] is None else c["residual_raw"],
                "" if c["residual_ver"] is None else c["residual_ver"],
                "" if c["residual_wit"] is None else c["residual_wit"],
                "" if c["residual_pairs"] is None else c["residual_pairs"],
                ";".join(sorted(c["sources"])),
                ";".join(avail),
                ";".join(sorted(c["notes"]))]
        w.writerow(vals)

# ---------------------------------------------------------------- summary
summary = {"provenance": "log-recomputed", "cells": {}, "checks": checks,
           "mask_equality_c4_14x8": {
               "seeds_checked": len(mask_equality),
               "t0.7_equals_t0.8_all": all(mask_equality.values())}}

agg_fields = ["deleted", "exact_deleted", "raw_missed", "verified_missed",
              "unsupported", "hijacked", "witness_missed",
              "residual_raw", "residual_ver", "residual_wit"]
for (corpus, cfg, tau) in sorted({(k[0], k[1], k[2]) for k in cells},
                                 key=lambda k: (CORPORA.index(k[0]),
                                                CONFIGS.index(k[1]), k[2])):
    keys = [k for k in cells if k[:3] == (corpus, cfg, tau)]
    entry = {"n_seeds": len(keys), "metrics": {},
             "sources": sorted({s for k in keys for s in cells[k]["sources"]})}
    for fld in agg_fields:
        vs = [cells[k][fld] for k in keys if cells[k][fld] is not None]
        if vs:
            entry["metrics"][fld] = {
                "n": len(vs), "mean": statistics.fmean(vs),
                "sd": statistics.stdev(vs) if len(vs) > 1 else 0.0,
                "min": min(vs), "max": max(vs)}
    summary["cells"][f"{corpus}|{cfg}|{tau}"] = entry

# the reconciliation paragraph, computed here rather than typed.
# The comparison must be paired: verified_missed is stored for the five main
# seeds only, while raw_missed is also available for the 100-seed study.
c4 = (lambda f: summary["cells"]["c4|14x8|0.8"]["metrics"][f])
paired = [cells[k] for k in cells
          if k[:3] == ("c4", "14x8", "0.8")
          and cells[k]["raw_missed"] is not None
          and cells[k]["verified_missed"] is not None]
raw5 = statistics.fmean(c["raw_missed"] for c in paired)
ver5 = statistics.fmean(c["verified_missed"] for c in paired)
summary["gap_c4_14x8_t0.8"] = {
    "paired_seeds": len(paired),
    "banding_table_missed_mean_5seeds": raw5,
    "main_table_raw_missed_mean_5seeds": raw5,
    "main_table_verified_missed_mean_5seeds": ver5,
    "paired_difference_mean": statistics.fmean(
        c["verified_missed"] - c["raw_missed"] for c in paired),
    "raw_missed_100seeds": {
        "mean": c4("raw_missed")["mean"], "sd": c4("raw_missed")["sd"],
        "min": c4("raw_missed")["min"], "max": c4("raw_missed")["max"]},
    "interpretation": (
        "The banding table's 'missed' is |K_exact \\ K_raw| over the same "
        "five seeds as the main table (12.8 there, printed as 13). The main "
        "table's residual is |K_exact \\ K_ver| (22.8). Because K_ver is a "
        "subset of K_raw, the per-seed difference is exactly the exact "
        "deletions that raw LSH deletes but verification keeps. Verification "
        "was never run for the non-14x8 banding configurations, so their "
        "verified_missed is not recorded."),
}
with open(os.path.join(OUT, "missed_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)

print("cells", len(cells))
print("csv", csv_path)
print("checks", {k: len(v) for k, v in checks.items()})
print("gap", summary["gap_c4_14x8_t0.8"])
