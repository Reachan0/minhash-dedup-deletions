#!/usr/bin/env python3
"""Build the main decision table and cross-method set metrics from result files.

Inputs (all on disk, written by earlier steps of this run):
  core/runs.jsonl               accuracy rows per (workload, tau, method, rep, seed)
  core/timing.jsonl             per-configuration timings from fresh OS processes
  core/bucket_scaling.jsonl     path-B bucket-expansion scaling (if present)
Outputs:
  core/main_decision_table.{csv,json,tex}
  core/representative_rules.jsonl   min-id vs datatrove representative, side by side
  core/set_metrics.json             raw/verified/witness missed and hijacked, per cell

Deletion masks are recomputed from the frozen shingles and signature caches (evidence
tier `rerun`), so the set relations between methods are exact rather than inferred from
counts.  Nothing here calls a model; all numbers come from earlier result files.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K

OUT = K.CORE
REPS = ("minid", "dt")
STAGE_ALGO = ("chain_edges", "union_find", "bucket_pairs", "verify", "greedy_mis",
              "exact_join")
STAGE_PREPROC = ("read_shingle", "ranks", "signature", "banding")


def load_rows(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _stage_sum(r, names):
    return float(sum(r.get(n + "_s", 0.0) for n in names))


def masks_and_metrics(w, tau, seed, off, flat, valid):
    n = len(off) - 1
    ti, tj, _ = K.compute_truth(w, tau, off)
    sig = np.load(K.sig_path(w, seed))
    Kb = K.band_keys(sig)
    del sig
    ei, ej, _ = K.chain_edges(Kb, valid)
    pi, pj, _ = K.all_bucket_pairs(Kb, valid, n)
    jac = K.L.jaccard_pairs(off, flat, pi, pj)
    ver = jac >= tau - 1e-12
    bei, bej = pi[ver], pj[ver]
    del pi, pj, jac
    masks = {}
    for rep, uf in (("minid", K.uf_minid), ("dt", K.uf_datatrove)):
        masks[("A", rep)] = (uf(n, ei, ej) != np.arange(n)) & valid
        masks[("B", rep)] = (uf(n, bei, bej) != np.arange(n)) & valid
        masks[("C", rep)] = (uf(n, ti, tj) != np.arange(n)) & valid
    kept = K.greedy_mis(n, bei, bej)
    masks[("W", "minid")] = (~kept) & valid
    del Kb, ei, ej, bei, bej

    hnb = K.has_true_neighbour(n, ti, tj)
    out = {}
    for rep in REPS:
        exact = masks[("C", rep)]
        raw = masks[("A", rep)]
        ver_m = masks[("B", rep)]
        wit = masks[("W", "minid")] if rep == "minid" else None
        m = dict(
            raw_missed=int((exact & ~raw).sum()),
            verified_missed=int((exact & ~ver_m).sum()),
            hijacked_raw=int((raw & hnb & ~exact).sum()),
            hijacked_verified=int((ver_m & hnb & ~exact).sum()),
            exact_deleted=int(exact.sum()),
            overlap_raw=int((exact & raw).sum()),
            overlap_verified=int((exact & ver_m).sum()),
        )
        if wit is not None:
            m["witness_missed"] = int((exact & ~wit).sum())
            m["overlap_witness"] = int((exact & wit).sum())
        out[rep] = m
    out["_deleted"] = {f"{k[0]}_{k[1]}": int(masks[k].sum()) for k in masks}
    out["_symmetric_diff"] = {
        f"A_{rep}_vs_B_{rep}": int((masks[("A", rep)] ^ masks[("B", rep)]).sum())
        for rep in REPS}
    out["_symmetric_diff"].update({
        f"B_{rep}_vs_C_{rep}": int((masks[("B", rep)] ^ masks[("C", rep)]).sum())
        for rep in REPS})
    out["_rep_symmetric_diff"] = {
        f"{m}_minid_vs_dt": int((masks[(m, "minid")] ^ masks[(m, "dt")]).sum())
        for m in ("A", "B", "C")}
    return out, ti, tj


def timing_cells(trows, w, tau):
    """Per method: measured cold (tau=0.8 only), incremental, preproc and RSS medians."""
    cell = {}
    for m in ("A", "B", "W", "C"):
        inc = [r for r in trows if r["workload"] == w and abs(r["tau"] - tau) < 1e-9
               and r["method"] == m and r["mode"] == "incremental"
               and r["threads"] == 12 and r.get("tag") == "main"]
        cold_method = "B" if m == "W" else m
        cold = [r for r in trows if r["workload"] == w and abs(r["tau"] - 0.8) < 1e-9
                and r["method"] == cold_method and r["mode"] == "cold"
                and r["threads"] == 12 and r.get("tag") == "main"]
        inc1 = [r for r in trows if r["workload"] == w and abs(r["tau"] - tau) < 1e-9
                and r["method"] == m and r["mode"] == "incremental"
                and r["threads"] == 1 and r.get("tag") == "main"]
        c = dict(incremental_n=len(inc), cold_n=len(cold), incremental_1thread_n=len(inc1))
        if inc:
            c["incremental_median_s"] = float(np.median([r["wall_s"] for r in inc]))
            c["incremental_min_s"] = float(min(r["wall_s"] for r in inc))
            c["incremental_max_s"] = float(max(r["wall_s"] for r in inc))
            c["incremental_peak_rss_mb"] = float(np.median([r["peak_rss_mb"] for r in inc]))
            c["incremental_algo_median_s"] = float(np.median(
                [_stage_sum(r, STAGE_ALGO) for r in inc]))
        if inc1:
            c["incremental_1thread_median_s"] = float(np.median(
                [r["wall_s"] for r in inc1]))
            c["incremental_1thread_algo_median_s"] = float(np.median(
                [_stage_sum(r, STAGE_ALGO) for r in inc1]))
        if cold:
            c["cold_peak_rss_mb"] = float(np.median([r["peak_rss_mb"] for r in cold]))
            c["shared_preproc_median_s"] = float(np.median(
                [_stage_sum(r, STAGE_PREPROC) for r in cold]))
            if abs(tau - 0.8) < 1e-9 and m != "W":
                c["cold_median_s"] = float(np.median([r["wall_s"] for r in cold]))
                c["cold_min_s"] = float(min(r["wall_s"] for r in cold))
                c["cold_max_s"] = float(max(r["wall_s"] for r in cold))
            else:
                pre = np.median([_stage_sum(r, STAGE_PREPROC) for r in cold])
                algo = (np.median([_stage_sum(r, STAGE_ALGO) for r in inc]) if inc else None)
                if algo is not None:
                    c["cold_composed_median_s"] = float(pre + algo)
                    c["cold_composition"] = ("cold preprocessing measured at tau=0.8 "
                                             "(tau-independent) + incremental algorithm "
                                             "at this tau")
        if m == "W" and cold and inc:
            pre = np.median([_stage_sum(r, STAGE_PREPROC) for r in cold])
            algo = np.median([_stage_sum(r, STAGE_ALGO) for r in inc])
            c["cold_composed_median_s"] = float(pre + algo)
            c["cold_composition"] = ("cold preprocessing of path B at tau=0.8 + "
                                     "incremental B-verification/MIS at this tau")
        cell[m] = c
    return cell


def main():
    runs = load_rows(K.RUNS)
    trows = load_rows(K.TIMING)
    set_metrics = {}
    rep_rules = []
    for w in K.WORKLOADS:
        off, flat, _ = K.load_shingles(w)
        n = len(off) - 1
        valid = np.diff(off) > 0
        for tau in K.TAUS:
            seeds = sorted({r["hash_seed"] for r in runs
                            if r["workload"] == w and abs(r["tau"] - tau) < 1e-9
                            and r["hash_seed"]})
            if not seeds:
                continue
            per = []
            for s in seeds:
                m, ti, tj = masks_and_metrics(w, tau, s, off, flat, valid)
                per.append(m)
                d = m["_deleted"]
                sd = m["_symmetric_diff"]
                rsd = m["_rep_symmetric_diff"]
                rep_rules.append(dict(
                    key=f"{w}|t{tau}|s{s}", workload=w, tau=tau, hash_seed=s, n=n,
                    deleted_A_minid=d.get("A_minid"), deleted_A_dt=d.get("A_dt"),
                    deleted_B_minid=d.get("B_minid"), deleted_B_dt=d.get("B_dt"),
                    deleted_C_minid=d.get("C_minid"), deleted_C_dt=d.get("C_dt"),
                    deleted_W_minid=d.get("W_minid"),
                    minid_vs_dt_symdiff_A=rsd["A_minid_vs_dt"],
                    minid_vs_dt_symdiff_B=rsd["B_minid_vs_dt"],
                    minid_vs_dt_symdiff_C=rsd["C_minid_vs_dt"],
                    A_minid_vs_B_minid=sd.get("A_minid_vs_B_minid"),
                    A_dt_vs_B_dt=sd.get("A_dt_vs_B_dt"),
                    B_minid_vs_C_minid=sd.get("B_minid_vs_C_minid"),
                    B_dt_vs_C_dt=sd.get("B_dt_vs_C_dt"),
                    evidence_tier="rerun", ts_utc=K.now()))
            agg = {}
            for rep in REPS:
                for k, v in per[0][rep].items():
                    vals = np.array([p[rep][k] for p in per], float)
                    agg[f"{k}_{rep}"] = dict(
                        mean=float(vals.mean()),
                        std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        min=float(vals.min()), max=float(vals.max()))
            set_metrics[f"{w}|t{tau}"] = dict(n=n, seeds=seeds, metrics=agg,
                                              evidence_tier="rerun", ts_utc=K.now())
            print(f"[{w} t{tau}] seeds {seeds}: raw_missed {agg['raw_missed_minid']['mean']:.1f}"
                  f", verified_missed {agg['verified_missed_minid']['mean']:.1f}"
                  f", witness_missed {agg['witness_missed_minid']['mean']:.1f}", flush=True)
        del off, flat
    json.dump(set_metrics, open(os.path.join(OUT, "set_metrics.json"), "w"), indent=1)
    with open(os.path.join(OUT, "representative_rules.jsonl"), "w") as f:
        for r in rep_rules:
            f.write(json.dumps(r) + "\n")

    # ---------------------------------------------------------------- table
    rows = []
    for w in K.WORKLOADS:
        for tau in K.TAUS:
            acc = [r for r in runs if r["workload"] == w and abs(r["tau"] - tau) < 1e-9]
            if not acc:
                continue
            tc = timing_cells(trows, w, tau)
            sm = set_metrics.get(f"{w}|t{tau}", {}).get("metrics", {})
            for method in ("A", "B", "W", "C"):
                for rep in REPS:
                    sub = [r for r in acc if r["method"] == method
                           and r["representative"] == rep]
                    if not sub:
                        continue
                    dele = np.array([r["deleted"] for r in sub], float)
                    uns = np.array([r["unsupported"] for r in sub], float)
                    row = dict(
                        workload=w, tau=tau, method=method, representative=rep,
                        n=sub[0]["n"],
                        seed_count=len({r["hash_seed"] for r in sub}),
                        deleted_mean=dele.mean(), deleted_min=dele.min(),
                        deleted_max=dele.max(),
                        unsupported_mean=uns.mean(),
                        unsupported_share_del_mean=float(np.mean(
                            [r["unsupported_share_del"] for r in sub
                             if r["unsupported_share_del"] is not None])),
                        unsupported_share_input_mean=float(np.mean(
                            [r["unsupported_share_input"] for r in sub])),
                        witness_coverage_mean=float(np.mean(
                            [r["witness_coverage"] for r in sub])),
                        residual_duplicate_pairs_mean=float(np.mean(
                            [r["residual_duplicate_pairs"] for r in sub])),
                    )
                    # cross-method set metrics (min-id semantics; dt rows carry the
                    # same definition under the datatrove representative)
                    # set metrics belong to the method whose deletion set they describe
                    want = {"A": ("raw_missed", "hijacked_raw"),
                            "B": ("verified_missed", "hijacked_verified"),
                            "W": ("witness_missed",),
                            "C": ()}[method]
                    for k in want:
                        key = f"{k}_{rep}"
                        row[k + "_mean"] = sm.get(key, {}).get("mean")
                        row[k + "_std"] = sm.get(key, {}).get("std")
                    row.update({k: v for k, v in tc[method].items()})
                    rows.append(row)
    os.makedirs(OUT, exist_ok=True)
    csvp = os.path.join(OUT, "main_decision_table.csv")
    head = ["workload", "tau", "method", "representative", "n", "seed_count",
            "deleted_mean", "deleted_min", "deleted_max", "unsupported_mean",
            "unsupported_share_del_mean", "unsupported_share_input_mean",
            "witness_coverage_mean", "residual_duplicate_pairs_mean",
            "raw_missed_mean", "verified_missed_mean", "witness_missed_mean",
            "hijacked_raw_mean", "hijacked_verified_mean",
            "cold_median_s", "cold_min_s", "cold_max_s", "cold_composed_median_s",
            "cold_composition", "shared_preproc_median_s",
            "incremental_median_s", "incremental_min_s", "incremental_max_s",
            "incremental_algo_median_s", "incremental_1thread_median_s",
            "incremental_1thread_algo_median_s", "cold_peak_rss_mb",
            "incremental_peak_rss_mb", "cold_n", "incremental_n",
            "incremental_1thread_n"]
    with open(csvp, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=head, extrasaction="ignore")
        wri.writeheader()
        for r in rows:
            wri.writerow(r)
    json.dump(rows, open(os.path.join(OUT, "main_decision_table.json"), "w"), indent=1)
    make_tex(rows, head, os.path.join(OUT, "main_decision_table.tex"))
    print("wrote", csvp, len(rows), "rows")


def make_tex(rows, head, path):
    """Compact LaTeX table: one row per workload/tau/method (min-id representative)."""
    cols = ["workload", "tau", "method", "deleted_mean", "unsupported_mean",
            "unsupported_share_del_mean", "residual_duplicate_pairs_mean",
            "witness_coverage_mean", "cold_median_s", "cold_composed_median_s",
            "shared_preproc_median_s", "incremental_median_s",
            "incremental_1thread_median_s", "cold_peak_rss_mb"]

    def fmt(v):
        if v is None:
            return "--"
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)

    lines = [r"\begin{tabular}{lllrrrrrrrrrrr}", r"\toprule",
             " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for r in rows:
        if r["representative"] != "minid":
            continue
        lines.append(" & ".join(fmt(r.get(c)) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
