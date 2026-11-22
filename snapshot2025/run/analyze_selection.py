#!/usr/bin/env python3
"""Recompute every reveal conclusion from the per-run JSONL files alone.

Inputs (measured data only; no hand-entered numbers):
  newsnap/runs.jsonl            one quality row per (method, tau, seed)
  newsnap/timing.jsonl          one timing row per fresh process
  newsnap/timing_sidecar.jsonl  process-tree RSS for wrapped runs
  newsnap/failed.jsonl          literal attempts that did not finish
  newsnap/band_stats.json       seed-1 bucket statistics (diagnostic)
  newsnap/collapsed_build.json  duplicate-collapsed corpus build
  newsnap/dup_sets.json         identical-shingle-set groups
  inputs/sampling_diag.json     2,000-document sampling diagnostic

Frozen constants below are the values of SELECTION_PREREG.md §3/§4, whose
SHA-256 was verified before any measurement (inputs/prereg_hash_check.txt);
point predictions are read from the frozen selection_predictions.json.

Outputs (same directory as this script's parent newsnap/):
  selection_results.json, prediction_vs_measured.csv, per_seed_quality.csv,
  timing_runs.csv, RUN_SUMMARY.md
"""
import csv
import json
import os
import statistics
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG = os.path.dirname(HERE)
SNAP = os.path.join(SLUG, "newsnap")
INPUTS = os.path.join(SLUG, "inputs")
N_INPUT = 531597          # SELECTION_PREREG.md §3 residual shares use input size
TAU_MAIN = 0.8
TAUS = [0.8, 0.75]
GI = 1024.0 * 1024.0 * 1024.0

# SELECTION_PREREG.md §3, transcribed verbatim (frozen before the reveal)
SCENARIOS = {
    "S1": dict(
        title="每次删除都要可审计（直达见证是硬约束）",
        recommended="W", tau=0.80,
        hard=dict(witness_coverage=1.0, unsupported=0),
        quality=dict(residual_share=0.001, verified_missed_share=0.05),
        cost=dict(cold_s=400.0, inc_s=20.0, rss_mb=16 * 1024.0),
        primary_cost="cold",
    ),
    "S2": dict(
        title="内存受限服务（峰值 RSS ≤ 6 GiB，增量摄入）",
        recommended="B", tau=0.80,
        hard=dict(rss_mb=6144.0, unsupported=0),
        quality=dict(verified_missed_share=0.05, residual_share=0.01),
        cost=dict(inc_s=30.0),
        primary_cost="inc",
    ),
    "S3": dict(
        title="签名缓存已存在（预处理已付费）",
        recommended="B", tau=0.80,
        hard=dict(unsupported=0, signatures_cached=True),
        quality=dict(verified_missed_share=0.05, residual_share=0.01),
        cost=dict(inc_s=15.0),
        primary_cost="inc",
    ),
}
SAMPLING_ALARM_FACTOR = 10.0
POOL_C4_RATE = 0.19          # SELECTION_PREREG.md §4
METHODS = ["A", "B", "W", "C"]


def load_jsonl(p):
    rows = []
    if not os.path.exists(p):
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def median(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


# ------------------------------------------------------------------ sidecar
def parse_sidecar(rows):
    """Map wrapped runs to their process-tree peak RSS by their command line."""
    out = []
    for r in rows:
        cmd = r.get("cmd") or []
        d = dict(label=r.get("label"), wall_s=r.get("wall_s"),
                 peak_tree_rss_mb=r.get("peak_tree_rss_mb"),
                 exit_code=r.get("exit_code"), timed_out=r.get("timed_out"),
                 ts_utc=r.get("ts_utc"), method=None, tau=None, mode=None,
                 repeat=None)
        j = " ".join(cmd)
        for m in METHODS:
            if f"--method {m}" in j and "wrap_timing.py" in j:
                d["method"] = m
        for tok in ("--tau", "--mode", "--repeat"):
            if tok in cmd:
                k = cmd.index(tok)
                if k + 1 < len(cmd):
                    v = cmd[k + 1]
                    if tok == "--tau":
                        d["tau"] = float(v)
                    elif tok == "--mode":
                        d["mode"] = v
                    else:
                        d["repeat"] = int(v)
        out.append(d)
    return out


def match_tree_rss(timing_rows, sidecar):
    by_key = {}
    for d in sidecar:
        if d["method"] and d["mode"] is not None:
            by_key[(d["method"], d["tau"], d["mode"], d["repeat"])] = d
    for r in timing_rows:
        k = (r.get("method"), r.get("tau"), r.get("mode"), r.get("repeat"))
        d = by_key.get(k)
        r["peak_tree_rss_mb"] = d.get("peak_tree_rss_mb") if d else None
        r["tree_wall_s"] = d.get("wall_s") if d else None
    return timing_rows


# ------------------------------------------------------------------ quality
def quality_summary(rows):
    """Per (method, tau): every seed row plus a conservative aggregate
    (max over seeds for error metrics, min for witness coverage)."""
    out = {}
    for method in METHODS:
        out[method] = {}
        for tau in TAUS:
            rs = [r for r in rows if r["method"] == method
                  and abs(float(r["tau"]) - tau) < 1e-9]
            if not rs:
                continue
            seeds = sorted(rs, key=lambda r: (r.get("hash_seed") or 0))
            agg = dict(
                n_seeds=len(seeds),
                deleted=median([r["deleted"] for r in seeds]),
                unsupported=median([r["unsupported"] for r in seeds]),
                unsupported_share_del=max(
                    (r.get("unsupported_share_del") or 0.0) for r in seeds),
                unsupported_share_input=max(
                    r.get("unsupported_share_input") or 0.0 for r in seeds),
                residual_duplicate_pairs=max(
                    r["residual_duplicate_pairs"] for r in seeds),
                residual_share=max(r["residual_duplicate_pairs"] for r in seeds)
                / N_INPUT,
                witness_coverage=min(r["witness_coverage"] for r in seeds),
                verified_missed_share=(0.0 if method == "C" else (max(
                    (r["verified_missed"] / r["exact_deleted"])
                    if r.get("exact_deleted") else 0.0 for r in seeds)
                    if any(r.get("verified_missed") is not None for r in seeds)
                    else None)),
                raw_missed_share=(max(
                    (r["raw_missed"] / r["exact_deleted"])
                    if r.get("exact_deleted") else 0.0 for r in seeds)
                    if any(r.get("raw_missed") is not None for r in seeds)
                    else None),
                doc_recall=min(r["doc_recall"] for r in seeds),
                exact_deleted=median([r["exact_deleted"] for r in seeds]),
            )
            out[method][tau] = dict(
                seeds=[dict(seed=r.get("hash_seed"), deleted=r["deleted"],
                            unsupported=r["unsupported"],
                            unsupported_share_del=r.get("unsupported_share_del"),
                            unsupported_share_input=r.get("unsupported_share_input"),
                            verified_missed=r.get("verified_missed"),
                            raw_missed=r.get("raw_missed"),
                            residual_duplicate_pairs=r["residual_duplicate_pairs"],
                            residual_share=r["residual_duplicate_pairs"] / N_INPUT,
                            witness_coverage=r["witness_coverage"],
                            doc_recall=r["doc_recall"],
                            exact_deleted=r["exact_deleted"],
                            hijacked_verified=r.get("hijacked_verified"),
                            hijacked_raw=r.get("hijacked_raw"))
                       for r in seeds],
                aggregate=agg)
    return out


# ------------------------------------------------------------------ timing
def timing_summary(rows):
    out = {}
    for method in METHODS:
        out[method] = {}
        for tau in TAUS:
            out[method][tau] = {}
            for mode in ("pathcold", "incremental", "cold"):
                rs = [r for r in rows if r.get("method") == method
                      and r.get("mode") == mode
                      and r.get("tau") is not None
                      and abs(float(r["tau"]) - tau) < 1e-9
                      and r.get("exit_code", 0) == 0]
                if not rs:
                    continue
                walls = [r["wall_s"] for r in rs]
                rss = [r["peak_rss_mb"] for r in rs]
                tree = [r["peak_tree_rss_mb"] for r in rs
                        if r.get("peak_tree_rss_mb") is not None]
                out[method][tau][mode] = dict(
                    n=len(rs),
                    median_wall_s=median(walls),
                    min_wall_s=min(walls),
                    max_wall_s=max(walls),
                    max_rss_mb=max(rss),
                    median_rss_mb=median(rss),
                    max_tree_rss_mb=max(tree) if tree else None,
                    repeats=[dict(repeat=r.get("repeat"), wall_s=r["wall_s"],
                                  rss_mb=r["peak_rss_mb"],
                                  tree_rss_mb=r.get("peak_tree_rss_mb"),
                                  ts_utc=r.get("ts_utc"),
                                  stages={k: v for k, v in r.items()
                                          if k.endswith("_s") and k != "wall_s"})
                              for r in rs],
                )
    return out


def timing_of(timing, method, tau, mode, field):
    d = timing.get(method, {}).get(tau, {}).get(mode)
    return d.get(field) if d else None


# ------------------------------------------------------------------ scenarios
def scenario_eval(sc, quality, timing, failed):
    tau = sc["tau"]
    rows = {}
    for m in METHODS:
        q = quality.get(m, {}).get(tau, {}).get("aggregate")
        pathcold = timing_of(timing, m, tau, "pathcold", "median_wall_s")
        inc = timing_of(timing, m, tau, "incremental", "median_wall_s")
        rss_inc = timing_of(timing, m, tau, "incremental", "max_rss_mb")
        rss_cold = timing_of(timing, m, tau, "pathcold", "max_rss_mb")
        rows[m] = dict(quality=q, cold_s=pathcold, inc_s=inc,
                       rss_inc_mb=rss_inc, rss_cold_mb=rss_cold,
                       feasible=pathcold is not None and inc is not None)
        # S2/S3 evaluate RSS on the cached state (incremental); S1 on cold.
        rows[m]["rss_mb"] = rss_inc if sc["primary_cost"] != "cold" else rss_cold
        rows[m]["failed"] = [f for f in failed
                             if f.get("label", "").startswith(f"literal_{m}")]

    checks = {}
    for m in METHODS:
        q = rows[m]["quality"] or {}
        c = {}

        def add(name, ok, detail):
            c[name] = dict(pass_=bool(ok), detail=detail)

        if sc["hard"].get("witness_coverage") is not None:
            add("witness_coverage", q.get("witness_coverage") == 1.0,
                f"witness_coverage={q.get('witness_coverage')} (需=1.0)")
        if sc["hard"].get("unsupported") == 0:
            add("unsupported", q.get("unsupported_share_del") == 0.0,
                f"无支持删除占比={q.get('unsupported_share_del')} (需=0)")
        if sc["hard"].get("rss_mb") is not None:
            v = rows[m]["rss_mb"]
            add("rss", v is not None and v <= sc["hard"]["rss_mb"],
                f"峰值 RSS={v} MB (需≤{sc['hard']['rss_mb']:.0f})")
        if sc["hard"].get("signatures_cached"):
            # incremental runs must not include a timed signature stage
            t = timing.get(m, {}).get(tau, {}).get("incremental")
            sig_stage = None
            if t:
                sig_stage = max((rp.get("stages", {}).get("signature_s", 0.0)
                                 for rp in t["repeats"]), default=0.0)
            add("signatures_cached",
                t is not None and (sig_stage or 0.0) == 0.0,
                f"增量运行签名重建耗时={sig_stage}s (需=0，读缓存)")
        for k, lim in sc["quality"].items():
            v = q.get(k)
            detail = f"{k}={v} (需≤{lim})"
            if (k == "verified_missed_share" and v is None
                    and q.get("raw_missed_share") is not None):
                v = q["raw_missed_share"]
                detail = (f"verified_missed 不适用（A 无验证图），用 raw missed="
                          f"{v:.6f} 保守代入 (需≤{lim})")
            add(k, v is not None and v <= lim, detail)
        for k, lim in sc["cost"].items():
            v = rows[m]["cold_s"] if k == "cold_s" else (
                rows[m]["inc_s"] if k == "inc_s" else rows[m]["rss_mb"])
            add(k, v is not None and v <= lim, f"{k}={v} (需≤{lim})")
        c["executable"] = dict(pass_=rows[m]["feasible"],
                               detail="该路径在快照上完成全部字面运行"
                               if rows[m]["feasible"] else
                               "字面运行未能完成（见 failed.jsonl）")
        checks[m] = c

    qualifying = [m for m in METHODS
                  if all(v["pass_"] for v in checks[m].values())]

    def cost_key(m):
        if sc["primary_cost"] == "cold":
            return (rows[m]["cold_s"] if rows[m]["cold_s"] is not None else 1e18,
                    rows[m]["rss_mb"] or 1e18)
        return (rows[m]["inc_s"] if rows[m]["inc_s"] is not None else 1e18,
                rows[m]["rss_mb"] or 1e18)

    cheapest = min(qualifying, key=cost_key) if qualifying else None
    rec = sc["recommended"]
    rec_pass = checks[rec] and all(v["pass_"] for v in checks[rec].values())
    return dict(checks=checks, per_method=rows,
                recommended=rec, recommended_pass=rec_pass,
                qualifying=qualifying, cheapest_qualifying=cheapest,
                executable=[m for m in METHODS if rows[m]["feasible"]])


def wrong_choice_costs(quality, timing, scen):
    out = {}
    b = quality.get("B", {}).get(TAU_MAIN, {}).get("aggregate")
    w = quality.get("W", {}).get(TAU_MAIN, {}).get("aggregate")
    a = quality.get("A", {}).get(TAU_MAIN, {}).get("aggregate")
    cq = quality.get("C", {}).get(TAU_MAIN, {}).get("aggregate")
    if b:
        out["S1_B_instead_of_W"] = dict(
            b_deletions_lacking_retained_witness_share=1.0 - b["witness_coverage"],
            b_unsupported_share_del=b["unsupported_share_del"],
            w_witness_coverage=w["witness_coverage"] if w else None)
    if a:
        out["S1_A_when_witness_required"] = dict(
            unsupported_share_del=a["unsupported_share_del"],
            unsupported_share_input=a["unsupported_share_input"],
            residual_duplicate_pairs=a["residual_duplicate_pairs"],
            residual_share=a["residual_share"],
            witness_coverage=a["witness_coverage"])
    pred = scen["predictions"]
    if cq:
        b_pred_cold = pred["cold_B"]["wall_s"]["pred"]
        c_pred_cold = pred["cold_C"]["wall_s"]["pred"]
        out["S1_C_instead_of_B"] = dict(
            predicted_excess_cold_s=c_pred_cold - b_pred_cold,
            predicted_excess_rss_mb=pred["cold_C"]["peak_rss_mb"]["pred"]
            - pred["cold_B"]["peak_rss_mb"]["pred"],
            measured_cold_B_s=timing_of(timing, "B", TAU_MAIN, "pathcold", "median_wall_s"),
            measured_cold_C_s=timing_of(timing, "C", TAU_MAIN, "pathcold", "median_wall_s"),
            note="两条路径在快照上的字面冷运行均未完成；此处为冻结预测差值与字面失败记录")
    if a:
        out["S3_A_counterfactual"] = dict(
            a_inc_s=timing_of(timing, "A", TAU_MAIN, "incremental", "median_wall_s"),
            b_pred_inc_s=pred["incremental_B"]["wall_s"]["pred"],
            a_unsupported_share_del=a["unsupported_share_del"],
            a_residual_share=a["residual_share"],
            note="仅作为注记：S3 的策略要求 unsupported=0，A 不满足，不构成可选路径")
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="从逐次 JSONL 重算选型结论")
    ap.add_argument("--snap", default=SNAP,
                    help="含 runs/timing/timing_sidecar/failed/band_stats/dup_sets/"
                         "collapsed_build/largest_group 的目录（默认 <slug>/newsnap）")
    ap.add_argument("--inputs", default=INPUTS,
                    help="含 INPUT_PATHS.json 与 sampling_diag.json 的目录")
    ap.add_argument("--predictions", default=None,
                    help="冻结的 selection_predictions.json（默认由 INPUT_PATHS.json 指定）")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认写入 --snap）")
    args = ap.parse_args(argv)
    snap, inputs, out_dir = args.snap, args.inputs, args.out or args.snap

    runs = load_jsonl(os.path.join(snap, "runs.jsonl"))
    timing_rows = load_jsonl(os.path.join(snap, "timing.jsonl"))
    side = load_jsonl(os.path.join(snap, "timing_sidecar.jsonl"))
    failed = load_jsonl(os.path.join(snap, "failed.jsonl"))
    timing_rows = match_tree_rss(timing_rows, parse_sidecar(side))
    # a timing row for a failed process (non-zero exit) is not a timing datum
    timing_rows = [r for r in timing_rows if r.get("exit_code", 0) == 0]

    paths = read_json(os.path.join(inputs, "INPUT_PATHS.json"), {})
    pred_file = args.predictions or paths.get("selection_predictions_json")
    predictions = read_json(pred_file, {}).get("predictions", {}) if pred_file else {}
    sampling = read_json(os.path.join(inputs, "sampling_diag.json"), {})
    band = read_json(os.path.join(snap, "band_stats.json"), {})
    dup = read_json(os.path.join(snap, "dup_sets.json"), {})
    collapsed = read_json(os.path.join(snap, "collapsed_build.json"), {})

    quality = quality_summary(runs)
    timing = timing_summary(timing_rows)
    scen = dict(predictions=predictions)
    scenarios = {name: scenario_eval(sc, quality, timing, failed)
                 for name, sc in SCENARIOS.items()}

    lg = read_json(os.path.join(snap, "largest_group.json"), {})
    dup_density = dict(
        n_documents=dup.get("n"),
        n_unique_sets=dup.get("n_unique_sets"),
        n_duplicate_groups=dup.get("n_duplicate_groups"),
        n_docs_in_duplicate_groups=dup.get("n_docs_in_duplicate_groups"),
        largest_group=dup.get("largest_group"),
        largest_group_shingles=lg.get("n_shingles"),
        largest_bucket=band.get("largest_bucket"),
        n_candidate_pairs_doc_unique=max(
            [r["n_candidate_pairs_doc"] for r in runs
             if r.get("n_candidate_pairs_doc")] or [None]),
        candidate_pairs_total=band.get("bucket_pairs_sum"),
        candidate_pairs_per_doc=band.get("pairs_per_doc"),
        pair_array_lower_bound_gib=(band.get("bucket_pairs_sum") or 0) * 4 / GI,
        duplicate_density_too_high_for_B=(band.get("bucket_pairs_sum") or 0)
        * 4 / GI > 32.0,
    )

    sampling_diag = dict(
        n_sample=sampling.get("n_sample"), pairs=sampling.get("pairs"),
        pairs_per_doc=sampling.get("pairs_per_doc"),
        threshold_pairs_per_doc=SAMPLING_ALARM_FACTOR * POOL_C4_RATE,
        alarm=sampling.get("flag_bucket_expansion_exceeds_10x"),
        largest_bucket_sampled=sampling.get("largest_bucket"),
        wall_s=sampling.get("wall_s"), peak_rss_mb=sampling.get("peak_rss_mb"),
        full_snapshot_pairs_per_doc=band.get("pairs_per_doc"),
    )

    out = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        constants=dict(n_input=N_INPUT, tau_main=TAU_MAIN,
                       scenarios={k: {kk: vv for kk, vv in v.items()}
                                  for k, v in SCENARIOS.items()},
                       prediction_file=pred_file),
        inputs=dict(n_quality_rows=len(runs), n_timing_rows=len(timing_rows),
                    n_sidecar_rows=len(side), n_failed=len(failed)),
        quality=quality, timing=timing,
        failed=[dict(label=f.get("label"), cmd=f.get("cmd"),
                     exit_code=f.get("exit_code"), timed_out=f.get("timed_out"),
                     wall_s=f.get("wall_s"),
                     peak_tree_rss_mb=f.get("peak_tree_rss_mb"),
                     ts_utc=f.get("ts_utc"),
                     tail=(f.get("failure_tail") or "")[-400:]) for f in failed],
        predictions=predictions,
        scenarios=scenarios,
        wrong_choice_costs=wrong_choice_costs(quality, timing, scen),
        sampling_diag=sampling_diag,
        duplicate_density=dup_density,
        collapsed=dict(collapsed),
    )
    json.dump(out, open(os.path.join(out_dir, "selection_results.json"), "w"),
              indent=1, ensure_ascii=False)

    # ---------------------------------------------------------------- CSVs
    with open(os.path.join(out_dir, "per_seed_quality.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "tau", "seed", "deleted", "unsupported",
                    "unsupported_share_del", "unsupported_share_input",
                    "verified_missed", "verified_missed_share", "raw_missed",
                    "residual_duplicate_pairs", "residual_share",
                    "witness_coverage", "doc_recall", "exact_deleted"])
        for m in METHODS:
            for tau in TAUS:
                d = quality.get(m, {}).get(tau)
                if not d:
                    continue
                for s in d["seeds"]:
                    vm = s.get("verified_missed")
                    vms = (vm / s["exact_deleted"]) if (vm is not None and s["exact_deleted"]) else ""
                    w.writerow([m, tau, s["seed"] if s["seed"] is not None else "once",
                                s["deleted"], s["unsupported"],
                                s["unsupported_share_del"], s["unsupported_share_input"],
                                "" if vm is None else vm, vms,
                                "" if s.get("raw_missed") is None else s["raw_missed"],
                                s["residual_duplicate_pairs"],
                                f"{s['residual_share']:.6f}",
                                s["witness_coverage"], s["doc_recall"],
                                s["exact_deleted"]])

    with open(os.path.join(out_dir, "timing_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "tau", "mode", "repeat", "wall_s",
                    "peak_rss_mb_self", "peak_tree_rss_mb", "ts_utc", "key", "status"])
        for r in timing_rows:
            w.writerow([r.get("method"), r.get("tau"), r.get("mode"),
                        r.get("repeat"), r.get("wall_s"), r.get("peak_rss_mb"),
                        r.get("peak_tree_rss_mb"), r.get("ts_utc"),
                        r.get("key"), "ok"])
        for r in failed:
            w.writerow([r.get("label"), "", "", "", r.get("wall_s"), "",
                        r.get("peak_tree_rss_mb"), r.get("ts_utc"),
                        r.get("label"), "failed"])

    with open(os.path.join(out_dir, "prediction_vs_measured.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "tau", "pred_cold_s", "meas_cold_s",
                    "pred_inc_s", "meas_inc_s", "pred_rss_mb", "meas_rss_mb",
                    "meas_deleted", "meas_unsupported_share_del",
                    "meas_unsupported_share_input", "meas_residual_pairs",
                    "meas_residual_share", "meas_witness_coverage",
                    "meas_verified_missed_share", "quality_source",
                    "literal_timing_status"])
        for m in METHODS:
            for tau in TAUS:
                q = quality.get(m, {}).get(tau, {}).get("aggregate") or {}
                pk = f"cold_{m}"
                ik = f"incremental_{m}"
                pred_cold = (predictions.get(pk) or {}).get("wall_s", {}).get("pred")
                pred_inc = (predictions.get(ik) or {}).get("wall_s", {}).get("pred")
                pred_rss = (predictions.get(ik) or {}).get("peak_rss_mb", {}).get("pred")
                cold = timing_of(timing, m, tau, "pathcold", "median_wall_s")
                inc = timing_of(timing, m, tau, "incremental", "median_wall_s")
                rss = timing_of(timing, m, tau, "incremental", "max_rss_mb")
                if cold is not None and inc is not None:
                    lit = "ok"
                else:
                    labels = sorted({f.get("label", "") for f in failed
                                     if f.get("label", "").startswith(f"literal_{m}")})
                    lit = ("failed:" + ",".join(labels)) if labels else "missing"
                w.writerow([m, tau, pred_cold, cold, pred_inc, inc, pred_rss, rss,
                            q.get("deleted"), q.get("unsupported_share_del"),
                            q.get("unsupported_share_input"),
                            q.get("residual_duplicate_pairs"),
                            f"{q.get('residual_share', ''):.6f}" if q else "",
                            q.get("witness_coverage"), q.get("verified_missed_share"),
                            "dup_collapsed_exact" if m in ("B", "W") else
                            ("exact_truth" if m == "C" else "literal"),
                            lit])

    write_summary(out, out_dir)
    brief = dict(scenarios={}, quality={}, timing={}, failed=[])
    for k, v in scenarios.items():
        brief["scenarios"][k] = dict(
            recommended=v["recommended"],
            recommended_pass=v["recommended_pass"],
            qualifying=v["qualifying"],
            cheapest=v["cheapest_qualifying"],
            executable=v["executable"])
    for m in METHODS:
        qm = {}
        for t in TAUS:
            q = quality.get(m, {}).get(t)
            if q:
                qm[str(t)] = q["aggregate"]
        brief["quality"][m] = qm
        tm = {}
        for t in TAUS:
            modes = {}
            for mo, d in timing.get(m, {}).get(t, {}).items():
                modes[mo] = dict(n=d["n"], med=d["median_wall_s"], rss=d["max_rss_mb"])
            tm[str(t)] = modes
        brief["timing"][m] = tm
    brief["failed"] = [(f.get("label"), f.get("exit_code"), f.get("wall_s"),
                        f.get("peak_tree_rss_mb")) for f in failed]
    print(json.dumps(brief, indent=1, ensure_ascii=False))


def write_summary(out, out_dir=None):
    L = []
    A = L.append
    A("# 运行汇总（analyze_selection.py 自动生成）\n")
    A(f"生成时间（UTC）：{out['generated_utc']}\n")
    A("全部数值由 `newsnap/*.jsonl` 重新计算，本文件不代表任何手工测量。\n")
    A("\n## 输入行数\n")
    i = out["inputs"]
    A(f"- 质量行（runs.jsonl）：{i['n_quality_rows']}\n")
    A(f"- 计时行（timing.jsonl，仅成功进程）：{i['n_timing_rows']}\n")
    A(f"- 进程树 RSS 采样行（timing_sidecar.jsonl）：{i['n_sidecar_rows']}\n")
    A(f"- 失败记录（failed.jsonl）：{i['n_failed']}\n")

    A("\n## 已完成的运行\n")
    for m in METHODS:
        for t in TAUS:
            tm = out["timing"].get(m, {}).get(t, {})
            if tm:
                parts = [f"{mo}×{d['n']}（中位 {d['median_wall_s']:.2f}s，"
                         f"峰值 RSS {d['max_rss_mb']:.0f} MB）"
                         for mo, d in tm.items()]
                A(f"- {m} τ={t}：" + "；".join(parts) + "\n")
    A("\n## 缺失/失败的运行\n")
    if not out["failed"]:
        A("- 无\n")
    for f in out["failed"]:
        lines = (f.get("tail") or "").strip().splitlines()
        A(f"- {f['label']}：退出码 {f['exit_code']}"
          f"{'（超时）' if f.get('timed_out') else ''}，"
          f"墙钟 {f['wall_s']}s，进程树峰值 {f['peak_tree_rss_mb']} MB；"
          f"末行日志：{lines[-1] if lines else '（空）'}\n")

    A("\n## 场景判定（阈值取自 SELECTION_PREREG.md §3）\n")
    for name, sc in out["scenarios"].items():
        cfg = SCENARIOS[name]
        A(f"### {name} — {cfg['title']}\n")
        A(f"- 推荐路径：{sc['recommended']}；是否满足全部硬约束与质量容差："
          f"{'是' if sc['recommended_pass'] else '否'}\n")
        A(f"- 可执行路径：{', '.join(sc['executable']) or '无'}\n")
        A(f"- 满足全部约束的路径：{', '.join(sc['qualifying']) or '无'}\n")
        A(f"- 实测最便宜合格路径：{sc['cheapest_qualifying'] or '无'}\n")
    A("\n## 采样诊断（§4）\n")
    s = out["sampling_diag"]
    A(f"- 样本 {s['n_sample']} 文档，候选对 {s['pairs']}，每文档 "
      f"{s['pairs_per_doc']:.2f} 对；告警阈值 {s['threshold_pairs_per_doc']:.2f}"
      f"（10×C4 池率 {POOL_C4_RATE}）；是否触发：{'是' if s['alarm'] else '否'}\n")
    A(f"- 采样成本：{s['wall_s']}s，峰值 RSS {s['peak_rss_mb']} MB\n")

    out_dir = out_dir or SNAP
    open(os.path.join(out_dir, "RUN_SUMMARY.md"), "w", encoding="utf-8").write("".join(L))


if __name__ == "__main__":
    main()
