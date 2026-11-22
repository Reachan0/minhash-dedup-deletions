#!/usr/bin/env python3
"""Baseline comparison: BigCode MinHash vs the core A/B/W/C methods.

Two comparisons are produced, because "the baseline is worse" is only meaningful
when the operating point is pinned:

  matched deletion budget : the baseline's unsupported share / recall at the number
                            of deletions a core method makes (linear interpolation
                            between the two bracketing (B,R) configs);
  matched pair recall     : the baseline's deletions / unsupported share at the
                            core method's candidate-edge recall.

Outputs `extra/baseline_comparison.json` and `extra/baseline_comparison.md`.
Evidence tier: log-recomputed (numbers read from rerun JSONL files; no new compute).
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRA = os.path.dirname(os.path.dirname(HERE)) + "/extra"
ROWS = os.path.join(EXTRA, "baseline_rows.jsonl")
CORE_RUNS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "core", "runs.jsonl")
TABLE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "core", "main_decision_table.json")
OUTJ = os.path.join(EXTRA, "baseline_comparison.json")
OUTM = os.path.join(EXTRA, "baseline_comparison.md")


def load():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    runs = [json.loads(l) for l in open(CORE_RUNS)]
    table = json.load(open(TABLE))
    return ok, runs, table


def core_stats(runs, table, workload, tau):
    """Mean per-method metrics from runs.jsonl (recall) and the decision table (rest)."""
    out = {}
    for m in ("A", "B", "W", "C"):
        sel = [r for r in runs if r["workload"] == workload and r["tau"] == tau
               and r["method"] == m and r.get("representative", "minid") == "minid"]
        rec = [r["pair_recall"] for r in sel if r.get("pair_recall") is not None]
        tsel = [r for r in table if r["workload"] == workload and r["tau"] == tau
                and r["method"] == m and r["representative"] == "minid"]
        t = tsel[0] if tsel else {}
        out[m] = dict(
            deleted_mean=t.get("deleted_mean"), deleted_seeds=len(sel),
            unsupported_share_del=t.get("unsupported_share_del_mean"),
            unsupported_share_input=t.get("unsupported_share_input_mean"),
            pair_recall_mean=(sum(rec) / len(rec)) if rec else None,
            pair_recall_note=("A: recall of chained adjacent edges, not co-bucket coverage"
                              if m == "A" else None),
            witness_coverage=t.get("witness_coverage_mean"),
            residual=t.get("residual_duplicate_pairs_mean"),
            incremental_s=t.get("incremental_median_s"),
            incremental_algo_s=t.get("incremental_algo_median_s"),
            peak_rss_mb=t.get("incremental_peak_rss_mb"),
            cold_s=t.get("cold_median_s"))
    return out


def interp(rows, key, target, metric):
    """Linear interpolation of `metric` at `target` on axis `key`, using bracketing rows."""
    pts = sorted([(r[key], r[metric]) for r in rows if r.get(metric) is not None])
    if not pts:
        return None
    if target <= pts[0][0]:
        return dict(value=pts[0][1], mode="below_range", at=pts[0][0])
    if target >= pts[-1][0]:
        return dict(value=pts[-1][1], mode="above_range", at=pts[-1][0])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= target <= x1:
            w = 0.0 if x1 == x0 else (target - x0) / (x1 - x0)
            return dict(value=y0 + w * (y1 - y0), mode="interpolated",
                        between=[x0, x1], weight=round(w, 4))
    return None


def matched(rows, core):
    out = {}
    for m in ("A", "B", "C", "W"):
        tgt = core[m]["deleted_mean"]
        if tgt is None:
            continue
        mets = {k: interp(rows, "deleted", tgt, k) for k in
                ("unsupported_share_del", "unsupported_share_input", "pair_recall",
                 "witness_coverage", "residual_duplicate_pairs")}
        out[m] = dict(target_deleted=tgt, **mets)
    out["_matched_recall"] = {}
    for m in ("B", "W"):
        tgt = core[m]["pair_recall_mean"]
        if tgt is None:
            continue
        mets = {k: interp(rows, "pair_recall", tgt, k) for k in
                ("deleted", "unsupported_share_del", "unsupported_share_input",
                 "witness_coverage", "residual_duplicate_pairs")}
        out["_matched_recall"][m] = dict(target_pair_recall=tgt, **mets)
    return out


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, dict):
        v = v["value"]
    return f"{v:.{nd}f}"


def main():
    ok, runs, table = load()
    groups = defaultdict(list)
    for r in ok:
        groups[(r["workload"], r["tau"])].append(r)
    out = dict(rows_in=len(ok), by_workload={}, matched={})
    md = ["# BigCode MinHash baseline vs core A/B/W/C", "",
          "Baseline: BigCode `text-dedup` MinHash estimator (256 permutations, SHA1-32, "
          "`optimal_param`, no exact verification), run on the same core shingles and "
          "scored against the same exact truth C. Core numbers: "
          "`core/main_decision_table.json` (seed means, min-id representative) and "
          "`core/runs.jsonl` (pair recall).", ""]

    for (wl, tau), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["deleted"])
        core = core_stats(runs, table, wl, tau)
        seeds = sorted({r["seed"] for r in rows})
        md += [f"## {wl.upper()} τ={tau}", "",
               "| config | deleted | unsupported | unsup/deleted | unsup/input | pair recall | "
               "residual | witness | t_pair+union (s) | peak RSS (MB) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append("| {config} | {deleted} | {unsupported} | {sd:.3f} | {si:.5f} | {pr:.3f} | "
                      "{res} | {wit:.3f} | {t:.1f} | {rss:.0f} |".format(
                          config=r["config"], deleted=r["deleted"], unsupported=r["unsupported"],
                          sd=r["unsupported_share_del"], si=r["unsupported"] / r["n"],
                          pr=r["pair_recall"], res=r["residual_duplicate_pairs"],
                          wit=r["witness_coverage"],
                          t=r.get("t_pairs_s", 0) + r.get("t_union_s", 0),
                          rss=r["peak_rss_mb"]))
        by_seed = defaultdict(dict)
        for r in rows:
            by_seed[r["seed"]][r["config"]] = (r["deleted"], r["unsupported_share_del"])
        out["by_workload"][f"{wl}_t{tau}"] = dict(
            seeds=seeds, signature_n=rows[0]["n"],
            signature_cold_s=rows[0].get("t_signature_s"), core=core,
            sweep=[{k: r[k] for k in ("config", "deleted", "unsupported",
                                      "unsupported_share_del", "pair_recall",
                                      "residual_duplicate_pairs", "witness_coverage",
                                      "t_banding_stats_s", "t_pairs_s", "t_union_s",
                                      "t_algo_s", "peak_rss_mb", "seed")} for r in rows],
            by_seed={str(s): v for s, v in by_seed.items()})
        m = matched(rows, core)
        out["matched"][f"{wl}_t{tau}"] = m

        md += ["", "**Core references (min-id, seed means).**", "",
               "| method | deleted | unsup/deleted | unsup/input | pair recall | witness | residual |",
               "|---|---|---|---|---|---|---|"]
        for k in ("A", "B", "W", "C"):
            c = core[k]
            md.append(f"| {k} | {c['deleted_mean']:.0f} | {fmt(c['unsupported_share_del'])} | "
                      f"{fmt(c['unsupported_share_input'], 5)} | {fmt(c['pair_recall_mean'])} | "
                      f"{fmt(c['witness_coverage'])} | {fmt(c['residual'], 1)} |")
        md += ["", "**Matched deletion budget** (baseline interpolated on `deleted`):", "",
               "| target | baseline deleted | baseline unsup/deleted | baseline pair recall | "
               "core unsup/deleted | core pair recall |", "|---|---|---|---|---|---|"]
        for k in ("A", "B", "C", "W"):
            if k not in m:
                continue
            c = core[k]
            md.append(f"| {k} ({m[k]['target_deleted']:.0f}) | {fmt(m[k]['deleted'] if 'deleted' in m[k] else m[k]['target_deleted'], 0)} | "
                      f"{fmt(m[k]['unsupported_share_del'])} | {fmt(m[k]['pair_recall'])} | "
                      f"{fmt(c['unsupported_share_del'])} | {fmt(c['pair_recall_mean'])} |")
        md += ["", "**Matched pair recall** (baseline interpolated on `pair_recall`):", "",
               "| target | baseline deleted | baseline unsup/deleted | core deleted | core unsup/deleted |",
               "|---|---|---|---|---|"]
        for k, mm in m["_matched_recall"].items():
            c = core[k]
            md.append(f"| {k} ({mm['target_pair_recall']:.3f}) | {fmt(mm['deleted'], 0)} | "
                      f"{fmt(mm['unsupported_share_del'])} | {c['deleted_mean']:.0f} | "
                      f"{fmt(c['unsupported_share_del'])} |")
        md.append("")

    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(OUTM, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
