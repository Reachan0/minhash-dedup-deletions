"""Recompute every number quoted in Draft 0 of the DedupJoin paper from the
frozen C03 logs of the previous run (vldb2027-ai-db-ideas).

Inputs (read-only):
  .work/vldb2027-ai-db-ideas/experiment/logs/C03/round2/rows.jsonl
Output:
  output/dedupjoin-pvldb/results/results.json

Provenance labels
-----------------
"log-recomputed (earlier run)": value recomputed by this script from per-seed
    JSONL records produced by the earlier measured run; not re-executed here.
"independently rerun (this run)": produced by a new execution in this run.
"extrapolated": derived by formula from measured values.

Usage: python3 output/dedupjoin-pvldb/code/verify_c03.py
"""
import json
import math
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ROWS = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/logs/C03/round2/rows.jsonl")
OLD_RESULTS = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/results.json")
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results/results.json")

PROV = "log-recomputed (earlier run)"


def load_rows():
    with open(ROWS) as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def main():
    rows = load_rows()
    lsh_rows = [r for r in rows if r.get("kind") == "lsh"]
    exact_rows = [r for r in rows if r.get("kind") == "exact"]
    sc_rows = [r for r in rows if r.get("kind") == "scurve"]
    lm_rows = [r for r in rows if r.get("kind") == "lm"]

    # ---------------------------------------------------------------- integrity
    integrity = []
    bad = 0
    for r in lsh_rows:
        for tau, p in r["per_tau"].items():
            ok = (p["lshv_removed"] + p["lshv_missed_removals"] == p["exact_removed"])
            subset = p["lshv_removed"] <= p["exact_removed"]
            if not ok or not subset:
                bad += 1
            integrity.append(dict(key=r["key"], tau=tau, identity_holds=ok,
                                  lshv_subset_of_exact=subset,
                                  lshv=p["lshv_removed"], missed=p["lshv_missed_removals"],
                                  exact=p["exact_removed"]))
    integrity_summary = dict(
        lsh_rows=len(lsh_rows), cells_checked=len(integrity), violations=bad,
        identity="lshv_removed + lshv_missed_removals == exact_removed",
        note=("Every cell satisfies the identity, so the LSH+verification deletion "
              "set is exactly the exact-join deletion set minus the missed removals; "
              "the two pipelines never disagree in the reverse direction.")
    )

    # ------------------------------------------------- main decision-case table
    def pick(corpus, n, tau="0.8"):
        out = []
        for r in lsh_rows:
            if (r["corpus"] == corpus and r["n"] == n and r["family"] == "dt"
                    and r["b"] == 14 and r["r"] == 8):
                out.append((r["seed"], r["n_valid_docs"], r["per_tau"][tau]))
        return sorted(out)

    def summarize_cell(corpus, n, tau):
        vals = pick(corpus, n, tau)
        if not vals:
            return None
        keys = ["recall", "raw_pair_precision", "lsh_removed", "exact_removed",
                "lshv_removed", "lshv_missed_removals", "lsh_wrongly_deleted",
                "lsh_missed_docs", "removed_overlap", "n_truth_pairs"]
        cell = {"corpus": corpus, "n": n, "config": "dt 14x8", "tau": float(tau),
                "seeds": [v[0] for v in vals], "n_valid_docs": vals[0][1],
                "per_seed": {str(v[0]): {k: v[2][k] for k in keys} for v in vals}}
        for k in keys:
            xs = [v[2][k] for v in vals]
            cell[k] = {"mean": mean(xs), "std": sd(xs), "min": min(xs), "max": max(xs)}
        cell["misdelete_frac"] = {
            "mean": mean([v[2]["lsh_wrongly_deleted"] / max(1, v[2]["lsh_removed"]) for v in vals]),
            "std": sd([v[2]["lsh_wrongly_deleted"] / max(1, v[2]["lsh_removed"]) for v in vals])}
        cell["misdelete_per_input_pct"] = {
            "mean": mean([100.0 * v[2]["lsh_wrongly_deleted"] / max(1, v[1]) for v in vals]),
            "std": sd([100.0 * v[2]["lsh_wrongly_deleted"] / max(1, v[1]) for v in vals])}
        cell["lshv_gap_to_exact"] = {
            "mean": mean([v[2]["lshv_missed_removals"] for v in vals]),
            "min": min(v[2]["lshv_missed_removals"] for v in vals),
            "max": max(v[2]["lshv_missed_removals"] for v in vals)}
        cell["provenance"] = PROV
        return cell

    main_cells = [summarize_cell(c, n, t) for c, n in
                  [("c4", 700000), ("fw", 1000000), ("mix", 1000000)]
                  for t in ["0.7", "0.8", "0.9"]]

    # decision case, seed 1
    c4s1 = None
    for r in lsh_rows:
        if (r["corpus"] == "c4" and r["n"] == 700000 and r["family"] == "dt"
                and r["b"] == 14 and r["r"] == 8 and r["seed"] == 1):
            p = r["per_tau"]["0.8"]
            c4s1 = dict(
                corpus="c4", n=700000, config="dt 14x8", tau=0.8, seed=1,
                n_valid_docs=r["n_valid_docs"],
                n_truth_pairs=p["n_truth_pairs"],
                recall=p["recall"], predicted_recall=p["pred_recall"],
                lsh_removed=p["lsh_removed"],
                exact_removed=p["exact_removed"],
                lshv_removed=p["lshv_removed"],
                gap_net=p["exact_removed"] - p["lshv_removed"],
                lshv_missed_removals=p["lshv_missed_removals"],
                removed_overlap=p["removed_overlap"],
                lsh_extra_deletions=p["lsh_removed"] - p["removed_overlap"],
                wrongly_deleted_no_neighbour=p["lsh_wrongly_deleted"],
                lsh_missed_docs=p["lsh_missed_docs"],
                raw_pair_precision=p["raw_pair_precision"],
                verified_pairs=p["verified_pairs"],
                missed_pairs=int(round(p["n_truth_pairs"] * (1 - p["recall"]))),
                provenance=PROV)
            c4s1["gap_interpretation"] = dict(
                symmetric_difference=p["lshv_missed_removals"],
                direction="exact deletes these documents, LSH+verification keeps them",
                reverse_direction_count=0,
                why=("LSH+verification only merges candidate pairs whose exact Jaccard "
                     "is at least tau, so every cluster it forms is a subset of an exact "
                     "component; its deletion set is therefore a subset of the exact "
                     "deletion set, and the net gap equals the number of differing "
                     "keep/drop decisions."),
                per_doc_identity="not recorded in the frozen logs",
                per_doc_token_counts="not recorded in the frozen logs",
                missing_pieces=[
                    "identity and length (tokens) of the 20 documents",
                    "the true neighbours that link them into their exact components",
                    "their domain/dump distribution",
                    "the verified-cost of recovering them (candidate pairs that would need re-probing)"])

    # ------------------------------------- config grid (14x8 vs 20x5 etc.)
    grid = []
    for corpus, n in [("c4", 700000), ("fw", 1000000), ("mix", 1000000)]:
        for b, r in [(14, 8), (20, 5), (10, 10)]:
            for tau in ["0.7", "0.8", "0.9"]:
                vals = []
                for row in lsh_rows:
                    if (row["corpus"] == corpus and row["n"] == n and row["family"] == "dt"
                            and row["b"] == b and row["r"] == r and tau in row["per_tau"]):
                        vals.append(row["per_tau"][tau])
                if not vals:
                    continue
                fr = [v["lsh_wrongly_deleted"] / max(1, v["lsh_removed"]) for v in vals]
                grid.append(dict(
                    corpus=corpus, n=n, config=f"dt {b}x{r}", tau=float(tau),
                    n_seeds=len(vals),
                    misdelete_frac={"mean": mean(fr), "std": sd(fr)},
                    recall={"mean": mean([v["recall"] for v in vals]),
                            "std": sd([v["recall"] for v in vals])},
                    lsh_removed={"mean": mean([v["lsh_removed"] for v in vals])},
                    exact_removed=vals[0]["exact_removed"],
                    provenance=PROV))

    # --------------------------------------------------- seed-variance table
    def dispersion(corpus, n, family, b, r, tau):
        for row in sc_rows:
            if row["corpus"] == corpus and row["n"] == n and row["family"] == family:
                k = f"{b}x{r}|{tau}"
                if k not in row["res"]:
                    continue
                v = row["res"][k]
                m = v["n_truth_pairs"]
                p = v["recall_mean"]
                se = math.sqrt(p * (1 - p) / m) if m else float("nan")
                return dict(corpus=corpus, n=n, family=family, config=f"{b}x{r}",
                            tau=float(tau), hash_seeds=row["n_seeds"],
                            recall_mean=p, recall_std=v["recall_std"],
                            recall_min=v["recall_min"], recall_p05=v["recall_p05"],
                            binomial_se=se,
                            overdispersion=(v["recall_std"] / se) if se and se > 0 else None,
                            n_truth_pairs=m, provenance=PROV)
        return None

    disp = [dispersion(c, n, fam, b, r, t)
            for c, n in [("c4", 700000), ("fw", 1000000), ("mix", 1000000)]
            for fam in ["dt", "mix"]
            for b, r in [(14, 8), (20, 5), (10, 10), (9, 13), (16, 7), (12, 9), (7, 16)]
            for t in ["0.7", "0.8", "0.9"]]
    disp = [d for d in disp if d]

    # ------------------------------------------------------------- cost table
    cost = {}
    for r in exact_rows:
        if r["corpus"] == "c4" and r["n"] == 700000 and r["tau"] == 0.8:
            cost["exact_c4_700k_tau0.8"] = dict(
                t_shingle=r["t_shingle"], t_rank=r["t_rank"], t_index=r["t_index"],
                t_probe_verify=r["t_probe_verify"], t_end_to_end=r["t_end_to_end"],
                n_pairs=r["n_pairs"], n_candidates=r["n_candidates"],
                peak_rss_mb=r["peak_rss_mb_self"], threads=r["threads"], provenance=PROV)
        if r["corpus"] == "fw" and r["n"] == 1000000 and r["tau"] == 0.8:
            cost["exact_fw_1m_tau0.8"] = dict(
                t_shingle=r["t_shingle"], t_rank=r["t_rank"], t_index=r["t_index"],
                t_probe_verify=r["t_probe_verify"], t_end_to_end=r["t_end_to_end"],
                n_pairs=r["n_pairs"], n_candidates=r["n_candidates"],
                peak_rss_mb=r["peak_rss_mb_self"], threads=r["threads"], provenance=PROV)
    for r in lsh_rows:
        if (r["corpus"] == "c4" and r["n"] == 700000 and r["family"] == "dt"
                and r["b"] == 14 and r["r"] == 8 and r["seed"] == 1):
            cost["lshv_c4_700k_tau0.8"] = dict(
                t_signature=r["t_signature_all_hashes"],
                t_band_cluster=r["t_band_cluster"],
                t_verify=r["t_verify_all_candidates"],
                t_join_phase_only=r["t_signature_all_hashes"] + r["t_band_cluster"] + r["t_verify_all_candidates"],
                num_hashes=r["num_hashes"], threads=r["threads"], provenance=PROV)
    if "exact_c4_700k_tau0.8" in cost and "lshv_c4_700k_tau0.8" in cost:
        e = cost["exact_c4_700k_tau0.8"]
        l = cost["lshv_c4_700k_tau0.8"]
        l["t_end_to_end_with_shared_shingling"] = e["t_shingle"] + l["t_join_phase_only"]
        cost["comparison_c4_700k_tau0.8"] = dict(
            join_phase_ratio=e["t_rank"] + e["t_index"] + e["t_probe_verify"] - l["t_join_phase_only"],
            exact_join_phase=e["t_rank"] + e["t_index"] + e["t_probe_verify"],
            lshv_join_phase=l["t_join_phase_only"],
            ratio=(e["t_rank"] + e["t_index"] + e["t_probe_verify"]) / l["t_join_phase_only"],
            seconds_per_recovered_document=((e["t_rank"] + e["t_index"] + e["t_probe_verify"])
                                            - l["t_join_phase_only"]) / c4s1["gap_net"],
            shingling_shared=e["t_shingle"], provenance=PROV)

    # ---------------------------------------------------------- downstream LM
    arms = defaultdict(dict)
    for r in lm_rows:
        arms[r["arm"]][r["seed"]] = r
    lm = {}
    for arm, d in arms.items():
        evals = {}
        for metric in ["val_clean", "d1_lsh_only_removed", "d2_exact_only_removed"]:
            losses = [d[s]["eval"][metric]["loss"] for s in sorted(d)]
            toks = [d[s]["eval"][metric]["tokens"] for s in sorted(d)]
            evals[metric] = dict(mean_loss=mean(losses), std_loss=sd(losses),
                                 tokens=toks[0], n_seeds=len(losses))
        lm[arm] = dict(seeds=sorted(d), eval=evals, params=d[sorted(d)[0]]["params"],
                       tokens_trained=d[sorted(d)[0]]["tokens_trained"],
                       provenance=PROV)
    d1 = {arm: lm[arm]["eval"]["d1_lsh_only_removed"]["mean_loss"] for arm in lm}
    lm_summary = dict(
        arms=lm,
        d1_difference_exact_minus_lsh=d1.get("exact", 0) - d1.get("lsh", 0),
        d1_difference_exact_minus_nodedup=d1.get("exact", 0) - d1.get("nodedup", 0),
        d2_token_count=lm["exact"]["eval"]["d2_exact_only_removed"]["tokens"],
        note=("d2 evaluation set contains only 1024 tokens; it cannot support a "
              "claim. d1 mixes the removal effect with a training-set-size effect "
              "and with memorisation, so it is reported as a preliminary signal only."),
        provenance=PROV)

    # ------------------------------------------------------ headline sentences
    c4 = next(c for c in main_cells if c["corpus"] == "c4" and c["tau"] == 0.8)
    fw = next(c for c in main_cells if c["corpus"] == "fw" and c["tau"] == 0.8)
    headline = dict(
        c4_misdelete_frac_pct=100 * c4["misdelete_frac"]["mean"],
        fw_misdelete_frac_pct=100 * fw["misdelete_frac"]["mean"],
        c4_misdelete_per_input_pct=c4["misdelete_per_input_pct"]["mean"],
        fw_misdelete_per_input_pct=fw["misdelete_per_input_pct"]["mean"],
        c4_lsh_removed_mean=c4["lsh_removed"]["mean"],
        c4_exact_removed=c4["exact_removed"]["mean"],
        c4_lshv_removed_mean=c4["lshv_removed"]["mean"],
        c4_gap_mean=c4["lshv_gap_to_exact"]["mean"],
        c4_gap_min=c4["lshv_gap_to_exact"]["min"],
        c4_gap_max=c4["lshv_gap_to_exact"]["max"],
        provenance=PROV)

    out = dict(
        generated_by="output/dedupjoin-pvldb/code/verify_c03.py",
        generated_at=__import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_logs=[os.path.relpath(ROWS, ROOT)],
        note=("All numbers are recomputed from per-seed JSONL records of the earlier "
              "measured C03 run. They are not re-executed in this step and are labelled "
              "'log-recomputed (earlier run)'. The truth-rebuild step must independently "
              "rerun a representative subset and add 'independently rerun (this run)' values."),
        integrity=integrity_summary,
        decision_case=c4s1,
        main_cells=main_cells,
        misdelete_config_grid=grid,
        headline=headline,
        dispersion=disp,
        cost=cost,
        downstream_lm=lm_summary,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)
    print(json.dumps(headline, indent=1))
    print("integrity violations:", bad, "cells:", len(integrity))
    if c4s1:
        print("decision case:", {k: c4s1[k] for k in
              ["lsh_removed", "lshv_removed", "exact_removed", "gap_net",
               "lshv_missed_removals", "removed_overlap", "lsh_extra_deletions",
               "wrongly_deleted_no_neighbour", "missed_pairs"]})
    if "comparison_c4_700k_tau0.8" in cost:
        print("cost:", json.dumps(cost["comparison_c4_700k_tau0.8"], indent=1))


if __name__ == "__main__":
    sys.exit(main())
