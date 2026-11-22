#!/usr/bin/env python3
"""Assemble results/truth_rebuild.json from the component results and index it in
results/results.json under the key "truth_rebuild".

Every number the paper takes from the truth-rebuild step is read from one of the
component files; the component files are written by the scripts named below.
"""
import json
import os
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RES = os.path.join(ROOT, "output/dedupjoin-pvldb/results")

COMPONENTS = {
    "data_manifest": "data_manifest.py",
    "main_table_truth": "recompute_main_table.py",
    "truth_comparison": "compare_truth.py",
    "shared_h117": "shared_h117_check.py",
    "seed_sensitivity": "seed_summary.py",
    "decision_case": "decision_case.py",
    "decision_cost": "cost_decision.py + cost_compare.py",
    "parity": "pin_parity.py",
    "correctness": "correctness_suite.py",
    "nn_binning": "nn_from_join.py",
    "nn_audited": "nn_audited.py",
    "quality_metrics": "quality_metrics.py",
    "quality_metrics_audited": "quality_metrics_audited.py",
    "corpus_shares": "corpus_shares.py",
}


def load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


def main():
    mt = load("main_table_truth")
    tc = load("truth_comparison")
    sh = load("shared_h117")
    ss = load("seed_sensitivity")
    dc = load("decision_case")
    cost = load("decision_cost")
    par = load("parity")
    cor = load("correctness")
    nn = load("nn_binning")
    nna = load("nn_audited")
    qm = load("quality_metrics")
    qma = load("quality_metrics_audited")
    cs = load("corpus_shares")

    # Three-seed summaries, one row per (cell, tau, design).
    cells = []
    for cell, taus in ss["cells"].items():
        corpus = cell.split("_")[0]
        for tau, designs in sorted(taus.items()):
            for design, v in designs.items():
                if not v:
                    continue
                cells.append(dict(
                    cell=cell, corpus=corpus, tau=float(tau), design=design,
                    n_seeds=v["n_seeds"],
                    mean_lsh_removed=v["mean_lsh_removed"], mean_wrong=v["mean_wrong"],
                    mean_lshv=v["mean_lshv"], mean_exact=v["mean_exact"],
                    gap=dict(mean=v["mean_gap"], min=v["min_gap"], max=v["max_gap"]),
                    mean_recall=v["mean_recall"],
                    unsupported_share_of_deleted_mean=v["share_of_deleted_mean_of_seeds"],
                    unsupported_share_of_deleted_pooled=v["share_of_deleted_pooled"],
                    unsupported_share_of_input_mean=v["share_of_input_mean_of_seeds"]))

    # Per-corpus seed-1 truth rows (log-recomputed) plus token shares.
    seed1 = {}
    for c in mt["cells"]:
        r = [x for x in c["per_seed"] if x["seed"] == 1][0]
        seed1[c["corpus"]] = dict(tau=c["tau"], **{k: r[k] for k in (
            "n_valid_docs", "lsh_removed", "wrongly_deleted", "exact_removed",
            "lshv_removed", "lshv_missed_removals", "removed_overlap",
            "n_truth_pairs", "recall", "pred_recall", "raw_pair_precision", "n_cand_pairs")})

    a = dc["audited_shared_design"]
    standalone_gap = dc["gaps"]["lshv_vs_exact"]
    c4 = seed1["c4"]
    shared_comp = dict(unsupported=c4["wrongly_deleted"],
                       representative_only=c4["lsh_removed"] - c4["wrongly_deleted"]
                       - c4["removed_overlap"],
                       deleted_by_both=c4["removed_overlap"], total=c4["lsh_removed"])
    perf = cost.get("comparison_to_earlier_run", {})
    parity_layers = par["layers"]
    cond = par["conditional_on_dt_tokens"]
    all_layers_identical = all(
        cond[k].get("identical") for k in cond if isinstance(cond[k], dict) and "identical" in cond[k])

    doc = dict(
        version=1,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        provenance=dict(
            logs="log-recomputed (earlier run)",
            rerun="independently rerun (this run)",
            note="the earlier run is recomputed from its per-seed records; the seed-1 cells "
                 "are re-executed from the raw corpus files with an independent implementation"),
        headline_check=mt["headline_check"],
        token_share=mt["token_share"],
        seed1_log_recomputed=seed1,
        seed1_comparison=dict(
            total=tc["n_comparisons"], reproduces=tc["n_reproduces"], differs=tc["n_differs"],
            per_document_meta_identity=tc["per_document_meta_identity"],
            differing=tc["differing"]),
        draw_explanation=dict(
            conclusion="all seed-1 differences are explained by the signature draw",
            columns_identical={k: v["signature_prefix_stability"]["n_columns_identical"]
                               for k, v in sh["cells"].items()},
            columns_compared={k: v["signature_prefix_stability"]["n_columns_compared"]
                              for k, v in sh["cells"].items()},
            fraction_docs_any_signature_difference={
                k: v["signature_prefix_stability"]["fraction_docs_with_any_difference"]
                for k, v in sh["cells"].items()},
            checks={k: v["checks"] for k, v in sh["cells"].items()}),
        three_seed_means=cells,
        decision_case=dict(
            shared=dict(deleted=a["deleted_sets"], gap=a["gap"]["net"],
                        gap_docs=a["gap"]["docs"], composition=shared_comp),
            standalone=dict(deleted=dc["deleted_sets"],
                            gap=standalone_gap["symmetric_difference"],
                            gap_docs=standalone_gap["docs"],
                            composition=dc["standalone_composition"]),
            gap_overlap_docs=a["overlap_with_standalone_gap"]["shared"]),
        cost=perf,
        correctness=dict(n_checks=cor["total_checks"], n_failures=cor["total_failures"]),
        parity=dict(pin=par["datatrove_pin"],
                    tokenizer=dict(n_docs=parity_layers["layer1_tokenization"]["n_docs"],
                                   docs_with_different_tokens=parity_layers["layer1_tokenization"]["token_list_differ"],
                                   fraction=parity_layers["layer1_tokenization"]["token_list_differ"]
                                   / parity_layers["layer1_tokenization"]["n_docs"],
                                   token_count_dt=parity_layers["layer1_tokenization"]["token_count_dt_total"],
                                   token_count_ours=parity_layers["layer1_tokenization"]["token_count_ours_total"]),
                    layers=parity_layers,
                    identical_on_datatrove_tokens=all_layers_identical),
        nearest_neighbour=dict(
            c4_shared=nna["per_tau"], c4_standalone=nn["cells"]["c4"]["per_tau"],
            fw_standalone=nn["cells"]["fw"]["per_tau"],
            bucket_limited=nna["bucket_limited"],
            corpus_short_doc_share={k: v["frac_lt100_shingles"] for k, v in cs["cells"].items()}),
        final_corpus_metrics=dict(
            audited=qma["cells"], standalone=qm["cells"],
            note="three metrics from one exact pair list; the audited design is the paper's main cell"),
        components={k: f"results/{k}.json ({v})" for k, v in COMPONENTS.items()},
    )
    with open(os.path.join(RES, "truth_rebuild.json"), "w") as f:
        json.dump(doc, f, indent=2)

    with open(os.path.join(RES, "results.json")) as f:
        full = json.load(f)
    full["truth_rebuild"] = dict(
        file="results/truth_rebuild.json",
        generated_at=doc["generated_at"],
        headline_check=mt["headline_check"],
        seed1_comparison=doc["seed1_comparison"],
        draw_explanation=doc["draw_explanation"],
        three_seed_means=cells,
        decision_case=doc["decision_case"],
        cost=doc["cost"],
        correctness=doc["correctness"],
        final_corpus_metrics_summary={
            k: {tau: {m: v[tau][m] for m in
                      ("candidate_edge_recall", "residual_duplicate_pairs_raw_lsh",
                       "residual_duplicate_pairs_lsh_verify",
                       "direct_witness_coverage_raw_lsh",
                       "direct_witness_coverage_lsh_verify")}
                for tau in v}
            for k, v in qma["cells"].items()},
    )
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(full, f, indent=2)
    print("wrote results/truth_rebuild.json and updated results/results.json")


if __name__ == "__main__":
    main()
