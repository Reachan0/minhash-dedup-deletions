#!/usr/bin/env python3
"""Append a cross-run comparison to results/decision_cost.json.

Compares the clean re-measurement of the two join phases (this run) with the
values recorded by the earlier run, and derives the per-recovered-document cost
under each. Writes the file back in place.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RES = os.path.join(ROOT, "output/dedupjoin-pvldb/results")


def main():
    dc = json.load(open(os.path.join(RES, "decision_cost.json")))
    old = json.load(open(os.path.join(RES, "results.json")))["cost"]
    old_ex = old["exact_c4_700k_tau0.8"]
    old_l = old["lshv_c4_700k_tau0.8"]
    new_l, new_e = dc["lsh"], dc["exact"]
    gap_shared, gap_standalone = 20, 21  # seed-1 shared cell; three-seed mean standalone
    dc["comparison_to_earlier_run"] = dict(
        earlier=dict(
            provenance=old_l["provenance"],
            lsh_join_phase_s=old_l["t_join_phase_only"],
            exact_join_phase_s=old_ex["t_rank"] + old_ex["t_index"] + old_ex["t_probe_verify"],
            ratio=(old_ex["t_rank"] + old_ex["t_index"] + old_ex["t_probe_verify"])
            / old_l["t_join_phase_only"],
            lsh_peak_rss_mb=None,
            exact_peak_rss_mb=old_ex["peak_rss_mb"],
            s_per_recovered_doc_shared=((old_ex["t_rank"] + old_ex["t_index"] + old_ex["t_probe_verify"])
                                        - old_l["t_join_phase_only"]) / gap_shared,
            s_per_recovered_doc_standalone=((old_ex["t_rank"] + old_ex["t_index"] + old_ex["t_probe_verify"])
                                            - old_l["t_join_phase_only"]) / gap_standalone),
        rerun=dict(
            provenance="independently rerun (this run)",
            lsh_join_phase_s=new_l["total_s"],
            exact_join_phase_s=new_e["total_s"],
            ratio=new_e["total_s"] / new_l["total_s"],
            lsh_peak_rss_mb=new_l["peak_rss_mb"],
            exact_peak_rss_mb=new_e["peak_rss_mb"],
            s_per_recovered_doc_shared=(new_e["total_s"] - new_l["total_s"]) / gap_shared,
            s_per_recovered_doc_standalone=(new_e["total_s"] - new_l["total_s"]) / gap_standalone),
        note="Absolute phase times differ by 2-3x between the two machines; the ratio and the "
             "per-document cost computed within a run are the transportable quantities.")
    with open(os.path.join(RES, "decision_cost.json"), "w") as f:
        json.dump(dc, f, indent=2)
    print(json.dumps(dc["comparison_to_earlier_run"], indent=1))


if __name__ == "__main__":
    main()
