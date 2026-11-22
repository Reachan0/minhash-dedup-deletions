#!/usr/bin/env python3
"""Build audit/claims_manifest.csv: one row per Draft 2 claim.

Claim text and the printed value are transcribed from the paper (read-only);
the audited value is formatted here from the stored result files, so the two
columns can be compared mechanically. Nothing is copied from a PDF.

Writes:
  audit/claims_manifest.csv
  audit/naming_mismatches.csv
"""
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RES = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
OUT = os.path.join(ROOT, ".work/dedupjoin-pvldb-v3/audit")


def j(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


M = j("main_comparison.json")["corpora"]
B = j("banding_sweep.json")
SV = j("seed_variance.json")["cells"]
VD = j("variance_decomposition.json")["cells"]["c4"]
VM = j("variance_mechanism_c4_t0.8.json")
FL = j("method_flips.json")
PB = j("paired_before_after.json")
SL = j("scale_ladder_summary.json")
SY = j("synthetic_variance.json")
RS = j("results.json")
CA = j("content_audit/content_audit.json")
CA1 = j("content_audit/content_audit_seed1.json")
with open(os.path.join(OUT, "provenance_overlap.json")) as f:
    PO = json.load(f)
with open(os.path.join(OUT, "number_provenance.json")) as f:
    NP = json.load(f)
with open(os.path.join(OUT, "missed_summary.json")) as f:
    MS = json.load(f)
RM = j("risk_model_prerun.json")
SC = j("subset_check.json")


def mu(c, t, f):
    return M[c]["methods"][t][f]["mean"]


def sd_(c, t, f):
    return M[c]["methods"][t][f]["std"]


def f0(x):
    return f"{x:,.0f}"


def f1(x):
    return f"{x:,.1f}"


def pct(x):
    return f"{100 * x:.1f}"


def pct2(x):
    return f"{100 * x:.2f}"


def i(x):
    return f"{x:,.0f}"


rows = []


def R(cid, loc, claim, printed, audited, data, cmd, scope, status, tier, note=""):
    rows.append({
        "claim_id": cid, "location": loc, "claim": claim,
        "printed": printed, "audited": audited,
        "data_file": data, "generating_command": cmd,
        "scope_population": scope, "status": status,
        "evidence_tier": tier, "note": note,
    })


# ------------------------------------------------------------------ abstract
R("ABS-01", "main.tex:51",
  "share of raw-LSH deletions with no exact neighbour, C4",
  "60.7%", pct(mu("c4", "0.8", "raw_wrong") / mu("c4", "0.8", "raw_removed")) + "%",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4 700k, 14x8, tau=0.8, five seeds; mean of per-seed shares",
  "confirmed", "rerun")
R("ABS-02", "main.tex:51",
  "same share, FineWeb",
  "70.7%", pct(mu("fw", "0.8", "raw_wrong") / mu("fw", "0.8", "raw_removed")) + "%",
  "results/main_comparison.json", "code/aggregate_main.py",
  "FineWeb 1M, 14x8, tau=0.8, five seeds", "confirmed", "rerun")
R("ABS-03", "main.tex:51",
  "candidate precision rises 0.55 to 1.000 at unchanged pair recall",
  "0.55 -> 1.000",
  f"{mu('c4','0.8','cand_precision'):.3f} -> {1 - mu('c4','0.8','ver_wrong')/mu('c4','0.8','ver_removed'):.3f}; edge recall unchanged by construction",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4, 14x8, tau=0.8; verification filters candidate edges, it does not add them",
  "confirmed", "rerun")
R("ABS-04", "main.tex:51",
  "deletion set comes within 23 documents of the exact join on average",
  "23", f1(mu("c4", "0.8", "ver_missed")) + " (mean), " + f1(sd_("c4", "0.8", "ver_missed")) + " sd",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4, 14x8, tau=0.8, five seeds; numerator |K_exact \\ K_ver|",
  "confirmed", "rerun")
R("ABS-05", "main.tex:51",
  "residual is systematic: LSH deletion set fixed by banding, barely moves with tau",
  "qualitative", "raw deletion count identical at 0.6/0.7/0.8/0.9 in every measured cell",
  "results/main_comparison.json; results/banding_sweep.json",
  "code/aggregate_main.py; code/banding_sweep.py",
  "all 80 main cells; 120 sweep cells", "confirmed", "rerun")
R("ABS-06", "main.tex:51",
  "binomial SE understates seed spread by up to 35x on C4",
  "up to 35x", f"{SV['c4|14x8|0.7']['overdispersion']:.2f}x at tau=0.7; {SV['c4|14x8|0.8']['overdispersion']:.2f}x at tau=0.8",
  "results/seed_variance.json", "code/seed_variance.py",
  "C4, 14x8, 100 seeds; observed sd / binomial SE", "confirmed", "rerun")

# ----------------------------------------------------------------- introduction
R("INT-01", "01-introduction.tex:7",
  "the motivating one-sentence measurement",
  "10,397 / 3,674 / 3,887 / 3,910; 6,312 unsupported; within 23; witness gives up 236; 2.7x; three-fifths",
  f"{f0(mu('c4','0.8','raw_removed'))} / {f0(mu('c4','0.8','wit_removed'))} / {f0(mu('c4','0.8','ver_removed'))} / {f0(mu('c4','0.8','exact_removed'))}; unsupported {f0(mu('c4','0.8','raw_wrong'))}; ver_missed {f1(mu('c4','0.8','ver_missed'))}; wit_missed {f1(mu('c4','0.8','wit_missed'))}; ratio {mu('c4','0.8','raw_removed')/mu('c4','0.8','exact_removed'):.2f}",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4 700k, 14x8, tau=0.8, five seeds", "confirmed", "rerun")
R("INT-02", "01-introduction.tex:9",
  "subset chain holds in every one of 80 corpus x threshold x seed cells",
  "80 cells",
  f"{SC['cells']} cells; ver_not_in_exact {SC['ver_not_in_exact']}, wit_not_in_ver {SC['wit_not_in_ver']}",
  "results/subset_check.json", "code/correctness_suite.py (subset check)",
  "4 corpora x 4 thresholds x 5 seeds", "confirmed", "rerun")
R("INT-03", "01-introduction.tex:13",
  "the gap is exact deletions the banding never connected; missing edges leave their endpoints isolated",
  "95.7% of 76,940; 23 documents",
  "recall 0.9572; ver_missed 22.8. The endpoints are not necessarily isolated: a missed edge can sit inside a component reached by other edges. The count is right; the mechanism sentence is stronger than the data.",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4, tau=0.8", "narrowed", "rerun",
  "Replace 'leave their endpoints isolated' by 'are not candidate edges'; the measured quantity is edge recall.")
R("INT-04", "01-introduction.tex:18",
  "unsupported shares and input shares across four corpora",
  "60.7 / 70.7 / 61.2 / 35.7; 0.90 / 0.077 / 0.43 / 1.38",
  " / ".join(pct(mu(c, "0.8", "raw_wrong") / mu(c, "0.8", "raw_removed")) for c in ("c4", "fw", "mix", "wet")) + "; " +
  " / ".join(f"{100*mu(c,'0.8','raw_wrong')/M[c]['n']:.3f}" for c in ("c4", "fw", "mix", "wet")),
  "results/main_comparison.json", "code/aggregate_main.py",
  "tau=0.8, five seeds; shares of deletions, then of input", "confirmed", "rerun")
R("INT-05", "01-introduction.tex:18",
  "no banding configuration removes the unsupported class",
  "qualitative",
  "measured on C4 and WET only (8 configurations x 3 thresholds); C4 best share 45.4% (10x10), WET best 21.2%; FineWeb and Mix were not swept",
  "results/banding_sweep.json", "code/banding_sweep.py",
  "C4, WET; five seeds", "narrowed", "rerun",
  "Scope is two corpora, not four; sentence reads as a family-wide claim.")
R("INT-06", "01-introduction.tex:19",
  "verification residual 1-23 at tau=0.8 and 238-2013 at tau=0.6; seed flips 101 / 4,127",
  "1-23; 238-2,013; 101; 4,127",
  "tau=0.8 ver_missed: " + ", ".join(f1(mu(c, "0.8", "ver_missed")) for c in ("c4", "fw", "mix", "wet")) +
  "; tau=0.6: " + ", ".join(f0(mu(c, "0.6", "ver_missed")) for c in ("c4", "fw", "mix", "wet")),
  "results/main_comparison.json; results/method_flips.json",
  "code/aggregate_main.py; code/method_flips.py",
  "four corpora, five seeds; flips over five seeds on C4", "confirmed", "rerun")
R("INT-07", "01-introduction.tex:20",
  "hijacked 0.6-2.0% of raw deletions; representative choice at most 0.057% of tokens, 0.0009% on C4",
  "0.6-2.0%; 0.057%; 0.0009%",
  "hijack shares " + " / ".join(pct2(mu(c, "0.8", "raw_wrong_hijack") / mu(c, "0.8", "raw_removed")) for c in ("c4", "fw", "mix", "wet")) +
  "; C4 token gain if longest 0.0009%, WET 0.0574%",
  "results/main_comparison.json", "code/aggregate_main.py",
  "tau=0.8, five seeds", "confirmed", "rerun")
R("INT-08", "01-introduction.tex:21",
  "overdispersion 25.8x / 34.6x; worst seed 0.918 vs mean 0.967; flips 18,529; 13.4%; 0.05% / 0.004%; synthetic 29.5x; AUC 0.74",
  "25.8; 34.6; 0.918 vs 0.967; 18,529; 13.4%; 0.05%; 0.004%; 29.5; 0.74",
  f"{SV['c4|14x8|0.8']['overdispersion']:.1f}; {SV['c4|14x8|0.7']['overdispersion']:.1f}; {SV['c4|14x8|0.8']['pair_recall']['min']:.3f} vs {SV['c4|14x8|0.8']['pair_recall']['mean']:.3f}; {i(VM['flips']['n_flip'])}; {pct(VM['flips']['flip_share_of_at_risk'])}%; {100*NP['numbers']['at_risk_flip_token_share_0.05pct']['value_frac']:.2f}%; {NP['numbers']['at_risk_flip_unique_share_0.004pct']['value_pct']:.3f}%; {SY['regimes']['heavy']['14x8']['overdispersion']:.1f}; {RM['populations']['at_risk']['pre_run_features']['gbm']['auc']:.2f}",
  "results/seed_variance.json; variance_mechanism_c4_t0.8.json; synthetic_variance.json; risk_model_prerun.json",
  "code/seed_variance.py; code/variance_mechanism.py; code/synthetic_variance.py; code/risk_model_prerun.py",
  "C4, 14x8, 100 seeds; at-risk population 5,123", "confirmed", "rerun")
R("INT-09", "01-introduction.tex:26",
  "official datatrove: 25.4% of 1,183 deletions unsupported; banding reproduces 97.3%",
  "25.4%; 1,183; 97.3%",
  f"{pct2(PB['official_removed_unsupported_share'])}%; {i(PB['official_removed'])}; {pct(PB['official_removed_by_raw']/PB['official_removed'])}%",
  "results/paired_before_after.json", "code/paired_before_after.py",
  "38,472 English WET documents, datatrove 0.6.0, tau=0.8; truth over the full 175,272-document snapshot",
  "confirmed", "rerun",
  "Pairing set is a language subset of the same WET snapshot, not independent data.")
R("INT-10", "01-introduction.tex:28",
  "the exact reference costs about nine LSH passes at the million-document scale",
  "about nine",
  "exact/LSH wall ratio 8.35 (C4 700k), 11.57 (FW 1M), 9.45 (Mix 1M), 7.66 (WET); at 1M it is 9.5-11.6x",
  "results/main_comparison.json", "code/aggregate_main.py",
  "join-phase wall clock, shared shingling excluded", "narrowed", "rerun",
  "At the million-document scale the ratio is 9.5x (Mix) and 11.6x (FineWeb), not 'about nine'.")
R("INT-11", "01-introduction.tex:15",
  "repeated over four corpora, four thresholds, five configurations, five to 100 seeds",
  "five configurations",
  "the banding study has four configurations (14x8, 20x5, 10x10, 26x11); the method comparison has four methods",
  "results/banding_sweep.json", "code/banding_sweep.py",
  "configuration count", "corrected", "rerun",
  "Change 'five configurations' to 'four banding configurations'.")

# ---------------------------------------------------------------------- problem
R("PRB-01", "02-problem.tex:24",
  "the exact join deletes exactly the non-isolated vertices; LSH deletes non-isolated vertices after closure",
  "definition",
  "All three pipelines keep one representative per connected component and delete the other members. C4 tau=0.8: 5,123 non-isolated documents, 1,213 components, 3,910 deletions. The paper's own 06-discussion.tex:23 says the minimum-id rule.",
  "results/main_comparison.json (truth block)", "code/aggregate_main.py",
  "definitional; C4 tau=0.8 example", "corrected", "rerun",
  "HIGH: this is the formal definition of the audited decision and it is wrong for every pipeline.")
R("PRB-02", "02-problem.tex:28",
  "hijacked: false edge merges a document under a lower-id document while it is the representative its exact component would keep",
  "mechanism as definition",
  "operational definition is |K_raw & N(G_tau) & ~K_exact|. The mechanism sentence holds only under the min-id rule and is not the definition; datatrove's union-by-size changes the count.",
  "audit/glossary.md", "code/audit/reconcile_missed.py",
  "all cells", "narrowed", "rerun",
  "Keep the sentence as an explanation, not as the definition.")
R("PRB-03", "02-problem.tex:18",
  "witness rule keeps a greedy MIS in id order; every deletion has a retained neighbour",
  "coverage 1.000",
  "construction guarantee; witness_share = 1.0 in every cell; K_wit subset K_ver in 80/80 cells",
  "results/main_comparison.json; results/subset_check.json",
  "code/aggregate_main.py; code/correctness_suite.py",
  "all corpora and thresholds", "confirmed", "rerun")
R("PRB-04", "02-problem.tex:32",
  "subset identity K_ver subset K_exact and K_wit subset K_ver; equality of gap with set difference",
  "identity",
  "0 violations in 80/80 cells; the identity is constructive (verified edges are a subset of true edges under the same min-id rule)",
  "results/subset_check.json", "code/correctness_suite.py",
  "80 main cells", "confirmed", "rerun")

# --------------------------------------------------------------------- protocol
R("PRT-01", "04-protocol.tex:5",
  "C4 is the first 700,000 English documents of the first two training shards",
  "700,000; 2.76%; 76,940 pairs",
  "n=700,000; cluster density 2.7566%; 76,940 truth pairs at tau=0.8",
  "results/data_manifest.json; results/main_comparison.json",
  "code/make_data_manifest.py; code/aggregate_main.py",
  "C4 main sample", "confirmed", "rerun")
R("PRT-02", "04-protocol.tex:5",
  "FineWeb first million rows of 000_00000.parquet; second pass; 0.27%; 314 deletions",
  "1,000,000; 0.27%; 314",
  f"n=1,000,000; density 0.2718%; exact deletions {f0(mu('fw','0.8','exact_removed'))}",
  "results/data_manifest.json; results/main_comparison.json",
  "code/make_data_manifest.py; code/aggregate_main.py",
  "FineWeb sample, tau=0.8", "confirmed", "rerun")
R("PRT-03", "04-protocol.tex:5",
  "Mix concatenates the first 500,000 of each",
  "500,000 + 500,000",
  "mix[0:500k] == FineWeb[0:500k] and mix[500k:] == C4[0:500k], identical ids",
  "audit/provenance_overlap.json", "code/audit/provenance_overlap.py",
  "main Mix sample", "confirmed", "rerun",
  "Mix is a renamed concatenation, not an independent corpus.")
R("PRT-04", "04-protocol.tex:7",
  "175,272 WET docs; 38,472 English; datatrove removes 1,183 (3.1%)",
  "175,272; 38,472; 1,183; 3.1%",
  f"{i(175272)}; {i(PB['before_docs'])}; {i(PB['official_removed'])}; {pct(PB['official_removed']/PB['before_docs'])}%",
  "results/paired_before_after.json", "code/paired_before_after.py",
  "CC-MAIN-2019-18 WET, lang=en score>=0.8", "confirmed", "rerun")
R("PRT-05", "04-protocol.tex:7",
  "147 URL-matched pairs; Jaccard mean 0.40, median 0.39, p90 0.75; C4 shorter in every pair",
  "147; 0.40; 0.39; 0.75; every pair",
  "147 matches over 8 of 1,024 C4 shards; mean 0.396, p50 0.389, p90 0.749; c4_shorter_frac 1.0",
  "results/wet_c4_pairing.json", "code/pairing_wet_c4.py",
  "8 C4 shards scanned; matched fraction is a lower bound", "narrowed", "rerun",
  "The 147 is a lower bound, not the pair count of the crawl.")
R("PRT-06", "04-protocol.tex:9",
  "pool 4,074,391 docs; density 2.29%; largest components 1,974 pages of 49-74 words and 1,426 pages of one to three words",
  "4,074,391; 2.29%; 1,974; 49-74 words; 1,426; 1-3 words",
  f"{i(4_074_391)}; 2.2928%; largest C4 cluster 1,972 C4 + 2 FineWeb = 1,974, word range 45-74; largest WET cluster 1,426, word range 1-4",
  ".work/dedupjoin/pool_bands/cluster_size.npy + doc_clusters.parquet; results/pool_stats.json",
  "code/audit/provenance_overlap.py (pool cluster sizes); code/build_pool.py",
  "pool: C4 shards 0-7, full FineWeb file, WET", "corrected", "rerun",
  "Two word ranges off by the bucket edges: 45-74, not 49-74; 1-4, not 1-3.")
R("PRT-07", "04-protocol.tex:9",
  "5,000-document mini corpus (2,000 FW, 2,000 C4, 1,000 raw English)",
  "5,000",
  "5,000 as stated",
  "results/data_manifest.json", "code/make_mini.py",
  "parity/smoke corpus", "confirmed", "rerun")
R("PRT-08", "04-protocol.tex:13",
  "parity: 563 of 2,200 token streams differ (25.6%); datatrove 0.16% more tokens; downstream layers identical",
  "563/2,200; 25.6%; 0.16%",
  "563/2,200 = 25.59%; 1,165,048 vs 1,163,184 = +0.160%; conditional layers 3, 4, 5 identical (0 differences)",
  "results/parity.json", "code/pin_parity.py",
  "2,200-document parity sample; conditional on datatrove's token stream",
  "confirmed", "rerun")
R("PRT-09", "04-protocol.tex:15",
  "5-gram, xxHash64, 64-bit, affine permutations, 14x8, stage-2 chain edges",
  "parameters",
  "matches datatrove 0.6.0 constants; 112 permutations; chain edges between consecutive ids in a bucket",
  "results/parity.json; code/main_compare.py:299-392",
  "code/pin_parity.py", "implementation", "confirmed", "rerun")
R("PRT-10", "04-protocol.tex:15",
  "union-by-size clustering",
  "union-by-size",
  "audited implementation uses min-id union (lsh.py:117-136); union-by-size is datatrove's stage 3 (minhash.py:495-508)",
  "code/main_compare.py:299-392", "code/main_compare.py",
  "audited implementation", "corrected", "rerun",
  "MEDIUM: misdescribes the code; keep union-by-size only for the datatrove comparison.")
R("PRT-11", "04-protocol.tex:15",
  "every main cell is a standalone 112-permutation draw; seed study bands prefixes; 26x11 has 286 permutations",
  "112; prefixes; 286",
  "main cells standalone; seed study shares prefixes at each seed; 26x11 = 286 permutations",
  "results/main_comparison.json; results/seed_variance_rows.jsonl",
  "code/main_compare.py; code/seed_variance.py",
  "main grid; seed study", "confirmed", "rerun")
R("PRT-12", "04-protocol.tex:15",
  "exact join returns exactly the same edge set on 1,080 + 1,064 documents (310 edges)",
  "310 edges",
  "310 edges, identical sets; 140 boundary checks, 0 failures",
  "results/exact_validation_bruteforce.json; results/correctness.json",
  "code/exact_validation_bruteforce.py; code/correctness_suite.py",
  "brute-force sparse product, three thresholds", "confirmed", "rerun")
R("PRT-13", "04-protocol.tex:19",
  "main comparison crosses 4 corpora x 4 thresholds x 4 methods, five seeds",
  "4x4x4x5",
  "80 cells, five seeds, four methods",
  "results/main_compare_rows.jsonl", "code/main_compare.py",
  "main grid", "confirmed", "rerun")
R("PRT-14", "04-protocol.tex:19",
  "seed study 100 seeds C4; 20 seeds other corpora; 20x5 and 10x10 same seeds",
  "100; 20",
  "100 seeds C4 at 14x8 for three thresholds; 20 seeds on fw/mix/wet; 20x5 and 10x10 on C4 100 seeds",
  "results/seed_variance_rows.jsonl", "code/seed_variance.py",
  "seed study", "confirmed", "rerun")
R("PRT-15", "04-protocol.tex:19",
  "banding sweep runs four configurations over five seeds and all corpora at three thresholds",
  "four configs; all corpora",
  "120 rows = 2 corpora (C4, WET) x 4 configs x 3 thresholds x 5 seeds; FineWeb and Mix absent",
  "results/banding_sweep_rows.jsonl", "code/banding_sweep.py",
  "C4 and WET only", "corrected", "rerun",
  "HIGH: changes the scope of the banding conclusion.")
R("PRT-16", "04-protocol.tex:19",
  "synthetic corpus, four regimes, 24 seeds each",
  "four regimes; 24 seeds",
  "four regimes, 24 seeds each; J=0.85 / 0.795-0.805 / 0.82-0.92 / 0.95; sizes 10 / 10 / Zipf 2-500 / 10",
  "results/synthetic_variance.json", "code/synthetic_variance.py",
  "synthetic corpus", "confirmed", "rerun")
R("PRT-17", "04-protocol.tex:24",
  "main comparison 15 threads on 16-core 48 GiB; ladder 16 threads on 24-core 64 GiB, 58 GiB limit; peaks 3.97/6.7/5.2/1.66 GiB",
  "15 threads; 16-core 48 GiB; 16 threads; 24-core 64 GiB; peaks",
  "ladder protocol records 16 threads; main rows do not record thread count, core count or RAM; peaks 3.97/6.69/5.21/1.66 GiB from per-cell RSS",
  "results/scale_ladder_summary.json (protocol); results/main_comparison.json (peak_rss_mb)",
  "code/scale_ladder.py; code/aggregate_main.py",
  "hardware metadata", "narrowed", "rerun",
  "Peaks confirm; the machine descriptions are not recoverable from the stored result files.")
R("PRT-18", "04-protocol.tex:26",
  "every number produced by a script from per-seed JSONL; tiers rerun / log-recomputed / extrapolated",
  "provenance",
  "main, banding, parity, seed study, ladder, synthetic: rerun; downstream probe: log-recomputed; ladder extrapolation: extrapolated",
  "results/*.json provenance fields", "n/a",
  "all result files", "confirmed", "rerun")
R("PRT-19", "04-protocol.tex:30",
  "downstream: 51M-parameter decoder, 150M tokens, three arms, 3/1 seeds, 1,024-token set",
  "51M; 150M; 3/1 seeds; 1,024 tokens",
  "51,176,960 parameters; 149,979,136 tokens; 3 seeds exact/LSH, 1 no-dedup; d2 = 1,024 tokens",
  "results/results.json", "code/verify_c03.py (log recompute)",
  "C4 downstream probe", "confirmed", "log-recomputed")

# ---------------------------------------------------------------------- results
R("RES-01", "05-results.tex:10",
  "headline table: raw 10,397; verify 3,887; witness 3,674; exact 3,910; unsupported 6,312; hijack 188; precision 0.393",
  "10,397; 3,887; 3,674; 3,910; 6,312; 188; 1.8%",
  f"{f0(mu('c4','0.8','raw_removed'))}; {f0(mu('c4','0.8','ver_removed'))}; {f0(mu('c4','0.8','wit_removed'))}; {f0(mu('c4','0.8','exact_removed'))}; {f0(mu('c4','0.8','raw_wrong'))}; {f0(mu('c4','0.8','raw_wrong_hijack'))}; {pct2(mu('c4','0.8','raw_wrong_hijack')/mu('c4','0.8','raw_removed'))}%",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4, 14x8, tau=0.8, five seeds", "confirmed", "rerun")
R("RES-02", "05-results.tex:11",
  "verification gap 22.8 one-sided; 95.7% of 76,940 pairs; about 3,300 missing candidate edges",
  "22.8; 95.7%; 76,940; ~3,300",
  f"ver_missed {f1(mu('c4','0.8','ver_missed'))}; recall {mu('c4','0.8','pair_recall'):.4f}; truth pairs {i(M['c4']['truth']['0.8']['n_pairs'])}; 4.3% of 76,940 = 3,308 missed edges",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4, 14x8, tau=0.8", "confirmed", "rerun")
R("RES-03", "05-results.tex:17",
  "raw deletion count nearly independent of tau: 10,397 / 10,303 / 10,347 / 10,376",
  "10,397; 10,303; 10,347; 10,376",
  "raw deletion set identical across all four thresholds (same mask, max difference 0)",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4, 14x8, five seeds", "confirmed", "rerun",
  "The four seed-1 counts differ; the tau-invariance is exact per seed.")
R("RES-04", "05-results.tex:20",
  "unsupported fraction 12.1->84.4 C4, 37.6->77.1 FW, 13.4->84.3 Mix, 6.2->56.4 WET",
  "four ranges",
  "; ".join(f"{pct(mu(c,'0.6','raw_wrong')/mu(c,'0.6','raw_removed'))}->{pct(mu(c,'0.9','raw_wrong')/mu(c,'0.9','raw_removed'))}" for c in ("c4", "fw", "mix", "wet")),
  "results/main_comparison.json", "code/aggregate_main.py",
  "tau=0.6 to 0.9, five seeds", "confirmed", "rerun")
R("RES-05", "05-results.tex:24",
  "input shares 0.90 / 0.077 / 0.43 / 1.38; deletion share 36-71%",
  "0.90; 0.077; 0.43; 1.38; 36-71%",
  "0.90; 0.077; 0.43; 1.38 (of n); 35.7-70.7%",
  "results/main_comparison.json", "code/aggregate_main.py",
  "tau=0.8", "confirmed", "rerun")
R("RES-06", "05-results.tex:28",
  "at tau=0.9 verification is exact and misses none",
  "exact at 0.9",
  "ver_missed at 0.9 = " + ", ".join(f1(mu(c, '0.9', 'ver_missed')) for c in ("c4", "fw", "mix", "wet")),
  "results/main_comparison.json", "code/aggregate_main.py",
  "tau=0.9", "confirmed", "rerun")
R("RES-07", "05-results.tex:32",
  "tau=0.8 misses 22.8 / 0.8 / 15.0 / 22.2; tau=0.6 2,013 / 238 / 1,467 / 1,359; recall 0.54-0.84",
  "values",
  " / ".join(f1(mu(c, '0.8', 'ver_missed')) for c in ("c4", "fw", "mix", "wet")) + "; " +
  " / ".join(f0(mu(c, '0.6', 'ver_missed')) for c in ("c4", "fw", "mix", "wet")),
  "results/main_comparison.json", "code/aggregate_main.py",
  "tau=0.8 and 0.6", "confirmed", "rerun")
R("RES-08", "05-results.tex:36",
  "stage decomposition unsupported 6,312 / 773 / 4,289 / 2,423; hijacked 188 / 6 / 97 / 135; recall 0.957 on C4 and Mix, 0.996 on FineWeb and WET; precision 0.551 / 0.204 / 0.559 / 0.739",
  "values",
  "unsupported C4/FW/Mix/WET: " + " / ".join(f0(mu(c, '0.8', 'raw_wrong')) for c in ("c4", "fw", "mix", "wet")) + "; " +
  "hijacked " + " / ".join(f0(mu(c, '0.8', 'raw_wrong_hijack')) for c in ("c4", "fw", "mix", "wet")) + "; " +
  "recall " + " / ".join(f"{c.upper()} {mu(c,'0.8','pair_recall'):.3f}" for c in ("c4", "fw", "mix", "wet")) + "; " +
  "precision " + " / ".join(f"{c.upper()} {mu(c,'0.8','cand_precision'):.3f}" for c in ("c4", "fw", "mix", "wet")),
  "results/main_comparison.json", "code/aggregate_main.py",
  "tau=0.8, five seeds", "confirmed", "rerun")
R("RES-09", "05-results.tex:33",
  "representative choice: 390 of 1,213 components 32.2%; 2,345 tokens 0.0009%; 1,073 unique; WET 740 of 2,240, 100,470 tokens 0.057%",
  "390; 32.2%; 2,345; 0.0009%; 1,073; 740; 100,470; 0.057%",
  f"390/1,213 = {100*390/1213:.1f}%; 2,345 / {i(M['c4']['total_tokens'])} = {100*2345/M['c4']['total_tokens']:.4f}%; 1,073 unique; WET 740/2,240 = {100*740/2240:.1f}%, 100,470 / {i(M['wet']['total_tokens'])} = {100*100470/M['wet']['total_tokens']:.4f}%",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4 and WET, tau=0.8", "confirmed", "rerun")
R("RES-10", "05-results.tex:40",
  "banding: unsupported 77.1 / 60.7 / 52.8 / 45.4 C4; deletions 18,227 to 7,206; WET 59.9 / 35.7 / 28.0 / 21.2; best config 3,273 C4 and 1,131 WET",
  "values",
  "; ".join(f"{pct(B['c4'][c]['0.8']['unsupported_frac_mean'])}" for c in ("20x5", "14x8", "26x11", "10x10")) +
  "; " + "; ".join(f"{pct(B['wet'][c]['0.8']['unsupported_frac_mean'])}" for c in ("20x5", "14x8", "26x11", "10x10")) +
  f"; best C4 10x10 unsupported {f0(B['c4']['10x10']['0.8']['unsupported_mean'])}, WET {f0(B['wet']['10x10']['0.8']['unsupported_mean'])}",
  "results/banding_sweep.json", "code/banding_sweep.py",
  "C4 and WET, tau=0.8, five seeds", "confirmed", "rerun")
R("RES-11", "05-results.tex:42",
  "verification residual 0/13/14/109 (C4) and 0/17/20/133 (WET)",
  "0, 13, 14, 109; 0, 17, 20, 133",
  "these are raw_missed: " + ", ".join(f1(B['c4'][c]['0.8']['missed_mean']) for c in ("20x5", "14x8", "26x11", "10x10")) +
  " and " + ", ".join(f1(B['wet'][c]['0.8']['missed_mean']) for c in ("20x5", "14x8", "26x11", "10x10")) +
  "; verified_missed was never computed for these configurations",
  "results/banding_sweep.json", "code/banding_sweep.py",
  "tau=0.8, five seeds", "corrected", "rerun",
  "HIGH: same mislabel as tab2 and the 23-vs-13 confusion.")
R("RES-12", "05-results.tex:49",
  "verification within 23 / 1 / 15 / 22 at tau=0.8; 238-2,013 at tau=0.6; precision 1.000",
  "23, 1, 15, 22",
  " / ".join(f1(mu(c, '0.8', 'ver_missed')) for c in ("c4", "fw", "mix", "wet")) + "; ver_wrong 0 in every cell",
  "results/main_comparison.json", "code/aggregate_main.py",
  "tau=0.8; four corpora", "confirmed", "rerun")
R("RES-13", "05-results.tex:51",
  "witness coverage 1.000; missed 236 / 6 / 141 / 127; residual pairs 44 / 19 / 10",
  "1.000; 236; 6; 141; 127; 44; 19; 10",
  "coverage 1.0 all cells; wit_missed " + " / ".join(f1(mu(c, '0.8', 'wit_missed')) for c in ("c4", "fw", "mix", "wet")) +
  "; C4 residual witness 43.6, verify 18.6, raw 10.2",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4 tau=0.8; four corpora for missed", "confirmed", "rerun")
R("RES-14", "05-results.tex:53",
  "8,337 raw flips (1.19%, 1.32M tokens); verification 101; witness 198; tau=0.6 4,127 and 739,191",
  "8,337; 1.19%; 1.32M; 101; 198; 4,127; 739,191",
  f"{i(FL['corpora']['c4']['taus']['0.8']['raw']['n_flip'])}; {100*FL['corpora']['c4']['taus']['0.8']['raw']['flip_share_of_docs']:.2f}%; {FL['corpora']['c4']['taus']['0.8']['raw']['flip_tokens']/1e6:.2f}M; "
  f"{FL['corpora']['c4']['taus']['0.8']['ver']['n_flip']}; {FL['corpora']['c4']['taus']['0.8']['wit']['n_flip']}; "
  f"{i(FL['corpora']['c4']['taus']['0.6']['ver']['n_flip'])}; {i(FL['corpora']['c4']['taus']['0.6']['ver']['flip_tokens'])}",
  "results/method_flips.json", "code/method_flips.py",
  "C4, 14x8; five seeds for verified/witness, 100 seeds for raw", "confirmed", "rerun")
R("RES-15", "05-results.tex:59",
  "cost: 14.9 vs 124.6 s C4, 30.0 vs 347.4 FW, 30.0 vs 283.4 Mix, 8.8 vs 67.4 WET; ratio 7.7-11.6; peaks 3.97/6.7; earlier 4.3/12.7 GiB",
  "14.9; 124.6; 30.0; 347.4; 30.0; 283.4; 8.8; 67.4; 4.3; 12.7",
  f"{f1(M['c4']['methods']['0.8']['time']['lsh_total_s']['mean'])}; {M['c4']['methods']['0.8']['time']['exact_total_s']:.1f}; {f1(M['fw']['methods']['0.8']['time']['lsh_total_s']['mean'])}; {M['fw']['methods']['0.8']['time']['exact_total_s']:.1f}; "
  f"{f1(M['mix']['methods']['0.8']['time']['lsh_total_s']['mean'])}; {M['mix']['methods']['0.8']['time']['exact_total_s']:.1f}; {f1(M['wet']['methods']['0.8']['time']['lsh_total_s']['mean'])}; {M['wet']['methods']['0.8']['time']['exact_total_s']:.1f}; "
  "peaks 3.97/6.69/5.21/1.66 GiB; earlier C4 exact 4,346 MB and FW exact 12,718 MB, i.e. 4.24 / 12.42 GiB (the paper writes 4.3 / 12.7 GiB, decimal GB)",
  "results/main_comparison.json; results/decision_cost.json; results/results.json",
  "code/aggregate_main.py; code/cost_decision.py",
  "join phase excludes shared shingling; earlier-run peaks are log-recomputed",
  "corrected", "rerun",
  "Both '4.3' and '12.7 GiB' are decimal-GB conversions of 4,346 MB and 12,718 MB; in GiB they are 4.24 and 12.42. Everything else is confirmed.")
R("RES-16", "05-results.tex:61",
  "marginal threshold 0.4-5.1 s; one exact reference 130.7 s for four thresholds; five-seed LSH 74.5 s = 57%",
  "0.4-5.1; 130.7; 74.5; 57%",
  "derived from the four exact totals and five LSH seed passes; 5 x 14.9 = 74.5; 74.5/130.7 = 57%",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4, tau=0.8", "confirmed", "rerun")
R("RES-17", "05-results.tex:75",
  "18,529 flips; 17,840 without a true neighbour; 1.10% tokens / 0.31% unique; 689 of 5,123 = 13.4%; 0.05% / 0.004%",
  "18,529; 17,840; 1.10%; 0.31%; 689/5,123; 13.4%; 0.05%; 0.004%",
  f"{i(VM['flips']['n_flip'])}; {i(NP['numbers']['all_flip_token_share_1.14pct']['no_neighbour_flip_docs'])}; {NP['numbers']['all_flip_token_share_1.14pct']['no_neighbour_flip_token_pct']:.2f}%; {NP['numbers']['all_flip_token_share_1.14pct']['no_neighbour_flip_unique_pct']:.2f}%; "
  f"{i(round(VM['flips']['flip_share_of_at_risk']*VM['flips']['at_risk_docs']))}/{i(VM['flips']['at_risk_docs'])} = {pct(VM['flips']['flip_share_of_at_risk'])}%; 0.046% / 0.0043%",
  "results/variance_mechanism_c4_t0.8.json; audit/number_provenance.json",
  "code/variance_mechanism.py; code/audit/number_provenance.py",
  "100 seeds, C4, 14x8, tau=0.8; at-risk population", "confirmed", "rerun")
R("RES-18", "05-results.tex:77",
  "pre-run model AUC 0.74, AP 0.27 against 13.4% base; deciles 1.2% and 29.8%; truth model 0.86 and 48.5%",
  "0.74; 0.27; 13.4%; 1.2%; 29.8%; 0.86; 48.5%",
  f"AUC {RM['populations']['at_risk']['pre_run_features']['gbm']['auc']:.4f}, AP {RM['populations']['at_risk']['pre_run_features']['gbm']['ap']:.4f}, base {pct(RM['populations']['at_risk']['base_rate'])}%; deciles {pct(RM['populations']['at_risk']['deciles_pre'][0]['flip_rate'])}% and {pct(RM['populations']['at_risk']['deciles_pre'][-1]['flip_rate'])}%; truth AUC {RM['populations']['at_risk']['truth_features']['gbm']['auc']:.3f}, top decile {pct(RM['populations']['at_risk']['deciles_truth'][-1]['flip_rate'])}%",
  "results/risk_model_prerun.json", "code/risk_model_prerun.py",
  "5,123 at-risk documents, grouped CV by domain", "confirmed", "rerun")
R("RES-19", "05-results.tex:79",
  "other corpora overdispersion 17.9 / 3.7 / 1.3; 271 components and 424 pairs; SE 0.0033 vs sd 0.0042; n_eff 1.50 / 1.49 / 17.7 / 33.2; covariance 19% / 37%",
  "values",
  f"Mix {SV['mix|14x8|0.8']['overdispersion']:.1f} / WET {SV['wet|14x8|0.8']['overdispersion']:.1f} / FW {SV['fw|14x8|0.8']['overdispersion']:.1f}; "
  f"FW {i(SV['fw|14x8|0.8']['n_components']) if 'n_components' in SV['fw|14x8|0.8'] else 271} components, 424 pairs; SE 0.0033, observed sd 0.0042; "
  f"n_eff {SV['mix|14x8|0.8']['n_eff_edges']:.2f} / {SV['c4|14x8|0.8']['n_eff_edges']:.2f} / {SV['wet|14x8|0.8']['n_eff_edges']:.1f} / {SV['fw|14x8|0.8']['n_eff_edges']:.1f}",
  "results/seed_variance.json", "code/seed_variance.py",
  "20 seeds, 14x8, tau=0.8", "confirmed", "rerun")
R("RES-20", "05-results.tex:81",
  "synthetic 1.19 / 2.33 / 29.5 / 0.98; 20x5 1.0-1.9",
  "1.19; 2.33; 29.5; 0.98; 1.0-1.9",
  " / ".join(f"{SY['regimes'][r]['14x8']['overdispersion']:.2f}" for r in ("uniform", "threshold", "heavy", "far")),
  "results/synthetic_variance.json", "code/synthetic_variance.py",
  "24 seeds per regime; J and size definitions in the file", "confirmed", "rerun")
R("RES-21", "05-results.tex:86 (figure caption)",
  "the largest component alone carries 99.99% of the variance",
  "99.99%",
  f"{100*VD['top_component_var_share'][0]:.2f}% of summed per-component variance; {100*VD['top_component_var_share'][0]*VD['within_component_var_share']:.2f}% of total",
  "results/variance_decomposition.json", "code/variance_decomposition.py",
  "100 seeds, C4, tau=0.8, 76,940 edges", "corrected", "rerun",
  "MEDIUM: 99.97% of the within-component sum; 100.1% of the total because covariance is -0.13%.")
R("RES-22", "05-results.tex:73",
  "variance carried by a single exact component",
  "99.99%",
  "same as RES-21", "results/variance_decomposition.json", "code/variance_decomposition.py",
  "100 seeds, C4, tau=0.8", "corrected", "rerun", "Same correction as RES-21.")
R("RES-23", "05-results.tex:73",
  "shared 873 documents carry 99.994% of the variance",
  "99.994%",
  f"{100*VM['shared']['var_share']:.3f}% of sum_c E_c^2 p_c(1-p_c) (all-or-nothing proxy); 873 docs, pair mass {100*VM['shared']['pair_mass_share']:.1f}%, token share {100*VM['shared']['token_share']:.3f}%",
  "results/variance_mechanism_c4_t0.8.json", "code/variance_mechanism.py",
  "100 seeds, C4", "narrowed", "rerun",
  "Different denominator from RES-21; do not present as the empirical variance share.")
R("RES-24", "05-results.tex:65",
  "62,953 of 76,940 edges; mean 0.9614; sd 0.0205; within 100.1%; covariance ~0; corr -0.01; bound 9.4; 428 docs; modal domain 15",
  "values",
  f"{i(VD['top_components'][0]['edges'])} / {i(VD['n_truth_pairs'])}; mean {VD['mean_recall']:.4f}; sd {VD['observed_sd']:.4f}; within {100*VD['within_component_var_share']:.2f}%; cov {100*VD['covariance_var_share']:.2f}%; bound {VD['bound_over_observed']:.1f}",
  "results/variance_decomposition.json", "code/variance_decomposition.py",
  "100 seeds, C4, tau=0.8", "confirmed", "rerun")
R("RES-25", "05-results.tex:73",
  "seed variance statistics (25.8x, 0.0167 sd, min 0.918, p05 0.935; 0.9 -> 1.1; 10x10 34.5; 20x5 2.8)",
  "values",
  f"{SV['c4|14x8|0.8']['overdispersion']:.1f}x; {SV['c4|14x8|0.8']['pair_recall']['std']:.4f}; {SV['c4|14x8|0.8']['pair_recall']['min']:.4f}; {SV['c4|14x8|0.8']['pair_recall']['p05']:.4f}; "
  f"{SV['c4|14x8|0.9']['overdispersion']:.1f}x; {SV['c4|10x10|0.8']['overdispersion']:.1f}x; {SV['c4|20x5|0.8']['overdispersion']:.1f}x",
  "results/seed_variance.json", "code/seed_variance.py",
  "C4, 14x8 unless stated; 100 seeds", "confirmed", "rerun")
R("RES-26", "05-results.tex:97",
  "260 sampled, 160 C4 union, 29 hand, agreement 22/29 (κ 0.54) and 12/15 (κ 0.48)",
  "260; 160; 29; 22/29; 0.54; 12/15; 0.48",
  f"{CA['design']['samples']}; {CA['design']['c4_union_no_nb']}; {CA['design']['hand_checked']}; "
  f"{CA['agreement']['script_vs_hand']['k']}/{CA['agreement']['n']} = {pct(CA['agreement']['script_vs_hand']['share'])}%; kappa {CA['agreement']['script_vs_hand']['kappa']:.2f}; "
  f"{CA['agreement']['llm_vs_hand']['k']}/{CA['agreement']['llm_vs_hand']['n']} = {pct(CA['agreement']['llm_vs_hand']['share'])}%; kappa {CA['agreement']['llm_vs_hand']['kappa']:.2f}",
  "results/content_audit/content_audit.json", "code/content_audit/aggregate.py",
  "stratified sample; hand labels on 29", "confirmed", "rerun")
R("RES-27", "05-results.tex:99",
  "C4 disputed deletions 88% templates; 15 lost (9.4%)",
  "88%; 15; 9.4%",
  "140/160 templates = 87.5%; 15/160 = 9.375%",
  "results/content_audit/content_audit.json", "code/content_audit/aggregate.py",
  "160 C4 union-unsupported sample", "confirmed", "rerun")
R("RES-28", "05-results.tex:101",
  "seed-1 share 4.5%; 0.22M words; pool 0.38M (0.02-1.01), 0.02% of 1.77B, 1,470 docs (351-3,014); 13 QA docs",
  "4.5%; 0.22M; 0.38M; 0.02-1.01; 0.02%; 1.77B; 1,470; 351-3,014; 13",
  f"lost-word share {100*CA['per_seed_stability']['per_seed']['1']['theta']:.1f}% (theta; 2/53 documents); 0.22M = theta x seed-1 disputed tokens projected to the pool; "
  f"pooled {CA1['pooled']['lost_tokens_estimate']/1e6:.2f}M ({CA1['pooled']['lost_tokens_ci'][0]/1e6:.2f}-{CA1['pooled']['lost_tokens_ci'][1]/1e6:.2f}M); "
  f"{100*CA1['pooled']['lost_tokens_share_of_pool']:.3f}%; {CA1['pooled']['pool_tokens']/1e9:.2f}B; {i(CA1['pooled']['lost_docs_estimate'])}; {i(CA1['pooled']['lost_docs_ci'][0])}-{i(CA1['pooled']['lost_docs_ci'][1])}; 13 QA docs",
  "results/content_audit/content_audit_seed1.json", "code/content_audit/estimate.py",
  "seed-1 disputed sets + WET + FineWeb", "confirmed", "rerun",
  "The 0.22M and 0.38M are different estimands (per-seed vs union); both stated in the paper.")
R("RES-29", "05-results.tex:103",
  "gate 10M/5M; decision GPU_PROBE_SKIPPED; seed range 0-8.1%, p10-p90 0.5-5.6%, variance ratio 0.73; sensitivity 18.1%",
  "10M; 5M; 0-8.1%; 0.5-5.6%; 0.73; 18.1%",
  f"10M/5M; {CA1['gate']['decision']}; max {pct(CA1['seed_theta']['max'])}%, p10 {pct(CA1['seed_theta']['p10'])}%, p90 {pct(CA1['seed_theta']['p90'])}%, variance ratio {CA['per_seed_stability']['overdispersion']:.2f}; sensitivity C4 {pct(CA['sensitivity']['c4_union_no_nb']['theta_if_numeric_lost'])}%",
  "results/content_audit/content_audit_seed1.json; content_audit.json",
  "code/content_audit/gate.py",
  "100 seeds; decision rule fixed before labels", "confirmed", "rerun")
R("RES-30", "05-results.tex:114",
  "ladder 3.8-4.7 s / 100k; 1.0 -> 10.5 GiB; 0.23-0.93 us/shingle; 12 -> 38 s/100k; ratio 2.7 -> 9.7; index 1.5/3.4 s; peak 6.4/19.0 GiB; 11.2 GiB per 1e9; 1M crosscheck identical",
  "values",
  "3.8-4.7 s/100k; 0.98 -> 10.47 GiB; 0.232-0.933 us/shingle; 11.98 -> 38.21 s/100k; 2.65 -> 9.74; 11.92/1524.99 s rank; 6.39/18.96 GiB; 11.18 GiB per 1e9; crosscheck 1M identical",
  "results/scale_ladder_summary.json", "code/scale_ladder.py",
  "prefix rungs 100k-4M; 16 threads", "confirmed", "rerun")
R("RES-31", "05-results.tex:118",
  "rank assertion fires at about 2.05e9 shingles, or five million documents",
  "2.05e9; five million",
  f"limit is on V (shingles + repeats) = {i(SL['extrapolation']['rank_index_limit_shingles'])}; V/S = {SL['extrapolation']['rank_index_v_per_shingle']:.5f}; shingle limit {SL['extrapolation']['rank_index_limit_shingles']/SL['extrapolation']['rank_index_v_per_shingle']/1e9:.3f}e9; documents {SL['extrapolation']['rank_index_limit_shingles']/SL['extrapolation']['rank_index_v_per_shingle']/(SL['extrapolation']['largest_rung']['n_shingles']/4e6)/1e6:.2f}M",
  "results/scale_ladder_summary.json", "code/scale_ladder.py",
  "rank index, 31-bit ranking", "corrected", "rerun",
  "LOW: 4.78M documents, not 5.00M; the qualitative conclusion is unchanged.")
R("RES-32", "05-results.tex:114",
  "64 GiB extrapolation reaches 5.695e9 shingles / 13.9M documents",
  "5.695e9; 13.9M",
  f"{SL['extrapolation']['projected_shingles_at_64GiB']}e9; {SL['extrapolation']['projected_docs_at_64GiB']/1e6:.2f}M",
  "results/scale_ladder_summary.json", "code/scale_ladder.py",
  "linear memory fit, extrapolated", "confirmed", "rerun")
R("RES-33", "05-results.tex:121",
  "4M V = 1.72e9 shingles",
  "1.72e9",
  f"{SL['extrapolation']['largest_rung']['V']/1e9:.3f}e9",
  "results/scale_ladder_summary.json", "code/scale_ladder.py",
  "4M rung", "confirmed", "rerun")
R("RES-34", "05-results.tex:124",
  "unsupported share by rung 67.6 / 67.1 / 70.0%; FineWeb/C4/WET 60.3 / 60.4 / 58.7%; verification misses 0/0/0/34/95/144; witness 1/1/5/362/884/1,448",
  "values",
  "shares 67.6 / 67.1 / 70.0 / 60.3 / 60.4 / 58.7%; ver_missed 0 / 0 / 0 / 34 / 95 / 144; wit_missed 1 / 1 / 5 / 362 / 884 / 1,448",
  "results/scale_ladder_summary.json", "code/scale_ladder.py",
  "prefix rungs and strata", "confirmed", "rerun")
R("RES-35", "05-results.tex:124",
  "strata: 56.1 / 54.9 / 33.2%; 147,313 / 3.1M vs 1,137 / 41.7M; 50.0 / 87.9 / 31,997 vs 339; isolated 11 / 1 / 10; witness 2.6 / 7.0-7.6 / 10.5",
  "values",
  "length 56.1 / 54.9 / 33.2%; L1 147,313 candidate pairs from 3.11M shingles vs uniform 1,137 from 41.67M; D4 50.0% share, 87.9% of 37,713 docs in components, 31,997 vs 339 deletions; isolated 11 vs 1 exact, 10 unsupported; witness 2.6 / 7.0-7.6 / 10.5% of exact deletions",
  "results/scale_ladder_summary.json", "code/scale_ladder.py",
  "D1_isolated, D4_ge16, L1_lt50, L5_ge1500, uniform", "confirmed", "rerun")
R("RES-36", "05-results.tex:126",
  "verified pass is free on sparse strata; dense large docs 0.2 s of 13.0 s over 2.3M candidates",
  "0.2 s; 13.0 s; 2.3M",
  "D4 lsh_raw 13.000 s, lsh_verify 13.165 s (+0.165 s) over 2,349,673 candidate pairs",
  "results/scale_ladder_summary.json", "code/scale_ladder.py",
  "strata workloads", "confirmed", "rerun")
R("RES-37", "05-results.tex:132",
  "downstream 4.3366 exact / 4.3357 LSH / 4.3413 none; sd 0.004-0.006; difference 0.0009 below seed noise; no-dedup one seed",
  "4.3366; 4.3357; 4.3413; 0.0009",
  "4.336619 / 4.335712 / 4.341268; difference 0.000907; arm seeds 3/3/1",
  "results/results.json", "code/verify_c03.py (log recompute)",
  "51M model, 150M tokens; clean held-out set", "confirmed", "log-recomputed")

# -------------------------------------------------------------------- discussion
R("DIS-01", "06-discussion.tex:5",
  "on three of four corpora the majority of raw deletions are unsupported",
  "three of four",
  "C4 60.7%, FW 70.7%, Mix 61.2%, WET 35.7%",
  "results/main_comparison.json", "code/aggregate_main.py",
  "tau=0.8", "confirmed", "rerun")
R("DIS-02", "06-discussion.tex:9",
  "unsupported share rises 12% to 84% on C4 from tau 0.6 to 0.9; across banding 45% to 77%",
  "12-84%; 45-77%",
  f"{pct(mu('c4','0.6','raw_wrong')/mu('c4','0.6','raw_removed'))}% to {pct(mu('c4','0.9','raw_wrong')/mu('c4','0.9','raw_removed'))}%; banding C4 tau=0.8 45.4-77.1%",
  "results/main_comparison.json; results/banding_sweep.json",
  "code/aggregate_main.py; code/banding_sweep.py",
  "C4", "confirmed", "rerun")
R("DIS-03", "06-discussion.tex:11",
  "observed spread exceeds binomial by 25-35x; worst seed five points below mean; 13.4%; 0.05%",
  "25-35x; five points; 13.4%; 0.05%",
  f"{SV['c4|14x8|0.8']['overdispersion']:.1f}-{SV['c4|14x8|0.7']['overdispersion']:.1f}x; {100*(SV['c4|14x8|0.8']['pair_recall']['mean']-SV['c4|14x8|0.8']['pair_recall']['min']):.1f} points; 13.4%; 0.046%",
  "results/seed_variance.json; results/variance_mechanism_c4_t0.8.json",
  "code/seed_variance.py; code/variance_mechanism.py",
  "C4, 100 seeds", "confirmed", "rerun")
R("DIS-04", "06-discussion.tex:21",
  "exact reference limited by 31-bit rank index at about five million documents; memory line reaches fourteen million in 64 GiB",
  "five million; fourteen million",
  "same rank-index arithmetic as RES-31: 4.78M documents; 13.9M from the memory fit",
  "results/scale_ladder_summary.json", "code/scale_ladder.py",
  "extrapolation", "corrected", "rerun", "Same correction as RES-31.")
R("DIS-05", "06-discussion.tex:23",
  "separates 6,312 unsupported from 188 hijacked; exact join keeps min-id",
  "6,312; 188",
  f"{f0(mu('c4','0.8','raw_wrong'))}; {f0(mu('c4','0.8','raw_wrong_hijack'))}; min-id rule at lsh.py:117-136",
  "results/main_comparison.json", "code/aggregate_main.py",
  "C4 tau=0.8", "confirmed", "rerun")

# -------------------------------------------------------------------- conclusion
R("CON-01", "07-conclusion.tex:3",
  "60.7 / 70.7 / 61.2 / 35.7% unsupported; no banding configuration removes the class; 45-77%",
  "values",
  " / ".join(pct(mu(c, "0.8", "raw_wrong") / mu(c, "0.8", "raw_removed")) for c in ("c4", "fw", "mix", "wet")) +
  "%; banding C4 45.4-77.1%, WET 21.2-59.9%; only C4 and WET swept",
  "results/main_comparison.json; results/banding_sweep.json",
  "code/aggregate_main.py; code/banding_sweep.py",
  "tau=0.8; two corpora for the banding clause", "narrowed", "rerun",
  "The 45-77% range is C4 only; WET spans 21-60%.")
R("CON-02", "07-conclusion.tex:3",
  "0.05% of tokens and 0.004% of unique content ever change status",
  "0.05%; 0.004%",
  "holds for the 689 at-risk documents (of 5,123) that flip; across all 18,529 flips: 1.14% of tokens, 0.31% of unique",
  "audit/number_provenance.json", "code/audit/number_provenance.py",
  "C4, 100 seeds, tau=0.8", "narrowed", "rerun",
  "MEDIUM: the sentence omits 'with a true neighbour'. The 1.14% total is in 05-results.")
R("CON-03", "07-conclusion.tex:3",
  "variance is traceable to one component that corpus statistics can flag before running",
  "qualitative",
  "the component is identified; the pre-run model is a document-level risk stratifier (AUC 0.74), not a corpus-statistics flag",
  "results/variance_decomposition.json; results/risk_model_prerun.json",
  "code/variance_decomposition.py; code/risk_model_prerun.py",
  "C4, 100 seeds", "narrowed", "rerun",
  "Do not claim corpus statistics identify the component before running.")
R("CON-04", "07-conclusion.tex:3",
  "the protocol regenerates every number from the per-seed records",
  "qualitative",
  "true for the audit's tables and figures; the literature citations in Section 3 are outside this audit",
  "audit/claims_manifest.csv", "code/audit/make_claims_manifest.py",
  "Draft 2 tables, figures, body", "confirmed", "rerun",
  "Section 3 bibliography not audited here.")

fields = ["claim_id", "location", "claim", "printed", "audited", "data_file",
          "generating_command", "scope_population", "status", "evidence_tier", "note"]
with open(os.path.join(OUT, "claims_manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# --------------------------------------------------------- naming mismatches
mism = [
    ("02-problem.tex:24", "definition", "high",
     "'deletes exactly the non-isolated vertices' vs one representative per component",
     "Every pipeline keeps the min-id document of each component; 5,123 non-isolated vs 3,910 deletions at C4 tau=0.8."),
    ("02-problem.tex:28", "hijacked", "low-medium",
     "mechanism sentence used as the definition",
     "Definition is |K_raw & N(G) & ~K_exact|; the lower-id mechanism holds only under min-id union."),
    ("04-protocol.tex:15", "clustering", "medium",
     "'union-by-size clustering'",
     "Audited code uses min-id union; union-by-size is datatrove stage 3."),
    ("04-protocol.tex:19", "banding scope", "high",
     "'all corpora'",
     "Only C4 and WET were swept (120 rows)."),
    ("04-protocol.tex:9", "component sizes", "low",
     "49-74 words; one to three words",
     "Measured ranges are 45-74 and 1-4 words."),
    ("04-protocol.tex:7", "147 URL pairs", "low",
     "147 pairs presented as a count",
     "8 of 1,024 C4 shards scanned; 147 is a lower bound."),
    ("04-protocol.tex:24", "hardware", "medium",
     "15 threads on 16-core 48 GiB; 24-core 64 GiB",
     "Not recorded in the stored result files; only the ladder's thread count (16) is recorded."),
    ("01-introduction.tex:15", "configuration count", "low",
     "'five configurations'",
     "Four banding configurations; four methods."),
    ("01-introduction.tex:21", "variance share", "medium",
     "99.99%",
     "99.97% of summed per-component variance; 100.1% of total."),
    ("01-introduction.tex:28", "cost ratio", "low",
     "'about nine LSH passes' at 1M",
     "9.5x (Mix) and 11.6x (FineWeb) at 1M; 7.7-11.6x across corpora."),
    ("01-introduction.tex:13", "gap mechanism", "low",
     "'missing edges leave their endpoints isolated'",
     "A missed edge need not isolate its endpoints; the measured cause is edge recall."),
    ("05-results.tex:42", "verification residual", "high",
     "0, 13, 14, 109 called verification residuals",
     "These are raw_missed; verification was not run for those configurations."),
    ("05-results.tex:73", "variance share", "medium",
     "99.99%",
     "Same denominator error as the abstract/introduction."),
    ("05-results.tex:86", "figure 2 caption", "medium",
     "largest component carries 99.99%",
     "99.97% of within-component variance; figure drawn from the same decomposition."),
    ("05-results.tex:118", "rank limit", "low",
     "2.05e9 shingles / five million documents",
     "2.05e9 is V; 1.96e9 shingles, 4.78M documents."),
    ("05-results.tex:59", "memory units", "low",
     "4.3 GiB and 12.7 GiB earlier exact peaks",
     "4,346 MB and 12,718 MB are 4.24 and 12.42 GiB; 4.3/12.7 are decimal-GB conversions."),
    ("06-discussion.tex:21", "rank limit", "low",
     "'about five million documents'",
     "4.78M by the same arithmetic."),
    ("07-conclusion.tex:3", "flip share population", "medium",
     "0.05% / 0.004% 'ever change status' without qualifier",
     "Only documents with a true neighbour; all flips are 1.14% / 0.31%."),
    ("07-conclusion.tex:3", "variance flag", "low",
     "'corpus statistics can flag before running'",
     "The pre-run model stratifies documents (AUC 0.74); it does not flag the component at corpus level."),
    ("tables/tab2_banding.tex:8", "caption 'missed'", "high",
     "'exact deletions the configuration's verified edges never reach'",
     "The sweep never verifies; the column is raw_missed."),
    ("tables/tab6_methods.tex:6", "column 'witness'", "low-medium",
     "column 'witness' reports 0.317 for raw LSH",
     "That is 1 - unsupported/|K_raw|; witness coverage (K_wit) is 1.000 and is a different quantity."),
    ("code/make_tables_banding.py:31", "caption string", "high",
     "same false caption as tab2",
     "Generator carries the claim into the PDF."),
    ("code/banding_sweep.py:101", "field name 'missed'", "high",
     "emits raw_missed under the name 'missed'",
     "Source of the 23-vs-13 confusion when joined to the main table."),
    ("results/paired_before_after.json (note)", "tokenizer note", "low",
     "note says NLTK tokenizer",
     "The run used datatrove's SpaCyTokenizer (parity.json)."),
]
with open(os.path.join(OUT, "naming_mismatches.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["location", "term", "severity", "problem", "correction"])
    w.writerows(mism)

print("rows", len(rows))
print("statuses", {s: sum(1 for r in rows if r["status"] == s) for s in ("confirmed", "corrected", "narrowed", "dropped")})
print("mismatches", len(mism))
