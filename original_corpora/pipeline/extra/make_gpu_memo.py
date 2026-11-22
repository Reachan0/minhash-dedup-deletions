#!/usr/bin/env python3
"""Inputs for the GPU / semantic-evidence decision memo.

Reads the existing content audit and Draft 2 downstream-probe records; computes
Wilson intervals, the annotation sample size for a target half-width, and the number
of independent seeds a downstream training probe would need.  Writes
`extra/gpu_decision.json`.  No new labels are produced here.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EXTRA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "extra")
AUDIT = os.path.join(ROOT, "output/dedupjoin-pvldb/results/content_audit/content_audit_seed1.json")
RESULTS = os.path.join(ROOT, "output/dedupjoin-pvldb/results/results.json")
OUT = os.path.join(EXTRA, "gpu_decision.json")
Z = 1.959963984540054


def wilson(k, n, z=Z):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(p=p, lo=max(0.0, mid - half), hi=min(1.0, mid + half), k=k, n=n)


def n_for_halfwidth(p, e, z=Z):
    return math.ceil(z * z * p * (1 - p) / (e * e))


def seeds_for_effect(d, sigma, alpha=0.05, power=0.8):
    za = 1.959963984540054
    zb = 0.8416212335729143
    return math.ceil(2 * (za + zb) ** 2 * (sigma / d) ** 2)


def main():
    a = json.load(open(AUDIT))
    r = json.load(open(RESULTS))
    c4 = a["c4"]
    n53, k2 = 53, 2
    w = wilson(k2, n53)
    arm = r["downstream_lm"]["arms"]
    ex = arm["exact"]["eval"]["val_clean"]
    ls = arm["lsh"]["eval"]["val_clean"]
    d_probe = abs(ex["mean_loss"] - ls["mean_loss"])
    sigma = math.sqrt((ex["std_loss"] ** 2 + ls["std_loss"] ** 2) / 2)
    lost_c4 = c4["lost_tokens"]                    # token-weighted estimate, seed-1
    lost_pool = a["pooled"]["lost_tokens_estimate"]
    manipulated_words = 1227237.0                 # disputed tokens removed by LSH only vs exact
    out = dict(
        generated_utc=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        sources=dict(content_audit=os.path.relpath(AUDIT, ROOT),
                     downstream_probe=os.path.relpath(RESULTS, ROOT)),
        annotation=dict(
            model_labels_n=n53, model_labels_k=k2, model_label_kind="model (qwen3.5-flash), NOT human",
            wilson95=w,
            halfwidth_targets={str(e): n_for_halfwidth(w["p"], e) for e in (0.02, 0.025, 0.03, 0.05)},
            hand_labels_n=29, hand_labels_note="QC subset, not a random sample of the 53; model-hand 12/15 agree",
            script_labels_note="rule labels judge template/copy, not partial rewrite vs independent"),
        downstream=dict(
            probe_arm_tokens=arm["exact"]["tokens_trained"], eval_tokens=ex["tokens"], probe_seeds=3,
            exact_mean=ex["mean_loss"], exact_sd=ex["std_loss"],
            lsh_mean=ls["mean_loss"], lsh_sd=ls["std_loss"],
            delta=d_probe, pooled_seed_sd=sigma, delta_over_sigma=d_probe / sigma,
            manipulated_words_lsh_minus_exact=manipulated_words,
            lost_words_c4_estimate=lost_c4, lost_words_pool_estimate=lost_pool,
            ratio_c4=lost_c4 / manipulated_words, ratio_pool=lost_pool / manipulated_words,
            expected_delta_c4=d_probe * lost_c4 / manipulated_words,
            expected_delta_pool=d_probe * lost_pool / manipulated_words,
            seeds_needed=dict(
                observed_probe_delta=seeds_for_effect(d_probe, sigma),
                scaled_c4=seeds_for_effect(d_probe * lost_c4 / manipulated_words, sigma),
                scaled_pool=seeds_for_effect(d_probe * lost_pool / manipulated_words, sigma),
            ),
            tokens_per_seed=arm["exact"]["tokens_trained"],
        ),
        decision="no-go for a GPU/semantic arm in this round",
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
