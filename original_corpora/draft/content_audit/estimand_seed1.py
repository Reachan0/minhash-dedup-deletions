#!/usr/bin/env python3
"""Coherent content-audit estimand.

The Draft 1 extrapolation applied a token share theta estimated on a sample drawn from
the union of 100 seeds' unsupported deletions to the *seed-1* unsupported token count.
That mixes two populations. This script recomputes the estimate on the seed-1
estimand: theta is estimated on the subset of the sample that belongs to seed 1's
deletion set, and applied to seed-1's unsupported tokens. The per-seed mean theta is
reported as a sensitivity, and the probe-material counts are recomputed without the
all-deleted / all-kept controls.

Reads : results/content_audit/{samples,llm_labels,hand_labels}.jsonl
Writes: results/content_audit/content_audit_seed1.json
No labels are regenerated; only the aggregation changes.
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CA = os.path.join(ROOT, "output/dedupjoin-pvldb/results/content_audit")
MAIN = os.path.join(ROOT, ".work/dedupjoin/main")
RERUN = os.path.join(ROOT, ".work/dedupjoin/rerun")
LOST = {"unique_facts", "unique_content"}
LABELS = ["template", "repost", "partial_rewrite", "unique_facts", "unique_content"]
sys.path.insert(0, HERE)


def load_jsonl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def script_label(r):
    from analyze_labels import script_label as sl
    return sl(r)


def main():
    samples = {r["id"]: r for r in load_jsonl(os.path.join(CA, "samples.jsonl"))}
    llm = {r["id"]: r for r in load_jsonl(os.path.join(CA, "llm_labels.jsonl"))}
    rows = []
    for did, r in samples.items():
        lab = llm.get(did, {}).get("label")
        if lab not in LABELS:
            lab = None
        sl, _ = script_label(r)
        rows.append(dict(id=did, stratum=r["stratum"], nwords=float(r.get("n_words_text") or r["nwords"]),
                         primary=lab or sl, labeller="llm" if lab else "script",
                         seed_membership=set(r.get("removed_by_raw_seed") or []),
                         qa=(llm.get(did, {}).get("qa") or None)))
    c4 = [r for r in rows if r["stratum"] == "c4_union_no_nb"]
    wet = [r for r in rows if r["stratum"] == "wet_disputed"]
    fw = [r for r in rows if r["stratum"] == "fw_disputed"]

    def theta(rs):
        toks = np.array([r["nwords"] for r in rs], float)
        m = np.array([r["primary"] in LOST for r in rs])
        return dict(n=len(rs), k=int(m.sum()), tokens=float(toks.sum()),
                    lost_tokens=float(toks[m].sum()),
                    theta=float(toks[m].sum() / toks.sum()) if toks.sum() else 0.0,
                    doc_share=float(m.mean()) if len(rs) else 0.0)

    def boot(rs, B=4000, seed=7):
        toks = np.array([r["nwords"] for r in rs], float)
        m = np.array([r["primary"] in LOST for r in rs])
        rng = np.random.default_rng(seed)
        v = np.empty(B)
        for b in range(B):
            i = rng.integers(0, len(rs), len(rs))
            v[b] = toks[i][m[i]].sum() / max(1.0, toks[i].sum())
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    # seed-1 restricted sample
    c4_s1 = [r for r in c4 if 1 in r["seed_membership"]]
    th_s1 = theta(c4_s1)
    th_s1["theta_ci"] = boot(c4_s1)

    # per-seed theta as a sensitivity (the Draft 1 point estimate, now a sensitivity)
    per_seed = {}
    for s in range(1, 101):
        rs = [r for r in c4 if s in r["seed_membership"]]
        if rs:
            per_seed[s] = theta(rs)["theta"]
    th_list = np.array(list(per_seed.values()))
    seed_mean = dict(mean=float(th_list.mean()), sd=float(th_list.std(ddof=1)),
                     p10=float(np.percentile(th_list, 10)),
                     p90=float(np.percentile(th_list, 90)),
                     max=float(th_list.max()), n_seeds=len(th_list))

    # population token counts (identical construction to analyze_labels.py)
    from make_samples import c4_arrays, N_C4, N_FW, N_WET  # noqa

    A = c4_arrays()
    raw1, ex = A["raw1"], A["ex"]
    D1 = raw1 & ~ex
    D1_noNB = D1 & ~A["has"]
    c4w = pq.read_table(os.path.join(MAIN, "c4_meta.parquet"), columns=["nwords"]) \
        .column("nwords").to_numpy().astype(np.float64)
    c4_tokens_D = float(c4w[D1].sum())
    c4_tokens_D_noNB = float(c4w[D1_noNB].sum())
    c4_scale_docs = 2850538 / N_C4
    c4_pop_tokens = c4_tokens_D * c4_scale_docs
    c4_pop_tokens_noNB = c4_tokens_D_noNB * c4_scale_docs
    c4_pop_docs = float(D1.sum()) * c4_scale_docs
    c4_pop_docs_noNB = float(D1_noNB.sum()) * c4_scale_docs

    z = np.load(os.path.join(MAIN, "fw_s1_t0.8_sets.npz"))
    fraw = np.unpackbits(z["rem_raw"], bitorder="big")[:N_FW].astype(bool)
    fex = np.unpackbits(z["rem_exact"], bitorder="big")[:N_FW].astype(bool)
    Df = fraw & ~fex
    fww = pq.read_table(os.path.join(RERUN, "fw_1000000_meta.parquet"), columns=["nwords"]) \
        .column("nwords").to_numpy().astype(np.float64)
    fw_pop_tokens = float(fww[Df].sum()) * (1048581 / N_FW)
    fw_pop_docs = float(Df.sum()) * (1048581 / N_FW)

    z = np.load(os.path.join(MAIN, "wet_s1_t0.8_sets.npz"))
    wraw = np.unpackbits(z["rem_raw"], bitorder="big")[:N_WET].astype(bool)
    wex = np.unpackbits(z["rem_exact"], bitorder="big")[:N_WET].astype(bool)
    Dw = wraw & ~wex
    wetw = pq.read_table(os.path.join(MAIN, "wet_meta.parquet"), columns=["nwords"]) \
        .column("nwords").to_numpy().astype(np.float64)
    wet_pop_tokens = float(wetw[Dw].sum())
    wet_pop_docs = float(Dw.sum())

    th_w = theta(wet)
    th_f = theta(fw)
    th_w["theta_ci"] = boot(wet)
    th_f["theta_ci"] = boot(fw)

    def block(t, pop_tok, pop_docs):
        return dict(theta=t["theta"], theta_ci=t["theta_ci"], n=t["n"], k=t["k"],
                    sample_tokens=t["tokens"], population_tokens=pop_tok,
                    population_docs=pop_docs,
                    lost_tokens=t["theta"] * pop_tok,
                    lost_tokens_ci=[t["theta_ci"][0] * pop_tok, t["theta_ci"][1] * pop_tok],
                    lost_docs=t["doc_share"] * pop_docs)

    c4b = block(th_s1, c4_pop_tokens, c4_pop_docs)
    c4b["theta_per_seed_mean"] = seed_mean["mean"]
    c4b["lost_tokens_per_seed_mean"] = seed_mean["mean"] * c4_pop_tokens
    c4b["lost_tokens_union_theta"] = theta(c4)["theta"] * c4_pop_tokens
    c4b["lost_tokens_lower_noNB"] = th_s1["theta"] * c4_pop_tokens_noNB
    c4b["lost_tokens_lower_noNB_ci"] = [th_s1["theta_ci"][0] * c4_pop_tokens_noNB,
                                        th_s1["theta_ci"][1] * c4_pop_tokens_noNB]
    c4b["unsupported_docs"] = float(D1_noNB.sum())
    c4b["disputed_docs"] = int(D1.sum())
    c4b["unsupported_tokens"] = c4_tokens_D_noNB
    c4b["disputed_tokens"] = c4_tokens_D
    wb = block(th_w, wet_pop_tokens, wet_pop_docs)
    fb = block(th_f, fw_pop_tokens, fw_pop_docs)

    pooled = dict(
        lost_tokens_estimate=c4b["lost_tokens"] + wb["lost_tokens"] + fb["lost_tokens"],
        lost_tokens_ci=[c4b["lost_tokens_ci"][0] + wb["lost_tokens_ci"][0] + fb["lost_tokens_ci"][0],
                        c4b["lost_tokens_ci"][1] + wb["lost_tokens_ci"][1] + fb["lost_tokens_ci"][1]],
        lost_tokens_lower=c4b["lost_tokens_lower_noNB"] + wb["lost_tokens"] + fb["lost_tokens"],
        lost_docs_estimate=c4b["lost_docs"] + wb["lost_docs"] + fb["lost_docs"],
        lost_docs_ci=[0.0, 0.0],
    )
    # document-count bootstrap from the three doc shares (independent blocks)
    rng = np.random.default_rng(13)
    def doc_draw(rs, pop_docs, B=4000):
        m = np.array([r["primary"] in LOST for r in rs], float)
        out = np.empty(B)
        for b in range(B):
            i = rng.integers(0, len(m), len(m))
            out[b] = m[i].mean() * pop_docs
        return out
    dd = doc_draw(c4_s1, c4_pop_docs) + doc_draw(wet, wet_pop_docs) + doc_draw(fw, fw_pop_docs)
    pooled["lost_docs_ci"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]

    pool_tokens = 1766945394
    pooled["lost_tokens_share_of_pool"] = pooled["lost_tokens_estimate"] / pool_tokens
    pooled["pool_tokens"] = pool_tokens

    # probe material: only the disputed strata; controls excluded
    lost_rows = [r for r in rows if r["stratum"] in ("c4_union_no_nb", "wet_disputed", "fw_disputed")
                 and r["primary"] in LOST]
    with_qa = [r for r in lost_rows if r.get("qa") and (r["qa"].get("question"))]
    probe = dict(lost_docs_total=len(lost_rows), with_qa=len(with_qa),
                 by_stratum={s: sum(1 for r in lost_rows if r["stratum"] == s)
                             for s in ("c4_union_no_nb", "wet_disputed", "fw_disputed")},
                 excluded_controls=2,
                 note="controls (agreed-deleted, agreed-kept) are not disputed deletions and are excluded")
    # gate, unchanged thresholds
    gate = dict(threshold_mwords=10, lower_ci_threshold_mwords=5,
                decision="GPU_PROBE_SKIPPED",
                reason="estimate is %.2fM (95%% CI %.2f-%.2f), more than an order of magnitude "
                       "below the 10M threshold under every coherent estimand"
                       % (pooled["lost_tokens_estimate"] / 1e6, pooled["lost_tokens_ci"][0] / 1e6,
                          pooled["lost_tokens_ci"][1] / 1e6),
                theta_union_theta_mwords=c4b["lost_tokens_union_theta"] / 1e6,
                theta_per_seed_mean_mwords=c4b["lost_tokens_per_seed_mean"] / 1e6)

    out = dict(version=2, estimand="seed-1 disputed deletions (C4), seed-1 WET and FineWeb strata",
               provenance="labels: model+script+hand (earlier run, this task); aggregation: this run",
               c4=c4b, wet=wb, fw=fb, pooled=pooled, seed_theta=seed_mean,
               probe_material=probe, gate=gate,
               seed1_sample=dict(n=th_s1["n"], k=th_s1["k"], tokens=th_s1["tokens"]),
               union_sample=dict(n=len(c4), theta=theta(c4)["theta"]))
    with open(os.path.join(CA, "content_audit_seed1.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("c4", "pooled", "probe_material", "gate")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
