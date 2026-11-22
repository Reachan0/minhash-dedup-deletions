#!/usr/bin/env python3
"""Aggregate the content-audit labels, estimate population quantities, evaluate the gate.

Inputs (all under output/dedupjoin-pvldb/):
  results/content_audit/samples.jsonl
  results/content_audit/llm_labels.jsonl          (primary labeller when present)
  results/content_audit/hand_labels.jsonl         (researcher's hand check)
  .work/dedupjoin/main/c4_meta.parquet            (token totals of the C4 sample)
  .work/dedupjoin/main/wet_meta.parquet
  .work/dedupjoin/rerun/fw_1000000_meta.parquet

Outputs:
  results/content_audit/content_audit.json
  artifacts/blind_review/agreement.json

The lost-information class is {unique_facts, unique_content}; every count is reported as
a fraction of the stratum and as extrapolated documents/tokens with a bootstrap interval.
The primary label is the LLM label when the outlet produced one, otherwise the deterministic
script label; the labeller is recorded per row and in the header so the degradation is visible.
"""
import json
import os
import re
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CA = os.path.join(ROOT, "output/dedupjoin-pvldb/results/content_audit")
BR = os.path.join(ROOT, "output/dedupjoin-pvldb/artifacts/blind_review")
MAIN = os.path.join(ROOT, ".work/dedupjoin/main")
RERUN = os.path.join(ROOT, ".work/dedupjoin/rerun")
LOST = {"unique_facts", "unique_content"}
LABELS = ["template", "repost", "partial_rewrite", "unique_facts", "unique_content"]


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


def tokens_of(meta_path, sel):
    t = pq.read_table(meta_path, columns=["nwords"])
    w = t.column("nwords").to_numpy().astype(np.float64)
    return float(w[sel].sum()), float(w.sum()), w


WORD = re.compile(r"[a-z0-9$%.]+")
NUM = re.compile(r"\b\d[\d,.]{2,}\b")


def _words(s):
    return WORD.findall((s or "").lower())


def script_label(r):
    """Deterministic labeller, used as fallback and as an agreement baseline."""
    text = r.get("text") or ""
    toks = _words(text)
    nw = len(toks)
    alpha = sum(1 for t in toks[:400] if re.search(r"[a-z]", t)) / max(1, min(nw, 400))
    if nw < 50 and (alpha < 0.65 or nw < 12):
        return "template", dict(nwords=nw, alpha_frac=round(alpha, 3))
    mj = r.get("max_j", 0.0) or 0.0
    if mj >= 0.95:
        return "repost", dict(max_j=mj)
    if mj >= 0.4 and (r.get("neighbor") or {}).get("text"):
        nb_t = _words(r["neighbor"]["text"])
        s_nb = set(nb_t)
        uniq = [t for t in toks if t not in s_nb]
        uf = len(uniq) / max(1, nw)
        new_nums = sorted(set(NUM.findall(text)) - set(NUM.findall(r["neighbor"]["text"])))
        if uf <= 0.18:
            return "template", dict(max_j=mj, uniq_frac=round(uf, 3))
        if len(new_nums) >= 2:
            return "unique_facts", dict(max_j=mj, uniq_frac=round(uf, 3),
                                        new_numbers=new_nums[:8])
        if uf >= 0.55:
            return "unique_content", dict(max_j=mj, uniq_frac=round(uf, 3))
        return "partial_rewrite", dict(max_j=mj, uniq_frac=round(uf, 3))
    if mj < 0.4:
        return "unique_content", dict(max_j=mj)
    return "unique_content", dict(max_j=mj)


def main():
    samples = {r["id"]: r for r in load_jsonl(os.path.join(CA, "samples.jsonl"))}
    llm = {r["id"]: r for r in load_jsonl(os.path.join(CA, "llm_labels.jsonl"))}
    hand = {r["id"]: r for r in load_jsonl(os.path.join(CA, "hand_labels.jsonl"))}
    rows = []
    for did, r in samples.items():
        lab = llm.get(did, {}).get("label")
        if lab not in LABELS:
            lab = None
        sl, feats = script_label(r)
        rows.append(dict(id=did, corpus=r["corpus"], stratum=r["stratum"],
                         idx=r["index"], nwords=r.get("n_words_text") or r["nwords"],
                         llm=lab, script=sl, primary=lab or sl,
                         labeller="llm" if lab else "script",
                         conf=llm.get(did, {}).get("confidence"),
                         qa=llm.get(did, {}).get("qa"),
                         lost_fact=llm.get(did, {}).get("lost_fact"),
                         script_feats=feats,
                         has_nb=r.get("has_nb"), max_j=r.get("max_j"),
                         seed_membership=r.get("removed_by_raw_seed")))
    labelled = [r for r in rows if r["primary"]]
    n_llm = sum(1 for r in rows if r["labeller"] == "llm")
    print(f"{len(rows)} sampled, {len(labelled)} labelled ({n_llm} by LLM), {len(hand)} hand-checked")

    def stratum_stats(rs, key="primary"):
        rs = [r for r in rs if r[key]]
        n = len(rs)
        toks = np.array([r["nwords"] for r in rs], float)
        out = {}
        for lab in LABELS:
            m = np.array([r[key] == lab for r in rs])
            k = int(m.sum())
            out[lab] = dict(n=k, share=k / n if n else None, ci=wilson(k, n),
                            tokens=int(toks[m].sum()))
        m = np.array([r[key] in LOST for r in rs])
        ts = float(toks[m].sum() / toks.sum()) if toks.sum() else None
        ts_ci = None
        if n >= 5 and toks.sum() > 0:
            rng = np.random.default_rng(11)
            v = np.empty(2000)
            for b in range(2000):
                i = rng.integers(0, n, n)
                v[b] = toks[i][m[i]].sum() / max(1.0, toks[i].sum())
            ts_ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out["lost_information"] = dict(
            n=int(m.sum()), share=float(m.mean()) if n else None, ci=wilson(int(m.sum()), n),
            tokens=int(toks[m].sum()), token_share=ts, token_share_ci=ts_ci)
        out["n"] = n
        out["tokens_total"] = int(toks.sum())
        return out

    strata = {}
    for s in sorted({r["stratum"] for r in rows}):
        ss = [r for r in rows if r["stratum"] == s]
        strata[s] = dict(primary=stratum_stats(ss), script=stratum_stats(ss, "script"))
        if any(r["llm"] for r in ss):
            strata[s]["llm"] = stratum_stats(ss, "llm")

    # per-seed theta over the 100 seeds: sample is uniform from the 100-seed union
    per_seed = {}
    c4 = [r for r in rows if r["stratum"] == "c4_union_no_nb" and r["primary"]]
    for seed in range(1, 101):
        rs = [r for r in c4 if seed in (r["seed_membership"] or [])]
        if not rs:
            continue
        toks = np.array([r["nwords"] for r in rs], float)
        m = np.array([r["primary"] in LOST for r in rs])
        per_seed[seed] = dict(n=len(rs), theta=float(toks[m].sum() / toks.sum()),
                              k=int(m.sum()), tokens=int(toks.sum()))
    th = np.array([v["theta"] for v in per_seed.values()])
    ns = np.array([v["n"] for v in per_seed.values()])
    pooled = float(np.mean(th))
    samp_var = float(np.mean(pooled * (1 - pooled) / np.maximum(ns, 1)))
    obs_var = float(np.var(th, ddof=1))
    stability = dict(n_seeds=len(per_seed), theta_mean=pooled,
                     theta_sd=float(np.std(th, ddof=1)),
                     theta_p10=float(np.percentile(th, 10)),
                     theta_p90=float(np.percentile(th, 90)),
                     theta_min=float(th.min()), theta_max=float(th.max()),
                     sampling_var=samp_var, observed_var=obs_var,
                     overdispersion=float(obs_var / samp_var) if samp_var > 0 else None,
                     seeds_below_half_mean=int((th < pooled / 2).sum()),
                     per_seed=per_seed)

    # population extrapolation
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from make_samples import c4_arrays, N_C4, N_FW, N_WET  # noqa

    A = c4_arrays()
    ex = A["ex"]
    raw1 = A["raw1"]
    D1 = raw1 & ~ex
    U100 = np.zeros(N_C4, bool)
    for b in A["var"].values():
        U100 |= b
    U100_noNB = U100 & ~A["has"]
    c4meta = os.path.join(MAIN, "c4_meta.parquet")
    wetmeta = os.path.join(MAIN, "wet_meta.parquet")
    fwmeta = os.path.join(RERUN, "fw_1000000_meta.parquet")
    c4_tokens_D, c4_tokens_all, c4w = tokens_of(c4meta, D1)
    # the sample is drawn from the unsupported part of D_s only; the upper extrapolation
    # applies its theta to every disputed token, the lower one only to unsupported tokens
    D1_noNB = D1 & ~A["has"]
    c4_tokens_D_noNB, _, _ = tokens_of(c4meta, D1_noNB)
    c4_pool_docs = 2850538
    c4_scale_docs = c4_pool_docs / N_C4
    z = np.load(os.path.join(MAIN, "wet_s1_t0.8_sets.npz"))
    wraw = np.unpackbits(z["rem_raw"], bitorder="big")[:N_WET].astype(bool)
    wex = np.unpackbits(z["rem_exact"], bitorder="big")[:N_WET].astype(bool)
    Dw = wraw & ~wex
    wet_tokens_D, wet_tokens_all, wetw = tokens_of(wetmeta, Dw)
    z = np.load(os.path.join(MAIN, "fw_s1_t0.8_sets.npz"))
    fraw = np.unpackbits(z["rem_raw"], bitorder="big")[:N_FW].astype(bool)
    fex = np.unpackbits(z["rem_exact"], bitorder="big")[:N_FW].astype(bool)
    Df = fraw & ~fex
    fw_tokens_D, fw_tokens_all, fww = tokens_of(fwmeta, Df)

    def theta_for(stratum):
        rs = [r for r in labelled if r["stratum"] == stratum]
        toks = np.array([r["nwords"] for r in rs], float)
        m = np.array([r["primary"] in LOST for r in rs])
        return (float(toks[m].sum() / toks.sum()) if toks.sum() else 0.0), rs, m, toks

    def boot_tokens(stratum, pop_tokens, B=4000):
        rs = [r for r in labelled if r["stratum"] == stratum]
        if not rs:
            return None
        toks = np.array([r["nwords"] for r in rs], float)
        m = np.array([r["primary"] in LOST for r in rs])
        rng = np.random.default_rng(7)
        vals = np.empty(B)
        for b in range(B):
            i = rng.integers(0, len(rs), len(rs))
            vals[b] = toks[i][m[i]].sum() / max(1.0, toks[i].sum()) * pop_tokens
        return dict(estimate=float((m * toks).sum()) / toks.sum() * pop_tokens,
                    ci=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))])

    # sensitivity: the LLM applies the slot rule aggressively, so count a document lost also
    # when it carries >=2 numbers absent from its counterpart (the hand-checked stat-page case)
    def numeric_only(r):
        nb = samples[r["id"]].get("neighbor") or {}
        if not nb.get("text"):
            return False
        return len(set(NUM.findall(samples[r["id"]].get("text") or ""))
                   - set(NUM.findall(nb["text"]))) >= 2

    sensitivity = {}
    for stratum in ("c4_union_no_nb", "wet_disputed", "fw_disputed"):
        rs = [r for r in labelled if r["stratum"] == stratum]
        if not rs:
            continue
        toks = np.array([r["nwords"] for r in rs], float)
        m0 = np.array([r["primary"] in LOST for r in rs])
        m1 = np.array([r["primary"] in LOST or numeric_only(r) for r in rs])
        sensitivity[stratum] = dict(
            theta=float(toks[m0].sum() / toks.sum()),
            theta_if_numeric_lost=float(toks[m1].sum() / toks.sum()),
            extra_docs=int((m1 & ~m0).sum()))

    th_c4, rs_c4, m_c4, t_c4 = theta_for("c4_union_no_nb")
    th_wet, rs_wet, m_wet, t_wet = theta_for("wet_disputed")
    th_fw, rs_fw, m_fw, t_fw = theta_for("fw_disputed")

    c4_pop_tokens = c4_tokens_D * c4_scale_docs
    fw_pop_tokens = fw_tokens_D * (1048581 / N_FW)
    wet_pop_tokens = wet_tokens_D
    pool_tokens = (int(c4_tokens_all * c4_scale_docs) + int(wet_tokens_all)
                   + int(fw_tokens_all * (1048581 / N_FW)))
    boot_c4 = boot_tokens("c4_union_no_nb", c4_pop_tokens)
    boot_c4_noNB = boot_tokens("c4_union_no_nb", c4_tokens_D_noNB * c4_scale_docs)
    boot_fw = boot_tokens("fw_disputed", fw_pop_tokens)
    boot_wet = boot_tokens("wet_disputed", wet_pop_tokens)
    pooled_est = sum(x["estimate"] for x in (boot_c4, boot_fw, boot_wet) if x)
    pooled_lo = sum(x["ci"][0] for x in (boot_c4, boot_fw, boot_wet) if x)
    pooled_hi = sum(x["ci"][1] for x in (boot_c4, boot_fw, boot_wet) if x)
    pooled_lower = sum(x["estimate"] for x in (boot_c4_noNB, boot_fw, boot_wet) if x)
    pooled_lower_ci = sum(x["ci"][0] for x in (boot_c4_noNB, boot_fw, boot_wet) if x)

    # probe material
    lost_rows = [r for r in labelled if r["primary"] in LOST]
    with_qa = [r for r in lost_rows if r.get("qa") and (r["qa"] or {}).get("question")]
    nonrec = [r for r in lost_rows if (r.get("max_j") or 0) < 0.5 or r["primary"] == "unique_facts"]
    g4 = len(nonrec) / len(lost_rows) if lost_rows else 0.0

    # agreement on hand-checked subset
    def conf_matrix(a, b, rec):
        return {f"{x}->{y}": sum(1 for r in rec if r[a] == x and r[b] == y)
                for x in LABELS for y in LABELS}

    agree = None
    if hand:
        ids = [i for i in hand if i in samples]
        rec = [r for r in rows if r["id"] in ids]
        n = len(rec)
        def kappa(x, y, recs=None):
            recs = rec if recs is None else recs
            m = len(recs)
            po = sum(1 for r in recs if x(r) == y(r)) / m if m else 0
            pe = sum((sum(1 for r in recs if x(r) == c) / m) *
                     (sum(1 for r in recs if y(r) == c) / m) for c in LABELS) if m else 0
            return (po - pe) / (1 - pe) if pe < 1 else None
        hl = lambda r: hand[r["id"]]["label"]
        script_hand = sum(1 for r in rec if r["script"] == hl(r))
        agree = dict(n=n,
                     script_vs_hand=dict(k=script_hand, share=script_hand / n if n else None,
                                         ci=wilson(script_hand, n), kappa=kappa(lambda r: r["script"], hl)),
                     confusion_script_hand=conf_matrix("script", "hand_label", [
                         dict(r, hand_label=hl(r)) for r in rec]),
                     hand_label_counts={c: sum(1 for r in rec if hl(r) == c) for c in LABELS})
        rec2 = [r for r in rec if r["llm"]]
        if rec2:
            n2 = len(rec2)
            lh = sum(1 for r in rec2 if r["llm"] == hl(r))
            ls = sum(1 for r in rec2 if r["llm"] == r["script"])
            agree["llm_vs_hand"] = dict(n=n2, k=lh, share=lh / n2, ci=wilson(lh, n2),
                                        kappa=kappa(lambda r: r["llm"], hl, rec2))
            agree["llm_vs_script"] = dict(n=n2, k=ls, share=ls / n2, ci=wilson(ls, n2),
                                          kappa=kappa(lambda r: r["llm"], lambda r: r["script"], rec2))
            agree["confusion_llm_hand"] = conf_matrix("llm", "hand_label", [
                dict(r, hand_label=hl(r)) for r in rec2])

    def lost_docs_pool(stratum, pop_docs):
        rs = [r for r in labelled if r["stratum"] == stratum]
        if not rs:
            return 0.0
        return sum(1 for r in rs if r["primary"] in LOST) / len(rs) * pop_docs

    def boot_docs(stratum, pop_docs, B=4000):
        rs = [r for r in labelled if r["stratum"] == stratum]
        if not rs:
            return dict(estimate=0.0, ci=[0.0, 0.0])
        m = np.array([r["primary"] in LOST for r in rs])
        rng = np.random.default_rng(13)
        v = np.empty(B)
        for b in range(B):
            i = rng.integers(0, len(rs), len(rs))
            v[b] = m[i].mean() * pop_docs
        return dict(estimate=float(m.mean() * pop_docs),
                    ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    pop_docs = {"c4_union_no_nb": D1.sum() * c4_scale_docs,
                "wet_disputed": float(Dw.sum()),
                "fw_disputed": Df.sum() * (1048581 / N_FW)}
    lost_docs_pool = sum(lost_docs_pool(s, d) for s, d in pop_docs.items())
    lost_docs_boot = {s: boot_docs(s, d) for s, d in pop_docs.items()}
    lost_docs_ci = [sum(v["ci"][0] for v in lost_docs_boot.values()),
                    sum(v["ci"][1] for v in lost_docs_boot.values())]

    gate = dict(
        G1_magnitude=dict(pass_=bool(pooled_lo >= 5e6 and pooled_est >= 1e7),
                          pooled_lost_tokens_estimate=pooled_est,
                          pooled_ci=[pooled_lo, pooled_hi],
                          criterion="estimate >= 10M and CI lower >= 5M"),
        G2_material=dict(pass_=bool(len(with_qa) >= 50 and lost_docs_pool >= 2000),
                         lost_docs_sampled=len(lost_rows), with_qa=len(with_qa),
                         lost_docs_pool_estimate=lost_docs_pool,
                         criterion=">=50 sampled lost docs with a self-contained QA; extrapolated lost docs >=2000"),
        G3_stability=dict(pass_=bool(stability["overdispersion"] is not None
                                     and stability["overdispersion"] <= 2.0
                                     and stability["seeds_below_half_mean"] == 0),
                          overdispersion=stability["overdispersion"],
                          seeds_below_half_mean=stability["seeds_below_half_mean"],
                          criterion="overdispersion <= 2 and no seed below half the pooled theta"),
        G4_nonrecoverability=dict(pass_=bool(g4 >= 0.70), share=float(g4),
                                  criterion=">=70% of lost docs have max J < 0.5 or are unique_facts"),
    )
    passed = all(v["pass_"] for v in gate.values())
    gate["decision"] = "PROBE_DESIGNED" if passed else "GPU_PROBE_SKIPPED"

    out = dict(
        version=1, generated_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        provenance="independently rerun (this run): decisions from stored per-seed records; labels from LLM + script + hand check",
        labeller_summary=dict(llm=n_llm, script=len(rows) - n_llm,
                              by_stratum={s: dict(llm=sum(1 for r in rows if r["stratum"] == s
                                                          and r["labeller"] == "llm"),
                                                  script=sum(1 for r in rows if r["stratum"] == s
                                                             and r["labeller"] == "script"))
                                          for s in sorted({r["stratum"] for r in rows})}),
        model="qwen3.5-flash (platform outlet, temperature 0, thinking disabled)",
        design=dict(samples=len(samples), labelled=len(labelled), hand_checked=len(hand),
                    c4_union_no_nb=int(len([r for r in rows if r["stratum"] == "c4_union_no_nb"])),
                    population=dict(pool_docs=4074391, pool_tokens=pool_tokens,
                                    c4_union_100_seeds=int(U100.sum()),
                                    c4_union_100_seeds_no_nb=int(U100_noNB.sum()),
                                    c4_disputed_docs_seed1=int(D1.sum()),
                                    c4_disputed_tokens_seed1=c4_tokens_D,
                                    wet_disputed_docs=int(Dw.sum()), wet_disputed_tokens=wet_tokens_D,
                                    fw_disputed_docs=int(Df.sum()), fw_disputed_tokens=fw_tokens_D)),
        strata=strata,
        per_seed_stability=stability,
        population_estimates=dict(
            theta_c4_prefix=th_c4, theta_wet=th_wet, theta_fw=th_fw,
            c4=dict(disputed_tokens_prefix=c4_tokens_D, pool_scale_docs=c4_scale_docs,
                    disputed_tokens_pool=c4_pop_tokens, lost_tokens_boot=boot_c4,
                    disputed_tokens_prefix_no_nb=c4_tokens_D_noNB,
                    disputed_tokens_pool_no_nb=c4_tokens_D_noNB * c4_scale_docs,
                    lost_tokens_boot_no_nb=boot_c4_noNB),
            fw=dict(disputed_tokens=fw_tokens_D, disputed_tokens_pool=fw_pop_tokens,
                    lost_tokens_boot=boot_fw),
            wet=dict(disputed_tokens=wet_tokens_D, lost_tokens_boot=boot_wet),
            pooled=dict(lost_tokens_estimate=pooled_est, ci=[pooled_lo, pooled_hi],
                        lost_tokens_share_of_pool=pooled_est / pool_tokens,
                        lost_docs_estimate=lost_docs_pool, lost_docs_ci=lost_docs_ci,
                        lost_docs_by_stratum=lost_docs_boot,
                        lost_tokens_lower=pooled_lower,
                        lower_ci=[pooled_lower_ci, pooled_hi],
                        note="upper applies theta to every disputed token; lower applies it only to unsupported (no exact neighbour) tokens")),
        sensitivity=sensitivity,
        probe_material=dict(lost_docs=len(lost_rows), with_qa=len(with_qa),
                            distinct_facts=[r["lost_fact"] for r in lost_rows if r.get("lost_fact")][:200]),
        agreement=agree,
        gate=gate)
    with open(os.path.join(CA, "content_audit.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.makedirs(BR, exist_ok=True)
    with open(os.path.join(BR, "agreement.json"), "w") as f:
        json.dump(dict(provenance=out["provenance"], labeller_summary=out["labeller_summary"],
                       agreement=agree), f, indent=2, ensure_ascii=False)
    print(json.dumps(dict(gate=gate, stability={k: v for k, v in stability.items() if k != "per_seed"},
                          pooled=out["population_estimates"]["pooled"],
                          strata={s: v["primary"]["lost_information"] for s, v in strata.items()},
                          agreement=agree), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
