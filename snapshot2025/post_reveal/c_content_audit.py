#!/usr/bin/env python3
"""C: exploratory content audit of path A's deletions on CC-MAIN-2025-30 with two LLM labellers.

Population (seed 1, tau 0.8, min-id A, documents outside the 86,771-copy template group):
  unsupported deletions (no neighbour at J >= 0.8), supported deletions, and retained documents.
Sample: 1,200 / 400 / 400 documents, stratified by word-count quintile within each stratum
(proportional allocation, seed 31); weights N_h / n_h are stored with each row.
Counterpart: the member of the document's A component with the highest exact Jaccard to it
(for retained documents, the best candidate among its seed-1 bucket co-members); its text is
shown when that Jaccard is >= 0.4, as in the v3 audit.
Labels: the v3 prompt (dedupjoin-pvldb/code/content_audit/llm_label.py), sent to two models from
different vendors through the gateway. Every prompt and raw response is stored.
Kept variant: `kept` re-pairs every sampled deletion with the document A retains from its component;
run label/analyze with C_VARIANT=kept to label and summarise that sample.
Usage: env/bin/python c_content_audit.py sample | kept | label <model> | analyze
"""
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(PKG, ".."))
BASE = os.path.join(PKG, "results_post_reveal", "content_audit")
# C_VARIANT=kept points label/analyze at the re-paired sample written by `kept`
OUT = os.path.join(BASE, os.environ["C_VARIANT"]) if os.environ.get("C_VARIANT") else BASE
sys.path[:0] = [os.path.join(PKG, "code", "newsnap"), os.path.join(PKG, "code", "run"),
                os.path.join(ROOT, "dedupjoin-pvldb", "code", "content_audit")]
# Any OpenAI-compatible chat-completions endpoint that serves the two labeller models.
GATEWAY = os.environ.get("LLM_GATEWAY", "")
PROXY = os.environ.get("LLM_PROXY", "")
SIZES = {"unsupported": 1200, "supported": 400, "retained": 400}


def prompt_parts():
    src = open(os.path.join(ROOT, "dedupjoin-pvldb", "code", "content_audit", "llm_label.py")).read()
    ns = {}
    start = src.index("SYSTEM = (")
    end = src.index("\ndef ", start)
    exec(src[start:end], ns)  # SYSTEM, INSTRUCTIONS and helpers defined as plain constants in v3
    return ns["SYSTEM"], ns["INSTRUCTIONS"]


def sample():
    import collapsed_lib as CL
    import pyarrow.parquet as pq
    S, K = CL.S, CL.K
    z = CL.load_collapsed()
    group, m = z["group_of_doc"], z["group_size"]
    n = len(group)
    off, flat, nw = S.load_snapshot()
    ti, tj, _ = CL.truth_u(0.8)
    st = CL.exact_state(z, ti, tj)
    with np.load(os.path.join(S.CACHE, "masks", "raw_s1.npz")) as f:
        root, rem = f["root"], f["rem"]
    big = int(np.argmax(m))
    outside = (group != big) & (np.diff(off) > 0)
    strata = {"unsupported": np.flatnonzero(rem & ~st["has_nb_doc"] & outside),
              "supported": np.flatnonzero(rem & st["has_nb_doc"] & outside),
              "retained": np.flatnonzero(~rem & outside)}
    rng = np.random.default_rng(31)
    rows = []
    for name, pop in strata.items():
        q = np.searchsorted(np.quantile(nw[pop], [0.2, 0.4, 0.6, 0.8]), nw[pop], side="right")
        for h in range(5):
            ph = pop[q == h]
            k = int(round(SIZES[name] * len(ph) / len(pop)))
            for d in rng.choice(ph, min(k, len(ph)), replace=False):
                rows.append(dict(doc=int(d), stratum=name, length_quintile=h, weight=len(ph) / k))
    # counterpart: best exact Jaccard among A-component members (deleted, supported) or
    # seed-1 bucket co-members (retained); member lists are capped at 500 random documents
    kb = CL.full_band_keys(CL.group_band_keys(z, 1), z)
    order = {}
    for r in rows:
        d = r["doc"]
        if r["stratum"] == "retained":
            mem = set()
            for b in range(kb.shape[0]):
                if b not in order:
                    order[b] = (np.argsort(kb[b], kind="stable"),)
                o = order[b][0]
                ks = kb[b][o]
                lo, hi = np.searchsorted(ks, kb[b, d]), np.searchsorted(ks, kb[b, d], "right")
                mem.update(o[lo:hi].tolist())
            mem = np.array(sorted(mem - {d}), np.int64)
        else:
            mem = np.flatnonzero(root == root[d])
            mem = mem[mem != d]
        if len(mem) > 500:
            mem = rng.choice(mem, 500, replace=False)
        best, bj = None, 0.0
        if len(mem):
            jac = K.L.jaccard_pairs(off, flat, np.full(len(mem), d, np.int64), mem.astype(np.int64))
            i = int(np.argmax(jac))
            best, bj = int(mem[i]), float(jac[i])
        r.update(counterpart=best, counterpart_jaccard=bj, words=int(nw[d]))
    need = sorted({r["doc"] for r in rows} | {r["counterpart"] for r in rows if r["counterpart"] is not None})
    texts, pos = {}, 0
    want = set(need)
    for f in sorted(os.listdir(S.CORPUS)):
        if not f.startswith("shard-"):
            continue
        t = pq.read_table(os.path.join(S.CORPUS, f), columns=["id", "text"])
        ids, tx = t["id"].to_pylist(), t["text"].to_pylist()
        for i in range(len(ids)):
            if pos + i in want:
                texts[pos + i] = (ids[i], tx[i])
        pos += len(ids)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "samples.jsonl"), "w") as f:
        for r in rows:
            r["id"], r["text"] = texts[r["doc"]]
            c = r["counterpart"]
            r["counterpart_text"] = texts[c][1] if c is not None and r["counterpart_jaccard"] >= 0.4 else None
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: int(len(v)) for k, v in strata.items()}), len(rows), "sampled", flush=True)


def kept():
    """Re-pair every sampled deletion with the document A retains from its component, so that the
    label says whether the retained corpus keeps the deleted material. Retained documents keep their
    counterparts. Rows whose counterpart already was the retained document reuse their labels, since
    their prompt is unchanged. Writes BASE/kept/{samples,labels_<model>}.jsonl."""
    import collapsed_lib as CL
    import pyarrow.parquet as pq
    S, K = CL.S, CL.K
    off, flat, nw = S.load_snapshot()
    with np.load(os.path.join(S.CACHE, "masks", "raw_s1.npz")) as f:
        root, rem = f["root"], f["rem"]
    order = np.argsort(root, kind="stable")
    rs = root[order]
    rows = [json.loads(l) for l in open(os.path.join(BASE, "samples.jsonl"))]
    reuse = set()
    for r in rows:
        if r["stratum"] == "retained":
            reuse.add(r["doc"])
            continue
        d = r["doc"]
        comp = order[np.searchsorted(rs, root[d]):np.searchsorted(rs, root[d], "right")]
        k = int(comp[~rem[comp]][0])
        r["closest_member"], r["closest_member_jaccard"] = r["counterpart"], r["counterpart_jaccard"]
        if k == r["counterpart"]:
            reuse.add(d)
            continue
        jk = float(K.L.jaccard_pairs(off, flat, np.array([d], np.int64), np.array([k], np.int64))[0])
        r.update(counterpart=k, counterpart_jaccard=jk, counterpart_text=None)
    want = {r["counterpart"] for r in rows
            if r["doc"] not in reuse and r["counterpart_jaccard"] >= 0.4}
    texts, pos = {}, 0
    for f in sorted(os.listdir(S.CORPUS)):
        if not f.startswith("shard-"):
            continue
        t = pq.read_table(os.path.join(S.CORPUS, f), columns=["text"])
        tx = t["text"].to_pylist()
        for i in range(len(tx)):
            if pos + i in want:
                texts[pos + i] = tx[i]
        pos += len(tx)
    for r in rows:
        if r["doc"] not in reuse and r["counterpart_jaccard"] >= 0.4:
            r["counterpart_text"] = texts[r["counterpart"]]
    dst = os.path.join(BASE, "kept")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "samples.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for fn in os.listdir(BASE):
        if fn.startswith("labels_"):
            with open(os.path.join(dst, fn), "w") as f:
                for l in open(os.path.join(BASE, fn)):
                    x = json.loads(l)
                    if x["doc"] in reuse and x.get("label"):
                        f.write(l)
    print("kept: reused", len(reuse), "of", len(rows), "to label", len(rows) - len(reuse))


def call(model, messages, key, timeout=180):
    body = json.dumps(dict(model=model, messages=messages, temperature=0)).encode()
    op = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY, "http": PROXY} if PROXY else {}))
    req = urllib.request.Request(GATEWAY, data=body, headers={"Authorization": "Bearer " + key,
                                                               "Content-Type": "application/json"})
    with op.open(req, timeout=timeout) as resp:
        return json.load(resp)


def label(model):
    system, instructions = prompt_parts()
    # the key comes from the environment and is never written to disk
    key = os.environ["LLM_API_KEY"]
    assert GATEWAY, "set LLM_GATEWAY to an OpenAI-compatible chat-completions URL"
    path = os.path.join(OUT, f"labels_{model}.jsonl")
    # rows with an error are retried on restart; only labelled rows count as done
    done = {r["doc"] for r in map(json.loads, open(path)) if r.get("label")} if os.path.exists(path) else set()
    rows = [json.loads(l) for l in open(os.path.join(OUT, "samples.jsonl"))]
    rows = [r for r in rows if r["doc"] not in done]
    # fixed random order, so that a partial run is a random subsample of every stratum
    rows = [rows[i] for i in np.random.default_rng(97).permutation(len(rows))]
    lock = threading.Lock()
    out = open(path, "a")

    def one(r):
        doc = r["text"][:12000]
        user = instructions + "\n\nDELETED DOCUMENT:\n" + doc
        if r["counterpart_text"]:
            user += (f"\n\nCLOSEST COUNTERPART (exact 5-gram Jaccard {r['counterpart_jaccard']:.2f}):\n"
                     + r["counterpart_text"][:12000])
        else:
            user += f"\n\nCLOSEST COUNTERPART: none shown (maximum exact Jaccard {r['counterpart_jaccard']:.2f})"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(8):  # up to ~15 minutes, enough to ride out a proxy reconnect
            try:
                res = call(model, msgs, key)
                raw = res["choices"][0]["message"]["content"]
                js = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
                row = dict(doc=r["doc"], id=r["id"], stratum=r["stratum"], model=model,
                           label=js.get("label"), confidence=js.get("confidence"), reason=js.get("reason"),
                           lost_fact=js.get("lost_fact"), usage=res.get("usage"), raw=raw)
                with lock:
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                return
            except Exception as exc:  # retry transient gateway errors
                err = repr(exc)
                time.sleep(30 * (attempt + 1))
        with lock:
            out.write(json.dumps(dict(doc=r["doc"], id=r["id"], stratum=r["stratum"], model=model,
                                      label=None, error=err)) + "\n")

    with ThreadPoolExecutor(max_workers=int(os.environ.get("C_WORKERS", "8"))) as ex:
        list(ex.map(one, rows))
    print(model, "LABEL_DONE", flush=True)


def analyze():
    rows = {json.loads(l)["doc"]: json.loads(l) for l in open(os.path.join(OUT, "samples.jsonl"))}
    models = sorted(f[7:-6] for f in os.listdir(OUT) if f.startswith("labels_"))
    lab = {mo: {r["doc"]: r["label"] for r in map(json.loads, open(os.path.join(OUT, f"labels_{mo}.jsonl"))) if r.get("label")}
           for mo in models}
    cats = ["template", "repost", "partial_rewrite", "unique_facts", "unique_content"]
    res = dict(models=models, n=len(rows), strata={})
    for s in ("unsupported", "supported", "retained"):
        docs = [d for d, r in rows.items() if r["stratum"] == s]
        w = np.array([rows[d]["weight"] for d in docs])
        words = np.array([rows[d]["words"] for d in docs])
        st = dict(n=len(docs))
        for mo in models:
            lb = [lab[mo].get(d) for d in docs]
            st[mo] = {c: float(sum(wi for wi, x in zip(w, lb) if x == c) / w.sum()) for c in cats}
            st[mo]["missing"] = int(sum(x is None for x in lb))
        if len(models) == 2:
            a, b = models
            pairs = [(lab[a].get(d), lab[b].get(d)) for d in docs]
            pairs = [p for p in pairs if None not in p]
            po = np.mean([x == y for x, y in pairs])
            pa = {c: np.mean([x == c for x, _ in pairs]) for c in cats}
            pb = {c: np.mean([y == c for _, y in pairs]) for c in cats}
            pe = sum(pa[c] * pb[c] for c in cats)
            st["agreement"] = float(po)
            st["kappa"] = float((po - pe) / (1 - pe)) if pe < 1 else None
            uniq = {"unique_facts", "unique_content"}
            both = np.array([lab[a].get(d) in uniq and lab[b].get(d) in uniq for d in docs])
            st["both_unique_share_docs"] = float((w * both).sum() / w.sum())
            st["both_unique_share_words"] = float((w * both * words).sum() / (w * words).sum())
        res["strata"][s] = st
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"sample": sample, "kept": kept, "label": lambda: label(sys.argv[2]), "analyze": analyze}[sys.argv[1]]()
