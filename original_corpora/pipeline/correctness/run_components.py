#!/usr/bin/env python3
"""Recheck the Draft 2 C4 counterexample and export real component examples.

At C4 / tau=0.8 / 14x8 the 14x8 seed study reports seed 3 with candidate pair
recall 94.65% but only 18 extra kept documents, and seed 1 with recall 97.47%
but 24 extra kept documents.  The explanation is structural: a missing edge only
changes the deletion set if it is a bridge; many missing edges inside one dense
component can leave it connected, while a single missing edge between two
documents splits a component and keeps one extra document.

This script recomputes B for c4 seeds 1 and 3, classifies every exact component
(k blocks, missing edges), exports one readable component of each class as JSON
for the figure, and records the two-class counts for every seed from the ledger.

Also records the status of the new independent snapshot: this step does not run
dedup on it; the counts are to be filled by `independent-snapshot-eval`.

Usage: python3 run_components.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correctness_lib as K
import run_ledger as L

CORPUS = "c4"
TAU = 0.8
SEEDS = [1, 3]
OUT = os.path.join(K.OUTDIR, "component_cases.json")


def recompute(seed):
    n = K.N_DOCS[CORPUS]
    off, _ = K.load_offsets(CORPUS)
    valid = np.diff(off) > 0
    ti, tj, root_exact, _rem = L.load_truth(CORPUS, TAU, n)
    flat = K.extract_flat(CORPUS)
    sig = K.load_signatures(CORPUS, seed)
    Kb = K.band_keys(sig, K.B, K.R)
    del sig
    pi, pj, _ = K.all_bucket_pairs(Kb, valid, n)
    jac = K.jaccard_pairs(off, flat, pi, pj)
    del flat, off, Kb
    ver = jac >= TAU - 1e-12
    ei, ej = pi[ver], pj[ver]
    rootB = K.uf_minid(n, ei, ej)
    det = K.detected_truth_mask(ti, tj, pi, pj, n)
    split = K.component_split(root_exact, rootB, n)
    cst = K.component_edge_stats(split, root_exact, ti, det)
    # component document ids, in the component order of split["uniq"]
    uniq, inv, cnt = np.unique(root_exact, return_inverse=True, return_counts=True)
    # map every true edge to its component index in the `ue` ordering
    ue = np.unique(root_exact[ti])
    eidx = np.searchsorted(ue, root_exact[ti])
    comp_pos = np.searchsorted(uniq, ue)
    return dict(n=n, ti=ti, tj=tj, det=det, root_exact=root_exact, rootB=rootB,
                uniq=uniq, inv=inv, cnt=cnt, ue=ue, eidx=eidx, comp_pos=comp_pos,
                ecount=cst["ecount"], missed=cst["missed"], k=cst["k"],
                split=split, cst=cst)


def export_example(R, comp_index, max_size=14):
    """Export one exact component as a small node/edge list."""
    root = R["ue"][comp_index]
    nodes = np.nonzero(R["root_exact"] == root)[0]
    if len(nodes) > max_size:
        return None
    local = {int(v): i for i, v in enumerate(nodes)}
    emask = R["eidx"] == comp_index
    edges = []
    for a, b, d in zip(R["ti"][emask], R["tj"][emask], R["det"][emask]):
        edges.append([local[int(a)], local[int(b)], int(bool(d))])
    # B blocks: distinct rootB inside the component
    blocks = {}
    for v in nodes:
        blocks.setdefault(int(R["rootB"][v]), []).append(local[int(v)])
    block_list = [sorted(v) for v in blocks.values()]
    # nodes are in ascending global-id order, so the smallest local index in a
    # block is also the smallest global id, i.e. the kept document
    kept_local = sorted(min(b) for b in block_list)
    return dict(size=int(len(nodes)),
                true_edges=int(emask.sum()),
                detected=int(R["det"][emask].sum()),
                missed=int((~R["det"][emask]).sum()),
                k=int(len(block_list)),
                nodes=[int(v) for v in nodes],
                edges=edges,
                blocks=block_list,
                kept_local=kept_local)


def pick_examples(R):
    size = R["cnt"][R["comp_pos"]]
    edges = R["ecount"]
    missed, k = R["missed"], R["k"]
    # leaky but still connected: keep the drawing readable (few nodes, few edges)
    leaky = None
    for max_size, max_edges in ((8, 10), (10, 14), (14, 22)):
        cand = np.where((k == 1) & (missed >= 2) & (size <= max_size)
                        & (edges <= max_edges))[0]
        if len(cand):
            leaky = int(cand[np.argmax(missed[cand])])
            break
    # split by a handful of missing edges
    split = None
    for max_size, max_edges in ((5, 6), (8, 8), (12, 10)):
        cand = np.where((k >= 2) & (missed <= 2) & (size <= max_size)
                        & (edges <= max_edges))[0]
        if len(cand):
            split = int(cand[np.argmax(size[cand])])
            break
    giant = int(np.argmax(R["ecount"]))
    return dict(leaky=leaky, split=split, giant=giant)


def main():
    cases = {"corpus": CORPUS, "tau": TAU,
             "generated_utc": K.now(),
             "seeds": {}}
    for seed in SEEDS:
        R = recompute(seed)
        pick = pick_examples(R)
        out = dict(
            total_missed_edges=int(R["missed"].sum()),
            total_verified_missed=None,
            giant=dict(size=int(R["cnt"][R["comp_pos"][pick["giant"]]]),
                       true_edges=int(R["ecount"][pick["giant"]]),
                       missed=int(R["missed"][pick["giant"]]),
                       k=int(R["k"][pick["giant"]])),
            connected_with_many_missed=R["cst"]["connected_with_many_missed"],
            missed_edges_in_those=R["cst"]["missed_edges_in_those"],
            split_with_few_missed=R["cst"]["split_with_few_missed"],
            split_total=R["cst"]["split_total"],
            sum_k_minus_1=int(R["split"]["sum_k_minus_1"]),
            examples={},
        )
        for name in ("leaky", "split"):
            ci = pick[name]
            if ci is not None:
                ex = export_example(R, ci)
                if ex is not None:
                    ex["class"] = ("connected_with_many_missed" if name == "leaky"
                                   else "split_with_few_missed")
                    out["examples"][name] = ex
        cases["seeds"][str(seed)] = out
        print(f"seed {seed}: missed={out['total_missed_edges']} "
              f"giant(k={out['giant']['k']}, missed={out['giant']['missed']}) "
              f"examples={list(out['examples'])}", flush=True)
        K.drop_flat(CORPUS)

    # verified_missed and per-seed class counts from the ledger
    led = []
    lp = os.path.join(K.OUTDIR, "ledger.jsonl")
    if os.path.exists(lp):
        for line in open(lp, encoding="utf-8"):
            r = json.loads(line)
            if r["corpus"] == CORPUS and abs(r["tau"] - TAU) < 1e-9:
                led.append(r)
    for r in led:
        s = str(r["seed"])
        if s in cases["seeds"]:
            cases["seeds"][s]["total_verified_missed"] = r["deletion_layer"]["verified_missed"]
            cases["seeds"][s]["pair_recall"] = r["candidate_layer"]["verified_pair_recall"]
            cases["seeds"][s]["ledger_classes"] = r["components"]
    # all five seeds: missed edges from the small component caches, verified
    # missed and recall from the ledger rows
    led_by_seed = {int(r["seed"]): r for r in led}
    per_seed = []
    for seed, path in sorted((s, os.path.join(K.WRK, f"{CORPUS}_comp_s{s}_t{TAU}.npz"))
                             for s in K.SEEDS[CORPUS]):
        if not os.path.exists(path):
            continue
        z = np.load(path)
        missed = int(int(z["ecount"].sum()) - int(z["hits"].sum()))
        z.close()
        r = led_by_seed.get(seed)
        per_seed.append(dict(
            seed=seed, missed_edges=missed,
            verified_missed=r["deletion_layer"]["verified_missed"] if r else None,
            pair_recall=r["candidate_layer"]["verified_pair_recall"] if r else None,
            giant_k=(cases["seeds"].get(str(seed), {}).get("giant") or {}).get("k"),
            giant_missed=(cases["seeds"].get(str(seed), {}).get("giant") or {}).get("missed"),
        ))
    cases["per_seed"] = per_seed

    # new snapshot: this step produces no dedup result on it
    newsnap = os.path.join(K.SLUG, "newsnap")
    has_result = any(
        os.path.exists(os.path.join(newsnap, p))
        for p in ("dedup", "results", "results.json", "dedup_results.json"))
    cases["new_snapshot"] = dict(
        path_hint="newsnap",
        dedup_result_available=bool(has_result),
        status="pending: independent-snapshot-eval has not produced deletion sets yet"
        if not has_result else "available: run the same two-class count",
        note="two-class counts on CC-MAIN-2025-30 are to be filled by independent-snapshot-eval",
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=1, ensure_ascii=False)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
