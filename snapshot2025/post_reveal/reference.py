"""Frozen reference run, snapshot identity checks, and synthetic literal oracle."""
import csv
import hashlib
import json
from pathlib import Path

from driver import PROJECT, TAUS, read, require, save


def fixture(c, cl):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import xxhash

    rng = np.random.default_rng(708)
    def word(i):
        return "v" + "".join(chr(97 + (i // div) % 26) for div in (676, 26, 1))

    vocabulary = [word(i) for i in range(4096)]
    def sentence(length):
        return " ".join(vocabulary[i] for i in rng.integers(len(vocabulary), size=length))

    template = sentence(24)
    texts = ["intro " + template] + [template] * 88 + [template.upper()] * 2
    texts += ["", "hi", "hi", sentence(49), sentence(50)]
    # One 49-word row has 50 normalized tokens (the hyphen becomes a space).
    texts.append(sentence(48) + " split-word")
    for _ in range(48):
        # Some sub-threshold pairs share a band, exercising the unsupported numerator.
        words = sentence(64).split()
        texts.extend((" ".join(words), " ".join(sentence(10).split() + words[10:])))
    for _ in range(80):
        # Prefix changes plant chains and a second exact-copy class.
        words = sentence(64).split()
        left = words.copy()
        right = words.copy()
        left[:4] = sentence(4).split()
        right[-4:] = sentence(4).split()
        texts.extend((" ".join(left), " ".join(right), " ".join(words), " ".join(words)))
    while len(texts) < 3000:
        texts.append(sentence((12, 49, 50, 56)[len(texts) % 4]))
    corpus = c.source / "corpus"
    corpus.mkdir(parents=True)
    c.snapshot.parent.mkdir(parents=True)
    nw = [len(t.split()) for t in texts]
    ids = [f"fixture-{i:05}" for i in range(len(texts))]
    pq.write_table(pa.table(dict(id=ids, text=texts, nwords=nw)), corpus / "shard-00000.parquet")
    sizes, flat, nw_token = cl.S._shingle_batch_spacy(texts)
    off = np.r_[np.int64(0), sizes.cumsum()]
    np.savez(c.snapshot, off=off, flat=flat, nwords=nw_token)
    np.save(c.hashes, np.array([xxhash.xxh64_intdigest(t.encode()) for t in texts], np.uint64))
    return [dict(n=len(texts), retained_50=sum(x >= 50 for x in nw))]


def verify_source(c):
    import numpy as np
    import pyarrow.parquet as pq
    import xxhash
    import sys
    sys.path.insert(0, str(PROJECT / "inputs"))
    from verify_inputs import id_hash_update, sha256_file

    spec = read(PROJECT / "inputs/INPUT_PATHS.json")
    frozen = np.load(c.hashes)
    ids, content = hashlib.sha256(), hashlib.sha256()
    cursor = 0
    words = 0
    shards = sorted((c.source / "corpus").glob("shard-*.parquet"))
    require([p.name for p in shards] == [Path(p).name for p in spec["shards"]],
            "Snapshot shard list differs from INPUT_PATHS.json")
    for path in shards:
        part = hashlib.sha256()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["id", "text", "nwords"]):
            data = batch.to_pydict()
            for doc, text, nw in zip(data["id"], data["text"], data["nwords"], strict=True):
                id_hash_update(ids, doc)
                h = xxhash.xxh64_intdigest(text.encode())
                require(cursor < len(frozen) and h == int(frozen[cursor]),
                        f"Text xxh64 differs at row {cursor}")
                part.update(h.to_bytes(8, "little"))
                require(nw == len(text.split()), f"Corpus nwords differs at source row {cursor}")
                words += nw
                cursor += 1
        content.update(part.digest())
    require(cursor == len(frozen) == spec["n_documents"] and words == spec["n_total_words"]
            and ids.hexdigest() == spec["id_sequence_sha256"]
            and content.hexdigest() == spec["content_sha256"], "Frozen document identity mismatch")
    digest = sha256_file(c.snapshot)
    return dict(cache_sha256=digest, cache_byte_identical=digest == spec["shingle_cache_npz_sha256"])


def literal_oracle(cl, off, flat, z, taus, seeds):
    """Independent document-level exact join, literal bucket expansion and graph choices."""
    import numpy as np
    k = cl.K
    n = len(off) - 1
    valid = np.diff(off) > 0
    ranks = k.EJ.build_ranks(off, flat)
    exact = {}
    masks = {}
    for tau in taus:
        truth = k.EJ.exact_join(off, *ranks, tau, 12)
        i, j = truth["i"], truth["j"]
        root = k.uf_minid(n, i, j)
        masks["C", tau, 0] = root != np.arange(n)
        exact[tau] = k.has_true_neighbour(n, i, j)
        ci, cj, _ = cl.truth_u(tau)
        state = cl.exact_state(z, ci, cj)
        np.testing.assert_array_equal(state["rem"], masks["C", tau, 0])
        np.testing.assert_array_equal(state["has_nb_doc"], exact[tau])
    for seed in seeds:
        bands = k.band_keys(k.L.signatures(off, flat, k.H, seed, "dt"))
        ai, aj, _ = k.chain_edges(bands, valid)
        a = k.uf_minid(n, ai, aj) != np.arange(n)
        pi, pj, _ = k.all_bucket_pairs(bands, valid, n)
        jac = k.L.jaccard_pairs(off, flat, pi, pj)
        for tau in taus:
            ver = jac >= tau - 1e-12
            i, j = pi[ver], pj[ver]
            masks["A", tau, seed] = a
            masks["B", tau, seed] = k.uf_minid(n, i, j) != np.arange(n)
            masks["W", tau, seed] = ~k.greedy_mis(n, i, j) & valid
    return masks, exact


def maskname(method, tau, seed):
    return {"A": f"raw_s{seed}.npz", "B": f"ver_s{seed}_t{tau}.npz",
            "W": f"wit_s{seed}_t{tau}.npz", "C": f"exact_t{tau}.npz"}[method]


def acceptance(c, cl):
    import numpy as np
    import group_accuracy as ga

    before = c.fingerprint()
    identity = {} if c.smoke else verify_source(c)
    info = cl.build_collapsed()
    z = cl.load_collapsed()
    if not c.smoke:
        dup = read(PROJECT / "results/dup_sets.json")
        total = int((z["group_size"].astype(np.int64) * z["group_nshingle"]).sum())
        require(info["n_docs"] == 531597 and info["n_empty_docs"] == 994 and total == 455796881
                and info["n_unique_sets"] == dup["n_unique_sets"]
                and int(z["group_size"].max()) == dup["largest_group"],
                f"Cache structural acceptance failed: {info}, total shingles {total}")
    del z
    cl.build_ranks_u()
    # Build truth before group_accuracy holds its own copy of the representative CSR.
    for tau in TAUS:
        cl.truth_u(tau)
    ga.main()
    rows = [json.loads(line) for line in Path(cl.S.RUNS).read_text().splitlines()]
    masks = c.work / "masks"
    if c.smoke:
        oracle, neighbours = literal_oracle(cl, *cl.S.load_snapshot()[:2], cl.load_collapsed(),
                                            TAUS, range(1, 6))
        expected = {key: (int(rem.sum()), int((rem & ~neighbours[key[1]]).sum()))
                    for key, rem in oracle.items()}
    else:
        with (PROJECT / "results/per_seed_quality.csv").open() as f:
            expected = {(r["method"], float(r["tau"]), 0 if r["seed"] == "once" else int(r["seed"])):
                        (int(r["deleted"]), int(r["unsupported"])) for r in csv.DictReader(f)}
    seen = set()
    for r in rows:
        key = (r["method"], r["tau"], r["hash_seed"] or 0)
        got = (r["deleted"], r["unsupported"])
        require(got == expected.get(key), f"Acceptance failed for {key}: observed {got}, expected {expected.get(key)}")
        seen.add(key)
        if c.smoke:
            with np.load(masks / maskname(*key)) as f:
                np.testing.assert_array_equal(f["rem"], oracle[key], err_msg=str(key))
    require(len(rows) == len(seen) == 32 and seen == expected.keys(), "Incomplete acceptance matrix")
    require(before == c.fingerprint(), "Inputs changed during the acceptance run")
    save(c.gate, dict(inputs=before, masks=str(masks), accepted_rows=32, **identity))
    counts = [dict(method=m, tau=t, seed=s or None, deleted=values[0], unsupported=values[1])
              for (m, t, s), values in expected.items()]
    return [dict(accepted_rows=32, checked_counts=counts, smoke_mask_checks=32 if c.smoke else None,
                 reference="literal synthetic masks" if c.smoke else "results/per_seed_quality.csv",
                 **identity, **info)]


def release_reference(c, cl):
    # Keep the accepted masks for E2. These reconstructible files use several GiB.
    for path in [cl.COLLAPSED_NPZ, cl.RANKS_U, *(cl.truth_u_path(t) for t in TAUS)]:
        Path(path).unlink(missing_ok=True)
