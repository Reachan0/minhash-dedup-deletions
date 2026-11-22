#!/usr/bin/env python3
from pathlib import Path

from driver import PROJECT, entry, read, require
from reference import acceptance, fixture, literal_oracle, release_reference
from timing import input_path, pack, prepare, release_cache, repetitions, timed, timing_summary


def filter50(c, cl):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    tables = [pq.read_table(p, columns=["id", "nwords"])
              for p in sorted((c.source / "corpus").glob("shard-*.parquet"))]
    meta = pa.concat_tables(tables)
    words = meta["nwords"].to_numpy()
    off, flat, token_counts = cl.S.load_snapshot()
    require(len(words) == len(off) - 1, "Corpus rows and shingle cache do not align")
    if not c.smoke:
        spec = read(PROJECT / "inputs/INPUT_PATHS.json")
        require(len(words) == spec["n_documents"] and words.sum() == spec["n_total_words"],
                "Corpus nwords total differs from the manifest")
    selected = np.flatnonzero(words >= 50)
    if c.smoke:
        source = pq.read_table(c.source / "corpus/shard-00000.parquet", columns=["text"])
        literal = [i for i, text in enumerate(source["text"].to_pylist()) if len(text.split()) >= 50]
        np.testing.assert_array_equal(selected, literal)
        require(np.any((words >= 50) != (token_counts >= 50)), "Fixture lost its word-boundary case")
    ou, fu = pack(off, flat, selected)
    np.savez(input_path(c), off=ou, flat=fu, nwords=token_counts[selected])
    np.save(c.work / "source_rows.npy", selected)
    pq.write_table(meta.take(pa.array(selected)).append_column("source_row", pa.array(selected)),
                   c.dir / "filter50_ids.parquet")
    return [dict(n=len(selected), input_n=len(words), input_nwords_mean=float(words.mean()),
                 input_total_words=int(words.sum()), at_49=int((words == 49).sum()),
                 at_50=int((words == 50).sum()), word_count="corpus.nwords = len(text.split())")]


def truth(c, cl):
    cl.S.SHINGLE_NPZ = str(input_path(c))
    info = cl.build_collapsed()
    cl.build_ranks_u()
    i, j, _ = cl.truth_u(.8)
    z = cl.load_collapsed()
    state = cl.exact_state(z, i, j)
    if c.smoke:
        import numpy as np
        import group_accuracy as ga
        ga.MASK = str(c.work / "subset_masks")
        ga.TAUS, ga.SEEDS = [.8], [1]
        ga.main()
        from reference import maskname
        oracle, _ = literal_oracle(cl, *cl.S.load_snapshot()[:2], z, [.8], [1])
        for key, rem in oracle.items():
            with np.load(Path(ga.MASK) / maskname(*key)) as f:
                np.testing.assert_array_equal(f["rem"], rem, err_msg=f"filtered {key}")
    return [dict(method="C", tau=.8, collapse_key="shingle_set", n=info["n_docs"],
                 exact_deleted=int(state["rem"].sum()), exact_group_edges=len(i),
                 filtered_smoke_masks_checked=4 if c.smoke else None)]


def quality(c, cl, samples):
    import numpy as np
    z = cl.load_collapsed()
    i, j, _ = cl.truth_u(.8)
    st = cl.exact_state(z, i, j)
    valid = z["group_nshingle"][z["group_of_doc"]] > 0
    rows = []
    verified_missed = None
    for sample in samples:
        rem = np.load(sample["mask"])
        n = len(rem)
        met = cl.lift_metrics(rem, z, i, j, st["has_nb_doc"], n, valid)
        missed = int((st["rem"] & ~rem).sum())
        method = sample["method"]
        if method == "B":
            verified_missed = missed
        if method in ("B", "W"):
            met["verified_missed"] = verified_missed
        if method == "W":
            met["witness_missed"] = missed
        if method == "A":
            met["raw_missed"] = missed
        rows.append(dict(method=method, tau=.8, seed=sample["seed"],
                         collapse_key=sample["collapse_key"], source_key=sample["key"], n=n,
                         exact_deleted=int(st["rem"].sum()), missed=missed,
                         doc_recall=int((st["rem"] & rem).sum()) / max(1, int(st["rem"].sum())),
                         largest_bucket=sample["largest_bucket"],
                         bucket_pairs_sum=sample["bucket_pairs_sum"],
                         bucket_pairs_per_doc=sample["bucket_pairs_sum"] / n, **met))
    return rows


def report_timing(c, rows):
    modes = timing_summary(c, rows)
    method = rows[0]["method"]
    summary = c.summary(rows, method=method, seed=1, tau=.8, timings=modes)
    c.report.append(f"P7. Literal {method} timing ({summary['key']}).")
    for mode, group in modes.items():
        if group["status"] == "ok":
            c.report.append(f"{mode} {group['median_wall_s']:.3f} s, "
                            f"{group['peak_tree_rss_mb']/1024:.3f} GiB peak.")
        else:
            failures = [r for r in rows if r["mode"] == mode and r["status"] != "ok"]
            for r in failures:
                c.report.append(f"{mode} failed at {r['failed_stage']} under 24 GiB ({r['key']}).")


def campaign(c):
    """Measure quality and literal timing on the frozen >=50 whitespace-word subset."""
    if c.smoke:
        c.job("acceptance")
        release_reference(c, c.configure())
    c.accepted()
    subset = c.job("filter50")[0]
    c.report.append(f"## E3\n\nP7. Corpus nwords = len(text.split()), mean "
                    f"{subset['input_nwords_mean']:.4f}. Retained {subset['n']} of "
                    f"{subset['input_n']} ({subset['key']}).")
    c.job("truth")
    c.job("prepare", collapse_key=None)
    samples = []
    for method in ("A", "B", "W"):
        if method == "W":
            rows = c.job("timed", method=method, seed=1, tau=.8, mode="pathcold",
                         repeat=1, collapse_key=None, allow_memory_failure=True)
        else:
            rows = repetitions(c, method, .8, None, allow_memory_failure=method == "B")
            report_timing(c, rows)
        successful = [r for r in rows if r["status"] == "ok"]
        if successful:
            samples.append(successful[0])
            # Repeated runs and modes must make exactly the same decisions.
            import numpy as np
            reference = np.load(successful[0]["mask"])
            for r in successful[1:]:
                np.testing.assert_array_equal(reference, np.load(r["mask"]), err_msg=r["key"])
        if len(successful) < len(rows):
            c.report.extend(f"{method} {r['mode']} failed at {r['failed_stage']} ({r['key']})."
                            for r in rows if r["status"] != "ok")
            fallback = c.job("timed", method=method, seed=1, tau=.8, mode="fallback",
                             repeat=1, collapse_key="shingle_set")[0]
            if successful:
                np.testing.assert_array_equal(reference, np.load(fallback["mask"]))
            samples.append(fallback)
        if method == "A":
            for seed in range(2, 6):
                samples.extend(c.job("timed", method="A", seed=seed, tau=.8,
                                     mode="quality_path", repeat=1, collapse_key=None))
    # Keep parameters and command lines small in the quality worker.
    fields = ["key", "mask", "method", "seed", "collapse_key", "largest_bucket", "bucket_pairs_sum"]
    metrics = c.job("quality", samples=[{k: r[k] for k in fields} for r in samples])
    for r in metrics:
        c.report.append(f"{r['method']} seed {r['seed']}, {r['collapse_key'] or 'literal'}: "
                        f"{r['deleted']} deleted, {r['unsupported']} unsupported "
                        f"(share {r['unsupported_share_del']}), {r['missed']} missed, "
                        f"{r['residual_duplicate_pairs']} residual pairs, witness coverage "
                        f"{r['witness_coverage']}. Largest bucket {r['largest_bucket']}, "
                        f"bucket pairs/doc {r['bucket_pairs_per_doc']:.5f} ({r['key']}).")
    release_reference(c, c.configure())
    release_cache(c, None)
    input_path(c).unlink()


if __name__ == "__main__":
    entry("E3", campaign, dict(fixture=fixture, acceptance=acceptance, filter50=filter50,
                               truth=truth, prepare=prepare, timed=timed, quality=quality))
