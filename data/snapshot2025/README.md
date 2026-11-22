# Snapshot-2025 release files (no document text)

Snapshot-2025: 531,597 English documents of CC-MAIN-2025-30, in the corpus order of the rebuild
scripts. Produced by `experiments/b1_tools/release_snapshot.py` and
`release_audit.py`. Shingles: word 5-grams of the normalised token stream; banding 14x8 (FineWeb).

## documents.parquet (one row per document)

| Column | Meaning |
|---|---|
| doc | index in the corpus, 0-based |
| record_id | record identifier of the WET extract |
| set_id | identifier of its distinct shingle set; documents with the same set_id have J = 1 |
| n_shingles | number of distinct shingles (0 for documents without any) |
| nb_080, nb_075 | has a neighbour at exact J >= 0.8 / 0.75 (soundness label) |
| C_080 | deleted by the exact join at tau 0.8 (minimum-identifier representative) |
| A, A_dt, B, W | deleted at tau 0.8; bit s-1 is set if deleted with hash seed s (s = 1..5). A: datatrove chain with min-id representative; A_dt: datatrove's own representative rule; B: every within-bucket pair verified; W: witness rule |

## reference_edges_t0.80.parquet, reference_edges_t0.75.parquet

The exact reference between distinct shingle sets: `set_i < set_j`, `jaccard >= tau`. Lift to
documents through `set_id`. The exact components at tau are the connected components of these
edges together with the identical-set groups.

## content_audit_labels.parquet

The 2,000 documents of the content audit (Section 11): design fields (stratum, word-count quintile,
weight), the retained document of the A cluster and the cluster's most similar member with their
Jaccard similarities, and the label of each model against each counterpart. The prompt is
`original_corpora/draft/content_audit/llm_label.py`. Model responses, which restate document content,
are not included.

## Use

Under the corpora's source terms (Common Crawl terms of use). An evaluation of another
deduplicator on Snapshot-2025 needs only its deletion set by `doc`: its unsupported share is the
share of deletions with `nb_080 == False`.
