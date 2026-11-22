#!/usr/bin/env python3
import time

from driver import TAUS, entry, require
from reference import acceptance, fixture, release_reference


def outside(c, cl):
    import numpy as np
    masks = c.accepted()
    z = cl.load_collapsed()
    largest = int(z["group_size"].argmax())
    ing = z["group_of_doc"] == largest
    g0 = int(z["group_rep"][largest])
    out = []
    for tau in TAUS:
        ti, tj, _ = cl.truth_u(tau)
        supported = cl.exact_state(z, ti, tj)["has_nb_doc"]
        for seed in range(1, 6):
            start = time.perf_counter()
            with np.load(masks / f"raw_s{seed}.npz") as f:
                root, rem = f["root"], f["rem"]
            component = root == root[g0]
            require(component[ing].all(), "G is split across A components")
            uns = rem & ~supported
            outside_d = int((rem & ~ing).sum())
            outside_u = int((uns & ~ing).sum())
            out.append(dict(method="A", mode="quality", tau=tau, seed=seed,
                            collapse_key="shingle_set", n=len(rem), largest_group=int(ing.sum()),
                            deleted=int(rem.sum()), deleted_in_G=int((rem & ing).sum()),
                            deleted_outside_G=outside_d, unsupported=int(uns.sum()),
                            unsupported_outside_G=outside_u,
                            share_outside=outside_u / outside_d if outside_d else None,
                            non_G_in_component=int((component & ~ing).sum()),
                            component_rep=int(root[g0]), representative_in_G=bool(ing[root[g0]]),
                            stage_count_s=time.perf_counter() - start))
    release_reference(c, cl)
    return out


def campaign(c):
    """Reproduce the quality gate, then measure A outside the largest shingle group."""
    import statistics
    gate = c.job("acceptance")[0]
    c.report.append(f"## E1\n\n{gate['accepted_rows']} frozen quality cells passed ({gate['key']}).")
    rows = c.job("outside")
    for tau in TAUS:
        members = [r for r in rows if r["tau"] == tau]
        shares = [r["share_outside"] for r in members]
        summary = c.summary(members, method="A", tau=tau, collapse_key="shingle_set",
                            mean_share_outside=statistics.mean(shares), min_share_outside=min(shares),
                            max_share_outside=max(shares))
        c.report.append(f"τ={tau}, five-seed mean {statistics.mean(shares):.3%}, "
                        f"range {min(shares):.3%}–{max(shares):.3%} ({summary['key']}).")
    r = rows[0]
    verdict = "within" if abs(r["share_outside"] - .359) <= .03 else "outside"
    c.report.append(f"P1/P3/P5/P8. Seed 1, τ=0.8: {r['unsupported_outside_G']} / "
                    f"{r['deleted_outside_G']} = {r['share_outside']:.3%}, {verdict} three percentage "
                    f"points of 35.9% ({r['key']}).")


if __name__ == "__main__":
    entry("E1", campaign, dict(fixture=fixture, acceptance=acceptance, outside=outside))
