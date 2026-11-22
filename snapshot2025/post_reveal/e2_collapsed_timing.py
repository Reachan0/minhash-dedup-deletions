#!/usr/bin/env python3
from driver import KEYS, TAUS, entry
from reference import acceptance, fixture, release_reference
from timing import prepare, release_cache, repetitions, timed, timing_summary


def campaign(c):
    """Time collapsed B/W and literal A on the same machine, then evaluate cost budgets."""
    c.report.append("## E2\n\nMedians in seconds and maximum sampled tree RSS in GiB.")
    if c.smoke:
        c.job("acceptance")
        release_reference(c, c.configure())
    c.accepted()
    for key in (*KEYS, None):
        c.job("prepare", collapse_key=key)
        for tau in TAUS:
            for method in (("B", "W") if key else ("A",)):
                rows = repetitions(c, method, tau, key)
                modes = timing_summary(c, rows)
                cold, inc = modes["pathcold"], modes["incremental"]
                cs, ins = cold["median_wall_s"], inc["median_wall_s"]
                cr, ir = cold["peak_tree_rss_mb"], inc["peak_tree_rss_mb"]
                checks = dict(S1_pathcold_400s=cs <= 400, S1_incremental_20s=ins <= 20,
                              S1_peak_16GiB=max(cr, ir) <= 16 * 1024,
                              S2_incremental_peak_6GiB=ir <= 6 * 1024, S2_incremental_30s=ins <= 30,
                              S3_incremental_15s=ins <= 15)
                summary = c.summary(rows, method=method, tau=tau, seed=1, collapse_key=key,
                                    timings=modes, cost_checks=checks, all_cost_checks=all(checks.values()))
                verdict = "pass" if all(checks.values()) else "fail"
                details = ", ".join(f"{name} {'pass' if ok else 'fail'}" for name, ok in checks.items())
                c.report.append(f"P2/P4/P6. {method}, {key or 'literal'}, τ={tau}. "
                                f"Pathcold {cs:.3f} s / {cr/1024:.3f} GiB, incremental {ins:.3f} s / "
                                f"{ir/1024:.3f} GiB. S1–S3 cost budgets {verdict}. {details} ({summary['key']}).")

        release_cache(c, key)


if __name__ == "__main__":
    entry("E2", campaign, dict(fixture=fixture, acceptance=acceptance, prepare=prepare, timed=timed))
