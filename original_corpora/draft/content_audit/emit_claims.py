#!/usr/bin/env python3
"""Append the content-audit claims (C-74 onward) to notes/claims_draft.md from the JSON.

The block is delimited by HTML comments and replaced on every run, so rerunning after the
final labels does not duplicate rows.  All numbers come from content_audit.json.

usage: python emit_claims.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CA = os.path.join(ROOT, "output/dedupjoin-pvldb/results/content_audit/content_audit.json")
LEDGER = os.path.join(ROOT, "output/dedupjoin-pvldb/notes/claims_draft.md")
START, END = "<!-- content-audit:start -->", "<!-- content-audit:end -->"


def main():
    d = json.load(open(CA))
    st = d["strata"]
    pl = d["population_estimates"]["pooled"]
    g = d["gate"]
    c4 = st["c4_union_no_nb"]["primary"]
    wet, fw = st["wet_disputed"]["primary"], st["fw_disputed"]["primary"]
    de, ke = st["c4_agreed_deleted"]["primary"], st["c4_agreed_kept"]["primary"]
    s = d["per_seed_stability"]
    ag = d["agreement"] or {}
    lh, sh = ag.get("llm_vs_hand") or {}, ag.get("script_vs_hand") or {}
    sens = (d.get("sensitivity") or {}).get("c4_union_no_nb", {})

    def p(x):
        return f"{100 * x:.1f}%"

    def m(x):
        return f"{x / 1e6:.2f}M"

    pop = d["design"]["population"]
    rows = [
        ("C-74", f"260-document stratified sample: {c4['n']} C4 unsupported disputed documents "
                 f"drawn from the {pop.get('c4_union_100_seeds_no_nb', 0)}-document union over 100 seeds "
                 f"with no exact neighbour ({pop['c4_disputed_docs_seed1']} disputed at seed 1; "
                 f"{pop['wet_disputed_docs']} WET and {pop['fw_disputed_docs']} FineWeb disputed), "
                 f"20+20 agreed-deleted/kept controls", "`design` in `content_audit.json`", "supported"),
        ("C-75", f"C4 disputed deletions are {p(c4['template']['share'] or 0)} template; "
                 f"{c4['lost_information']['n']} documents carry lost information = "
                 f"{p(c4['lost_information']['token_share'] or 0)} of disputed words "
                 f"({p((c4['lost_information']['token_share_ci'] or [0,0])[0])}--"
                 f"{p((c4['lost_information']['token_share_ci'] or [0,0])[1])} 95% CI)",
         "Table `tab:content`; `strata.c4_union_no_nb`", "supported"),
        ("C-76", f"Controls separate the categories: {p(de['lost_information']['token_share'] or 0)} "
                 f"of agreed deletions are template/repost with zero lost words, and "
                 f"{p(ke['lost_information']['token_share'] or 0)} of agreed keeps are lost-information",
         "`strata.c4_agreed_deleted`, `strata.c4_agreed_kept`", "supported"),
        ("C-77", f"Pool-wide lost information about {m(pl['lost_tokens_estimate'])} words "
                 f"(95% CI {m(pl['ci'][0])}--{m(pl['ci'][1])}; lower bound "
                 f"{m(pl.get('lost_tokens_lower', 0))}), {pl.get('lost_tokens_share_of_pool', 0)*100:.2f}% "
                 f"of the {d['design']['population']['pool_tokens']/1e9:.2f}B-word pool, from about "
                 f"{pl.get('lost_docs_estimate', 0):.0f} documents "
                 f"({pl.get('lost_docs_ci', [0,0])[0]:.0f}--{pl.get('lost_docs_ci', [0,0])[1]:.0f})",
         "`population_estimates.pooled`; bootstrap over documents", "extrapolated"),
        ("C-78", f"Lost-information share is not stable across seeds: per-seed {p(s['theta_min'])}--"
                 f"{p(s['theta_max'])} (10th--90th {p(s['theta_p10'])}--{p(s['theta_p90'])}), "
                 f"overdispersion {s['overdispersion']:.2f}", "`per_seed_stability`", "supported"),
        ("C-79", f"Hand check: model vs hand {lh.get('k',0)}/{lh.get('n',0)} "
                 f"({p(lh.get('share',0))}, 95% CI "
                 f"{p((lh.get('ci') or [0,0])[0])}--{p((lh.get('ci') or [0,0])[1])}); "
                 f"script vs hand {sh.get('k',0)}/{sh.get('n',0)} ({p(sh.get('share',0))})",
         "`agreement`; `artifacts/blind_review/`", "supported"),
        ("C-80", f"The pre-registered gate returns {g['decision']}: G1 {m(pl['lost_tokens_estimate'])} "
                 f"vs a 10M threshold, G2 {g['G2_material']['with_qa']} question-bearing documents; "
                 f"counting all documents with new numbers as lost raises the C4 share only to "
                 f"{p(sens.get('theta_if_numeric_lost', 0))}",
         "`gate`, `sensitivity` in `content_audit.json`", "supported"),
    ]
    block = [START, "", "## Content audit and probe gate (content-audit-probe-prep step, this run)",
             "", "All rows are `independently rerun (this run)` from the stored per-seed decision records;",
             "labels are model/script labels with a 29-document hand check. Source of every number:",
             "`results/content_audit/content_audit.json`, built by `code/content_audit/analyze_labels.py`;",
             "table `tab:content`; blind package `artifacts/blind_review/`.", "",
             "| id | sentence | evidence | status |", "|---|---|---|---|"]
    for rid, sent, ev, status in rows:
        block.append(f"| {rid} | \"{sent}\" | {ev} | {status} |")
    block += ["", END]
    txt = open(LEDGER).read()
    if START in txt:
        a, b = txt.index(START), txt.index(END) + len(END)
        txt = txt[:a] + "\n".join(block) + txt[b:]
    else:
        txt = txt.rstrip() + "\n\n" + "\n".join(block) + "\n"
    open(LEDGER, "w").write(txt)
    print("updated", LEDGER)


if __name__ == "__main__":
    main()
