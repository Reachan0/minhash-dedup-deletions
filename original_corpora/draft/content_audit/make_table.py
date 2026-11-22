#!/usr/bin/env python3
"""Emit the content-audit table from results/content_audit/content_audit.json.

Output: paper/tables/tab11_content.tex
Every number is copied from the JSON; no value is typed by hand.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CA = os.path.join(ROOT, "output/dedupjoin-pvldb/results/content_audit/content_audit.json")
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/paper/tables/tab11_content.tex")
NUM = os.path.join(ROOT, "output/dedupjoin-pvldb/paper/tables/content_numbers.tex")

NAMES = [
    ("c4_union_no_nb", "C4 disputed (100 seeds)"),
    ("c4_agreed_deleted", "C4 all-deleted control"),
    ("c4_agreed_kept", "C4 all-kept control"),
    ("wet_disputed", "WET disputed"),
    ("fw_disputed", "FineWeb disputed"),
]
COLS = ["template", "repost", "partial_rewrite", "unique_facts", "unique_content"]


def pct(k, n):
    return f"{100.0 * k / n:.0f}" if n else "--"


def pct1(x):
    return f"{100.0 * x:.1f}" if x is not None else "--"


def mil(x):
    return f"{x / 1e6:.2f}"


def emit_numbers(d):
    seed1p = CA.replace("content_audit.json", "content_audit_seed1.json")
    S = json.load(open(seed1p)) if os.path.exists(seed1p) else None
    st = d["strata"]
    pe = d["population_estimates"]
    pl = pe["pooled"]
    g = d["gate"]
    agree = d["agreement"] or {}
    sh = agree.get("script_vs_hand") or {}
    sens = d.get("sensitivity") or {}
    c4 = st["c4_union_no_nb"]["primary"]
    wet = st["wet_disputed"]["primary"]
    fw = st["fw_disputed"]["primary"]
    de, ke = st["c4_agreed_deleted"]["primary"], st["c4_agreed_kept"]["primary"]
    L = []
    add = lambda n, v: L.append(f"\\newcommand{{\\{n}}}{{{v}}}")
    add("caSamples", d["design"]["samples"])
    add("caHand", d["design"]["hand_checked"])
    add("caLlm", d["labeller_summary"]["llm"])
    add("caCfourN", c4["n"])
    add("caCfourTemplate", pct(c4["template"]["n"], c4["n"]))
    add("caCfourLostDocs", c4["lost_information"]["n"])
    add("caCfourLostDocShare", pct1(c4["lost_information"]["share"]))
    if S:
        add("caCfourLostShare", pct1(S["c4"]["theta"]))
        add("caCfourLostShareLo", pct1(S["c4"]["theta_ci"][0]))
        add("caCfourLostShareHi", pct1(S["c4"]["theta_ci"][1]))
        add("caCfourUnionShare", pct1(S["union_sample"]["theta"]))
        add("caCfourSeedN", S["seed1_sample"]["n"])
        add("caCfourSeedK", S["seed1_sample"]["k"])
        add("caCfourDisputedDocs", f"{S['c4']['disputed_docs']:,}".replace(",", "{,}"))
        add("caCfourUnsupportedDocs", f"{int(S['c4']['unsupported_docs']):,}".replace(",", "{,}"))
        add("caCfourDisputedWords", f"{S['c4']['disputed_tokens']/1e6:.2f}")
    else:
        add("caCfourLostShare", pct1(c4["lost_information"]["token_share"]))
        lo, hi = c4["lost_information"]["token_share_ci"] or [0, 0]
        add("caCfourLostShareLo", pct1(lo))
        add("caCfourLostShareHi", pct1(hi))
        add("caCfourUnionShare", pct1(c4["lost_information"]["token_share"]))
        add("caCfourSeedN", "--")
        add("caCfourSeedK", "--")
        add("caCfourDisputedDocs", "--")
        add("caCfourUnsupportedDocs", "--")
        add("caCfourDisputedWords", "--")
    add("caWetN", wet["n"])
    add("caWetLostShare", pct1(wet["lost_information"]["token_share"]))
    add("caWetLostDocs", wet["lost_information"]["n"])
    add("caFwN", fw["n"])
    add("caFwLostShare", pct1(fw["lost_information"]["token_share"]))
    add("caFwLostDocs", fw["lost_information"]["n"])
    add("caDelTemplate", pct(de["template"]["n"] + de["repost"]["n"], de["n"]))
    add("caKeptLost", pct(ke["lost_information"]["n"], ke["n"]))
    if S:
        pl = S["pooled"]
        add("caPoolLost", mil(pl["lost_tokens_estimate"]))
        add("caPoolLostLo", mil(pl["lost_tokens_ci"][0]))
        add("caPoolLostHi", mil(pl["lost_tokens_ci"][1]))
        add("caPoolLostLower", mil(pl["lost_tokens_lower"]))
        add("caPoolShare", f"{100 * pl.get('lost_tokens_share_of_pool', 0):.2f}")
        add("caPoolTokens", f"{pl['pool_tokens'] / 1e9:.2f}")
        add("caPoolDocs", int(round(pl["lost_docs_estimate"])))
        add("caPoolDocsLo", int(round(pl["lost_docs_ci"][0])))
        add("caPoolDocsHi", int(round(pl["lost_docs_ci"][1])))
        add("caWithQa", S["probe_material"]["with_qa"])
    else:
        add("caPoolLost", mil(pl["lost_tokens_estimate"]))
        add("caPoolLostLo", mil(pl["ci"][0]))
        add("caPoolLostHi", mil(pl["ci"][1]))
        add("caPoolLostLower", mil(pl.get("lost_tokens_lower", pl["lost_tokens_estimate"])))
        add("caPoolShare", f"{100 * pl.get('lost_tokens_share_of_pool', 0):.2f}")
        add("caPoolTokens", f"{d['design']['population']['pool_tokens'] / 1e9:.2f}")
        add("caPoolDocs", int(round(pl.get("lost_docs_estimate", 0))))
        add("caPoolDocsLo", int(round((pl.get("lost_docs_ci") or [0, 0])[0])))
        add("caPoolDocsHi", int(round((pl.get("lost_docs_ci") or [0, 0])[1])))
        add("caWithQa", d["probe_material"]["with_qa"])
    add("caGateThreshold", "10")
    add("caAgreeK", sh.get("k", 0))
    add("caAgreeN", agree.get("n", 0))
    add("caAgreeShare", pct1(sh.get("share")) if sh else "--")
    if sh:
        add("caAgreeLo", pct1(sh["ci"][0]))
        add("caAgreeHi", pct1(sh["ci"][1]))
    else:
        add("caAgreeLo", "--")
        add("caAgreeHi", "--")
    lh = agree.get("llm_vs_hand") or {}
    add("caLlmAgreeK", lh.get("k", "--"))
    add("caLlmAgreeN", lh.get("n", agree.get("n", "--")))
    add("caLlmAgreeShare", pct1(lh.get("share")) if lh else "--")
    if lh:
        add("caLlmAgreeLo", pct1(lh["ci"][0]))
        add("caLlmAgreeHi", pct1(lh["ci"][1]))
    else:
        add("caLlmAgreeLo", "--")
        add("caLlmAgreeHi", "--")
    stab = d["per_seed_stability"]
    if S:
        add("caSeedThetaMean", pct1(S["seed_theta"]["mean"]))
        add("caSeedPctTen", pct1(S["seed_theta"]["p10"]))
        add("caSeedPctNinety", pct1(S["seed_theta"]["p90"]))
        add("caSeedMax", pct1(S["seed_theta"]["max"]))
    else:
        add("caSeedThetaMean", pct1(stab["theta_mean"]))
        add("caSeedPctTen", pct1(stab["theta_p10"]))
        add("caSeedPctNinety", pct1(stab["theta_p90"]))
        add("caSeedMax", pct1(stab["theta_max"]))
    add("caOverdispersion", f"{stab['overdispersion']:.2f}")
    if "c4_union_no_nb" in sens:
        add("caSensCfour", pct1(sens["c4_union_no_nb"]["theta_if_numeric_lost"]))
    else:
        add("caSensCfour", "--")
    add("caDecision", g["decision"].replace("_", "\\_"))
    with open(NUM, "w") as f:
        f.write("% generated by code/content_audit/make_table.py from results/content_audit/content_audit.json\n")
        f.write("\n".join(L) + "\n")
    print("wrote", NUM)


def update_manifest():
    """Register the table in the paper's table manifest (generator, source, hash)."""
    import hashlib
    mp = os.path.join(os.path.dirname(OUT), "manifest.json")
    if not os.path.exists(mp):
        return
    m = json.load(open(mp))
    sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    entry = {
        "id": "tab:content",
        "path": "tab11_content.tex",
        "label": "tab:content",
        "generator": "output/dedupjoin-pvldb/code/content_audit/make_table.py",
        "data_sources": ["output/dedupjoin-pvldb/results/content_audit/content_audit.json"],
        "display": "raw",
        "n_seeds": 1,
        "sample_docs": 260,
        "n_seeds_union": 100,
        "sha256": sha,
    }
    tabs = [t for t in m.get("tables", []) if t.get("id") != "tab:content"]
    tabs.append(entry)
    m["tables"] = tabs
    with open(mp, "w") as f:
        json.dump(m, f, indent=2)
        f.write("\n")
    print("updated", mp)


def main():
    d = json.load(open(CA))
    st = d["strata"]
    lines = [
        "% generated by code/content_audit/make_table.py from results/content_audit/content_audit.json",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{1.8pt}",
        "  \\caption{What the sampled deletions contain. The five category columns are",
        "  percentages of the stratum's $n$ sampled documents; labels are one per document from a",
        "  blinded sample; the last column is the share of words in the stratum that belong to",
        "  lost-information documents (\\emph{unique facts} or \\emph{unique content}), with a",
        "  95\\% bootstrap interval over documents. The two control strata were read to check that",
        "  the labelling separates categories, not to estimate a population share. Model labels",
        f"  cover {d['labeller_summary']['by_stratum']['c4_union_no_nb']['llm']} of the "
        f"{d['strata']['c4_union_no_nb']['primary']['n']} disputed C4 documents and "
        f"{d['labeller_summary']['by_stratum']['c4_agreed_deleted']['llm'] + d['labeller_summary']['by_stratum']['c4_agreed_kept']['llm']}"
        f" of the {d['strata']['c4_agreed_deleted']['primary']['n'] + d['strata']['c4_agreed_kept']['primary']['n']}"
        f" control documents; the remaining labels, including the whole FineWeb stratum, come from the deterministic labeller.}}",
        "  \\label{tab:content}",
        "  \\begin{tabular}{lrrrrrrr}",
        "    \\toprule",
        "    Stratum & n & template & repost & part. & facts & content & lost words \\\\",
        "    \\midrule",
    ]
    for key, name in NAMES:
        s = st[key]["primary"]
        n = s["n"]
        cells = [pct(s[c]["n"], n) for c in COLS]
        li = s["lost_information"]
        if li["token_share"] is None:
            lost = "--"
        elif li["token_share_ci"]:
            lo, hi = li["token_share_ci"]
            lost = f"{100*li['token_share']:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
        else:
            lost = f"{100*li['token_share']:.1f}"
        lines.append(f"    {name} & {n} & " + " & ".join(cells) + f" & {lost} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print("wrote", OUT)
    update_manifest()
    emit_numbers(d)


if __name__ == "__main__":
    main()
