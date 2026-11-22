#!/usr/bin/env python3
"""Build the self-contained reading view (paper/reading.html) from the LaTeX source.

The PDF remains the authority; this view is generated from the same sources so that
no number or sentence is retyped.  Citation numbers come from main.aux (\\bibcite),
the reference list is converted from main.bbl, and every figure is embedded as a
data URI by pandoc --self-contained.

Run from anywhere:  python3 output/dedupjoin-pvldb/code/build_reading_html.py
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PAPER = os.path.join(ROOT, "output/dedupjoin-pvldb/paper")
BUILD = os.path.join(ROOT, ".work/dedupjoin/htmlbuild")
OUT = os.path.join(PAPER, "reading.html")

CSS = """
body { max-width: 46rem; margin: 2.5rem auto; padding: 0 1.2rem;
       font-family: "Liberation Serif", "Noto Serif", Georgia, serif;
       font-size: 17px; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 1.65rem; line-height: 1.25; text-align: center; margin: 0 0 .4rem; }
h2 { font-size: 1.22rem; margin: 1.9rem 0 .55rem; border-bottom: 1px solid #e2b4ad;
     padding-bottom: .18rem; }
h3 { font-size: 1.03rem; margin: 1.35rem 0 .4rem; }
p.author, p.date { text-align: center; color: #444; }
div.abstract { margin: 1.6rem 3rem; font-size: .95rem; }
div.abstract-title { font-weight: bold; text-align: center; margin-bottom: .4rem; }
table { border-collapse: collapse; margin: .9rem auto 1.1rem; font-size: 12.5px;
        line-height: 1.3; }
th, td { padding: .16rem .42rem; }
thead th { border-top: 1.1px solid #222; border-bottom: .6px solid #222; }
tbody tr:last-child td { border-bottom: 1.1px solid #222; }
table caption, figure figcaption { font-size: 13.5px; line-height: 1.35; margin: .45rem auto .9rem;
        max-width: 92%; color: #222; }
figure { margin: 1.2rem 0 1.4rem; text-align: center; }
figure img { max-width: 100%; height: auto; }
code, pre { font-size: .86em; }
math { font-size: 1.02em; }
ol, ul { padding-left: 1.4rem; }
li { margin: .18rem 0; }
#refs p { font-size: 13.5px; line-height: 1.35; margin: .28rem 0; text-indent: -1.45rem;
          padding-left: 1.45rem; }
"""


def expand_inputs(text, depth=0):
    def repl(m):
        path = m.group(1)
        if not path.endswith(".tex"):
            path += ".tex"
        full = os.path.join(PAPER, path)
        if depth < 3 and os.path.exists(full):
            return expand_inputs(open(full).read(), depth + 1)
        return ""

    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*$", "", text, flags=re.M)


def build_body():
    tex = open(os.path.join(PAPER, "main.tex")).read()
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    body = expand_inputs(body)
    body = strip_comments(body)

    aux = open(os.path.join(PAPER, "main.aux")).read()
    labels = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux))
    cites = {k: int(v) for k, v in re.findall(
        r"\\bibcite\{([^}]+)\}\{\{(\d+)\}", aux)}

    def cite(m):
        nums = sorted(cites[k.strip()] for k in m.group(1).split(",") if k.strip() in cites)
        return "[" + ", ".join(str(n) for n in nums) + "]"

    body = re.sub(r"\\cite\{([^}]+)\}", cite, body)
    for key, num in labels.items():
        body = body.replace("\\ref{%s}" % key, num)
        body = body.replace("\\eqref{%s}" % key, "(%s)" % num)

    body = body.replace(r"\begin{table*}", r"\begin{table}").replace(
        r"\end{table*}", r"\end{table}")
    body = body.replace(r"\begin{figure*}", r"\begin{figure}").replace(
        r"\end{figure*}", r"\end{figure}")
    for cmd in (r"\maketitle", r"\vldbtopmatter", r"\FloatBarrier", r"\clearpage",
                r"\normalfont\mdseries", r"\bibliographystyle{ACM-Reference-Format}",
                r"\bibliography{bibliography}"):
        body = body.replace(cmd, "")
    return body.strip()


def build_refs():
    bbl = open(os.path.join(PAPER, "main.bbl")).read()
    bbl = bbl.split(r"\begin{thebibliography}{33}", 1)[1].split(
        r"\end{thebibliography}", 1)[0]
    bbl = re.sub(r"%\s*\n\s*", "", bbl)
    bbl = "\n".join(
        line for line in bbl.splitlines()
        if not line.strip().startswith(("%", r"\ifx", r"\providecommand")))
    bbl = bbl.replace("{{FOLD}", "{FOLD").replace("{{PUFFER}", "{PUFFER")
    parts = re.split(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", bbl)
    head = (
        r"\newcommand{\bibinfo}[2]{#2}" "\n"
        r"\newcommand{\bibfield}[2]{#2}" "\n"
        r"\newcommand{\natexlab}[1]{#1}" "\n"
        r"\newcommand{\showeprint}[2][]{arXiv:#2}" "\n"
        r"\newcommand{\showDOI}[1]{#1}" "\n"
        r"\newcommand{\showURL}[1]{#1}" "\n"
        r"\newcommand{\showCODEN}[1]{}" "\n"
        r"\newcommand{\showISBNx}[1]{}" "\n"
        r"\newcommand{\showISBNxiii}[1]{}" "\n"
        r"\newcommand{\showISSN}[1]{}" "\n"
        r"\newcommand{\showLCCN}[1]{}" "\n"
        r"\newcommand{\shownote}[1]{#1}" "\n"
        r"\newcommand{\showarticletitle}[1]{#1}" "\n"
    )
    chunks = [head]
    n = 0
    for i in range(1, len(parts), 2):
        n += 1
        chunks.append(r"\noindent\textbf{[%d]} " % n + parts[i + 1].strip() + "\n\n")
    return "".join(chunks), n


def pandoc(src, dst):
    subprocess.run(
        ["pandoc", src, "-f", "latex", "-t", "html5", "--mathml",
         "--self-contained", "--standalone", "-o", dst],
        check=True, cwd=PAPER)


def clean(fragment):
    # MathML annotation nodes carry the TeX source; they are not visible in browsers.
    fragment = re.sub(r"<annotation[^>]*>.*?</annotation>", "", fragment, flags=re.S)
    # The reading view is set in a serif stack; canonically equivalent decompositions keep
    # Latin Extended-A names (e.g. Kydlíček) renderable when the first fallback font lacks
    # the precomposed glyph.  The PDF, which embeds its own fonts, is unaffected.
    fragment = fragment.replace("ı́", "í").replace("č", "č")
    return fragment


def main():
    os.makedirs(BUILD, exist_ok=True)
    body, refs = build_body(), build_refs()[0]
    open(os.path.join(BUILD, "body.tex"), "w").write(body)
    open(os.path.join(BUILD, "refs.tex"), "w").write(refs)
    pandoc(os.path.join(BUILD, "body.tex"), os.path.join(BUILD, "body.html"))
    pandoc(os.path.join(BUILD, "refs.tex"), os.path.join(BUILD, "refs.html"))

    html = open(os.path.join(BUILD, "body.html")).read()
    ref_html = open(os.path.join(BUILD, "refs.html")).read()
    ref_html = ref_html.split("<body", 1)[-1].split(">", 1)[-1].rsplit("</body>", 1)[0]
    html = clean(html)
    ref_html = clean(ref_html)
    # The citation gate reads plain text: a reference entry must start at the beginning of a
    # line, so put the number on its own line after the paragraph tag (renders identically).
    ref_html = re.sub(r"<p><strong>\[(\d+)\]</strong>\s*",
                      lambda m: "<p>\n[%s] " % m.group(1), ref_html)
    html = html.replace("</style>", CSS + "\n</style>", 1)
    html = html.replace(
        "</body>",
        '<section id="refs">\n<h2>References</h2>\n' + ref_html + "</section>\n</body>",
        1)
    open(OUT, "w").write(html)
    print("wrote", OUT, len(html), "bytes,", len(re.findall(r'<img', html)), "images")


if __name__ == "__main__":
    main()
