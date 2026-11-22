#!/usr/bin/env python3
"""Label the content-audit sample with the platform LLM outlet.

One document per call; the closest counterpart is included when the exact max Jaccard
is >= 0.4 so that "unique_facts" (a near-duplicate that still carries a fact its
counterparts lack) can be judged.  Prompt and raw response are stored with every row.
Results append to results/content_audit/llm_labels.jsonl, one JSON object per line, and
completed document ids are skipped on restart.

Cost is computed from returned usage: input $0.0288/M, output $0.288/M (llm-outlet.md).
A global 2.3 s spacing keeps the run under 30 requests/minute; concurrency 5 is used.

usage: python llm_label.py --max-docs 240 [--limit 8] [--workers 5]
"""
import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results/content_audit")
SAMPLES = os.path.join(OUT, "samples.jsonl")
LABELS = os.path.join(OUT, "llm_labels.jsonl")
PROMPT_LOG = os.path.join(OUT, "llm_prompts.jsonl")

PIN, POUT = 0.0288e-6, 0.288e-6
BASE = os.environ.get("LLM_BASE_URL", "").rstrip("/")
KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-flash")

SYSTEM = (
    "You audit near-duplicate removal in a web-crawl deduplication pipeline. "
    "You see a document that a pipeline deleted, and optionally the text of its closest "
    "counterpart (the document with the highest exact 5-gram Jaccard similarity to it). "
    "Classify what the deleted document contains and whether anything is lost. "
    "Return only a JSON object, no prose outside it."
)

INSTRUCTIONS = """Category definitions (choose exactly one):
- "template": boilerplate, navigation, listing/category page, auto-generated index, error page, cookie/legal notice, or a page whose text is mostly repeated site furniture. It carries no substantive information.
- "repost": a near-verbatim copy (or minor-format copy) of the counterpart or of a widely syndicated text. All of its substantive content is present in the counterpart; it adds no fact.
- "partial_rewrite": an edited, abridged, expanded, or paraphrased version of the counterpart. Its substantive facts are recoverable from the counterpart; only wording or ordering differs.
- "unique_facts": the counterpart exists (similarity is high enough that a deduplicator would treat them as near-duplicates), but this document contains at least one specific substantive fact (a number, date, name, place, decision, event, measurement) that is absent from the counterpart. Deleting it loses that fact.
- "unique_content": no close counterpart (the maximum similarity is low) and the document carries substantive content that exists nowhere in the supplied counterpart. Deleting it loses information.

Precedence: template > repost > unique_facts > partial_rewrite > unique_content.
A number that appears only in a footer, date stamp, page counter, price widget, or URL is NOT a substantive fact.
SLOT RULE: when a city, business name, product name, model number, phone number or event title is substituted into otherwise identical boilerplate (an SEO city page, a product-catalogue entry, an event listing), that is a slot, not a unique fact: label "template" if the page carries no other substantive content.
A short generic sentence about a business or product (e.g. "X offers quality service in Y") is boilerplate, not substantive content.
Judge only from the text shown; do not use outside knowledge about the website.

Return JSON with these keys:
{"label": one of the five strings,
 "confidence": number between 0 and 1,
 "reason": "at most 30 words, pointing at the decisive evidence",
 "lost_fact": "if label is unique_facts or unique_content: one self-contained sentence stating the fact or content that would be lost; otherwise null",
 "qa": if lost_fact is not null and the text supports it, {"question": "a question answerable only from this document", "answer": "a short answer"} ; otherwise null}
"""


def clip(text, head, tail):
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= head + tail:
        return text
    return text[:head] + " [...] " + text[-tail:]


def build_prompt(r):
    parts = [INSTRUCTIONS, ""]
    parts.append(f"CORPUS: {r['corpus']}   DOCUMENT ID: {r['id']}   WORDS: {r['nwords']}")
    if r.get("domain"):
        parts.append(f"DOMAIN: {r['domain']}")
    parts.append("DELETED DOCUMENT TEXT:")
    parts.append(clip(r.get("text", ""), 1800, 1200))
    nb = r.get("neighbor")
    if nb and nb.get("text"):
        parts.append("")
        parts.append(f"CLOSEST COUNTERPART (exact 5-gram Jaccard {nb['j']:.3f}), DOCUMENT ID {nb['id']}:")
        parts.append(clip(nb["text"], 1000, 700))
    else:
        parts.append("")
        parts.append("CLOSEST COUNTERPART: none with Jaccard >= 0.4 in the corpus.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


_lock = threading.Lock()
_last = [0.0]


def throttle():
    with _lock:
        now = time.time()
        wait = _last[0] + 2.3 - now
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def call(messages, timeout=300):
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0.0,
                       "max_tokens": 800, "enable_thinking": False}).encode()
    for attempt in range(4):
        throttle()
        try:
            req = urllib.request.Request(
                f"{BASE}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except urllib.error.HTTPError as e:
            txt = e.read().decode(errors="ignore")
            if "budget" in txt.lower():
                raise SystemExit(f"budget_exceeded: {txt[:300]}")
            if e.code == 429 or e.code >= 500:
                if attempt < 3:
                    time.sleep(15 * (attempt + 1))
                    continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < 3:
                time.sleep(15 * (attempt + 1))
                continue
            raise


def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--order", default="",
                    help="comma-separated stratum priority, e.g. c4_union_no_nb,wet_disputed")
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(SAMPLES)]
    done = set()
    if os.path.exists(LABELS):
        for x in open(LABELS):
            try:
                done.add(json.loads(x)["id"])
            except Exception:
                pass
    todo = [r for r in rows if r["id"] not in done]
    if a.order:
        pri = {s: i for i, s in enumerate(a.order.split(","))}
        todo.sort(key=lambda r: pri.get(r["stratum"], len(pri)))
    if a.max_docs:
        todo = todo[:a.max_docs]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(rows)} samples, {len(done)} labelled, {len(todo)} to do", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    spent = [0.0]

    def work(r):
        msgs = build_prompt(r)
        resp = call(msgs)
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        cost = usage.get("prompt_tokens", 0) * PIN + usage.get("completion_tokens", 0) * POUT
        j = parse_json(content) or {}
        label = j.get("label")
        if label not in ("template", "repost", "partial_rewrite", "unique_facts", "unique_content"):
            label = None
        row = dict(id=r["id"], corpus=r["corpus"], stratum=r["stratum"], index=r["index"],
                   model=MODEL, label=label, confidence=j.get("confidence"),
                   reason=j.get("reason"), lost_fact=j.get("lost_fact"), qa=j.get("qa"),
                   cost_usd=cost, prompt_tokens=usage.get("prompt_tokens"),
                   completion_tokens=usage.get("completion_tokens"),
                   raw=content, prompt_chars=sum(len(m["content"]) for m in msgs),
                   generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(LABELS, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        with open(PROMPT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(id=r["id"], messages=msgs), ensure_ascii=False) + "\n")
        spent[0] += cost
        return row

    t0 = time.time()
    ok = bad = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                row = fut.result()
                ok += 1
                print(f"[{k}/{len(todo)}] {r['id']} {row['label']} ${spent[0]:.4f} "
                      f"elapsed {time.time()-t0:.0f}s", flush=True)
            except SystemExit:
                raise
            except Exception as e:
                bad += 1
                print(f"[{k}/{len(todo)}] FAIL {r['id']}: {type(e).__name__} {e}", flush=True)
    print(f"done ok={ok} failed={bad} spend_this_run=${spent[0]:.4f} "
          f"elapsed={(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
