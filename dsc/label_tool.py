"""Human-in-the-loop graded relevance labeling for the DSC golden set.

Build a queue (BM25+HDC top-k per query, gold forced in), serve a tiny local web page,
save labels as JSONL, export a lab_eval-compatible golden JSON with graded gains.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .fusion import wrrf
from .hdc import cosine_q, vector
from .lexical import BM25Index


def _docs_by_id(data: dict) -> dict:
    return {d["id"]: d for d in data["docs"]}


def _rank(query: str, docs: list[dict], top_k: int) -> list[str]:
    bm = BM25Index()
    for d in docs:
        bm.add(d["id"], d["text"])
    bm.finalize()
    lex = [r["id"] for r in bm.search(query, k=len(docs))]
    qv = vector(query)
    hdc = [i for c, i in sorted(((cosine_q(qv, vector(d["text"])), d["id"]) for d in docs),
                               key=lambda x: (-x[0], x[1]))]
    fused = wrrf([lex, hdc], [1.0, 0.8], k=10)
    return [i for i, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:top_k]]


def build_queue(data: dict, top_k: int = 4, max_queries: int = 100) -> dict:
    docs = data["docs"]
    by_id = _docs_by_id(data)
    queries = [q for q in data["queries"] if q.get("origin")] or data["queries"]
    items = []
    for q in queries[:max_queries]:
        gold = list(q.get("gold") or [])
        ids = _rank(q["q"], docs, top_k)
        for g in gold:
            if g not in ids:
                ids.insert(0, g)
        gold_cats = {by_id[g].get("cat") for g in gold if g in by_id}
        cand = []
        for did in ids:
            d = by_id.get(did)
            if not d:
                continue
            rel = 2 if did in gold else (1 if d.get("cat") in gold_cats else 0)
            cand.append({"id": did, "text": d.get("text", ""), "rel": rel})
        items.append({"q": q["q"], "gold": gold, "docs": cand})
    return {"docs": docs, "items": items}


def load_labels(path: str) -> dict:
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("q"):
                    out[rec["q"]] = {k: int(v) for k, v in (rec.get("rels") or {}).items()}
    except OSError:
        pass
    return out


def save_label(path: str, q: str, rels: dict) -> None:
    rec = {"q": q, "rels": rels, "ts": time.time()}
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def to_golden(queue: dict, labels: dict, min_gold_gain: int = 2) -> dict:
    queries = []
    for item in queue["items"]:
        rels = labels.get(item["q"])
        if rels is None:
            rels = {d["id"]: int(d.get("rel", 0)) for d in item["docs"]}
            labeled = False
        else:
            labeled = True
        graded = {k: int(v) for k, v in rels.items() if int(v) > 0}
        gold = [k for k, v in graded.items() if v >= min_gold_gain]
        if not gold:
            continue
        queries.append({"q": item["q"], "gold": gold, "graded": graded,
                        "cat": "human", "origin": True, "labeled": labeled})
    return {"human_labeled": True, "docs": queue["docs"], "queries": queries}


def _page(queue: dict, labels: dict, idx: int) -> str:
    n = len(queue["items"])
    if idx >= n:
        done = sum(1 for it in queue["items"] if it["q"] in labels)
        return ("<!doctype html><meta charset='utf-8'><body style='font-family:system-ui;max-width:720px;margin:40px auto'>"
                "<h2>Labeling done</h2><p>%d/%d labeled.</p><p><a href='/download'>Download golden_graded.json</a></p></body>"
                % (done, n))
    item = queue["items"][idx]
    rels = labels.get(item["q"], {d["id"]: int(d.get("rel", 0)) for d in item["docs"]})
    rows = []
    for d in item["docs"]:
        did = html.escape(str(d["id"]))
        cur = int(rels.get(d["id"], 0))
        radios = "".join(
            "<label style='margin-right:14px'><input type=radio name='rel_%s' value='%d' %s> %d</label>"
            % (did, v, "checked" if cur == v else "", v) for v in (0, 1, 2))
        rows.append("<div style='border:1px solid #ddd;border-radius:8px;padding:10px;margin:8px 0'>"
                    "<div>%s</div><p style='font-size:14px;color:#333'>%s</p></div>"
                    % (radios, html.escape(d["text"][:700])))
    done = sum(1 for it in queue["items"] if it["q"] in labels)
    return f"""<!doctype html><meta charset='utf-8'><title>DSC labels</title>
<body style='font-family:system-ui;max-width:820px;margin:30px auto;color:#111'>
<div style='display:flex;justify-content:space-between'><b>DSC labels</b><span>{idx+1}/{n} · {done} labeled</span></div>
<h2 style='margin:12px 0 4px'>{html.escape(item['q'])}</h2>
<p style='color:#666;font-size:13px'>Gold: {', '.join(item['gold']) or '(none)'} · 2=relevant, 1=partial, 0=not</p>
{''.join(rows)}
<div style='margin-top:16px'><button onclick='save()' style='font-size:16px;padding:10px 18px'>Save &amp; next (Enter)</button>
<a href='/download' style='margin-left:16px'>export</a></div>
<script>
const I={idx};
function save(){{
  const rels={{}};
  document.querySelectorAll("input[type=radio]:checked").forEach(x=>{{
    const id=x.name.replace("rel_",""); rels[id]=parseInt(x.value);
  }});
  fetch("/save",{{method:"POST",headers:{{"Content-Type":"application/json"}},
    body:JSON.stringify({{i:I, q:{json.dumps(item['q'])}, rels:rels}})}})
    .then(()=>location="/?i="+(I+1));
}}
document.addEventListener("keydown",e=>{{ if(e.key==="Enter") save(); }});
</script></body>"""


def make_handler(queue: dict, labels: dict, labels_path: str, data: dict):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/download"):
                golden = to_golden(queue, labels)
                self._send(200, json.dumps(golden, ensure_ascii=False, indent=1),
                           "application/json", {"Content-Disposition": "attachment; filename=golden_graded.json"})
                return
            idx = 0
            if "i=" in self.path:
                try:
                    idx = int(self.path.split("i=")[1].split("&")[0])
                except ValueError:
                    idx = 0
            self._send(200, _page(queue, labels, max(0, idx)))

        def do_POST(self):
            if not self.path.startswith("/save"):
                self._send(404, "not found")
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                q = body["q"]
                rels = {str(k): int(v) for k, v in (body.get("rels") or {}).items()}
            except (ValueError, KeyError, TypeError):
                self._send(400, "bad request")
                return
            labels[q] = rels
            save_label(labels_path, q, rels)
            self._send(200, json.dumps({"ok": True}))

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc.label_tool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "serve", "export"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="docs/dsc/golden_synth300.json")
        p.add_argument("--labels", default="docs/dsc/labels.jsonl")
        p.add_argument("--out", default="docs/dsc/golden_graded.json")
        p.add_argument("--top-k", type=int, default=4)
        p.add_argument("--max-queries", type=int, default=100)
        p.add_argument("--port", type=int, default=8977)
        p.add_argument("--queue", default="docs/dsc/label_queue.json")
    args = ap.parse_args(argv)
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    if args.cmd == "build":
        q = build_queue(data, args.top_k, args.max_queries)
        with open(args.queue, "w", encoding="utf-8") as f:
            json.dump(q, f, ensure_ascii=False, indent=1)
        print(json.dumps({"items": len(q["items"]), "out": args.queue}))
        return 0
    if args.cmd == "export":
        try:
            with open(args.queue, encoding="utf-8") as f:
                queue = json.load(f)
        except OSError:
            queue = build_queue(data, args.top_k, args.max_queries)
        labels = load_labels(args.labels)
        golden = to_golden(queue, labels)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(golden, f, ensure_ascii=False, indent=1)
        print(json.dumps({"queries": len(golden["queries"]),
                          "labeled": sum(1 for q in golden["queries"] if q.get("labeled")),
                          "out": args.out}))
        return 0
    # serve
    try:
        with open(args.queue, encoding="utf-8") as f:
            queue = json.load(f)
    except OSError:
        queue = build_queue(data, args.top_k, args.max_queries)
    labels = load_labels(args.labels)
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(queue, labels, args.labels, data))
    except OSError as e:
        print("label tool: port %d busy (%s); try --port 8977" % (args.port, e), file=sys.stderr)
        return 2
    print("label tool: http://127.0.0.1:%d/  (Ctrl+C to stop)" % args.port, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
