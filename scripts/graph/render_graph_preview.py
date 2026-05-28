"""
Quick graph preview from extracted triple JSON files.

Usage:
  python scripts/graph/render_graph_preview.py --files f1.json f2.json --out preview.html
  python scripts/graph/render_graph_preview.py --db data/knowledge_graph/triples.db --out preview.html
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from pyvis.network import Network

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.store_graph_triples import _is_valid, _canonicalise


def load_from_files(paths: list[str]) -> list[dict]:
    triples = []
    for p in paths:
        triples += json.loads(Path(p).read_text(encoding="utf-8"))
    return triples


def load_from_db(db_path: str, min_docs: int = 1) -> list[dict]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        SELECT subject, relationship, object, verbatim_quote, conditional, source_doc
        FROM triples
        WHERE human_reviewed >= 0
    """)
    rows = cur.fetchall()
    con.close()
    return [
        {"subject": r[0], "relationship": r[1], "object": r[2],
         "verbatim_quote": r[3], "conditional": r[4], "source_doc": r[5]}
        for r in rows
    ]


def build_graph(triples: list[dict], min_edge_count: int = 1, top_nodes: int = 0) -> Network:
    # Filter ungrounded
    triples = [t for t in triples if _is_valid(
        t.get("subject", ""), t.get("relationship", ""),
        t.get("object", ""), t.get("verbatim_quote", "")
    )]

    # Group by (subject, object) pair
    edges: dict[tuple, list] = defaultdict(list)
    for t in triples:
        # Canonicalise entity names so "Estrogen" and "excess estrogen" merge
        t["subject"] = _canonicalise(t.get("subject") or "")
        t["object"] = _canonicalise(t.get("object") or "")
        key = (t["subject"], t["object"])
        edges[key].append(t)

    # Filter by min_edge_count first (dedup by quote happens per-edge below)
    # Then drop nodes with degree < min_degree to remove isolated one-offs
    from collections import Counter
    if top_nodes > 0:
        degree: Counter = Counter()
        for subj, obj in edges:
            degree[subj] += 1
            degree[obj] += 1
        keep = {name for name, _ in degree.most_common(top_nodes)}
        edges = {k: v for k, v in edges.items() if k[0] in keep and k[1] in keep}

    # Build pyvis graph
    net = Network(
        height="800px", width="100%",
        bgcolor="#1a1a2e", font_color="#e0e0e0",
        directed=True,
    )
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 150 }
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.15 },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
        "font": { "size": 10, "align": "middle" }
      },
      "nodes": {
        "font": { "size": 13 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    # Category → (shape, color)
    CATEGORIES = [
        (
            ["estrogen", "progesterone", "cortisol", "thyroid hormone", "serotonin",
             "prolactin", "insulin", "testosterone", "dhea", "pregnenolone",
             "aldosterone", "t3", "t4", "adrenalin", "dopamine", "melatonin",
             "glucagon", "oxytocin", "vasopressin"],
            "dot", "#ff6b9d",   # circle — hormones
        ),
        (
            ["vitamin", "magnesium", "calcium", "zinc", "aspirin", "caffeine",
             "niacinamide", "selenium", "copper", "iron", "sodium", "potassium",
             "glucose", "sugar", "fructose", "gelatin", "glycine", "taurine",
             "coconut oil", "butter", "pufa", "linoleic", "omega"],
            "square", "#ffd166",  # square — nutrients/substances
        ),
        (
            ["cancer", "hypothyroid", "hyperthyroid", "inflammation", "aging",
             "dementia", "insomnia", "stress", "fibrosis", "obesity", "diabetes",
             "fatigue", "syndrome", "disease", "disorder", "deficiency"],
            "diamond", "#ef476f",  # diamond — conditions
        ),
        (
            ["liver", "brain", "mitochondria", "thyroid gland", "adrenal",
             "pituitary", "ovaries", "muscle", "heart", "kidney", "intestine",
             "gut", "cell", "neuron", "tissue", "blood"],
            "hexagon", "#06d6a0",  # hexagon — organs/tissues
        ),
    ]

    added_nodes: set[str] = set()

    def _classify(name: str) -> tuple[str, str]:
        name_l = name.lower()
        for keywords, shape, color in CATEGORIES:
            if any(w in name_l for w in keywords):
                return shape, color
        return "ellipse", "#4a9eff"  # default — other concepts

    def add_node(name: str):
        if name in added_nodes:
            return
        added_nodes.add(name)
        shape, color = _classify(name)
        net.add_node(name, label=name, color=color, shape=shape, title=name, size=18)

    for (subj, obj), group in edges.items():
        # Deduplicate by verbatim quote — same sentence generating multiple triples
        # (e.g. "produce and secrete") should count as one source, not two
        seen_quotes: set[str] = set()
        unique_group = []
        for t in group:
            q = (t.get("verbatim_quote") or "").strip()
            if q not in seen_quotes:
                seen_quotes.add(q)
                unique_group.append(t)

        if len(unique_group) < min_edge_count:
            continue

        # Use the canonical casing from the first triple
        subj_display = group[0]["subject"].strip()
        obj_display = group[0]["object"].strip()
        add_node(subj_display)
        add_node(obj_display)

        # Build tooltip: distinct quotes only, with source doc
        rel = group[0]["relationship"]
        quote_lines = []
        for t in unique_group:
            q = (t.get("verbatim_quote") or "").strip()
            cond = t.get("conditional")
            doc = (t.get("source_doc") or "")
            entry = f'"{q}"'
            if cond:
                entry += f" [{cond}]"
            if doc:
                entry += f"\n— {doc[:60]}"
            quote_lines.append(entry)

        n_sources = len(unique_group)
        tooltip = f"{subj_display} → {rel} → {obj_display} ({n_sources} source{'s' if n_sources>1 else ''})\n\n" + "\n\n".join(quote_lines)
        width = 1 + n_sources  # thicker = more distinct sources

        net.add_edge(
            subj_display, obj_display,
            label=rel,
            title=tooltip,
            width=width,
            color="#888888" if n_sources == 1 else "#4a9eff",
        )

    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="JSON files to load")
    parser.add_argument("--db", help="SQLite DB to load from")
    parser.add_argument("--out", default="data/knowledge_graph/preview.html")
    parser.add_argument("--min-edges", type=int, default=1, help="Minimum distinct-source count to show an edge")
    parser.add_argument("--top-nodes", type=int, default=0, help="Only show the N most-connected entities (0 = all)")
    args = parser.parse_args()

    if args.files:
        triples = load_from_files(args.files)
    elif args.db:
        triples = load_from_db(args.db)
    else:
        print("Provide --files or --db")
        sys.exit(1)

    net = build_graph(triples, min_edge_count=args.min_edges, top_nodes=args.top_nodes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out))

    legend_html = """
<style>
#graph-legend {
  position: fixed; bottom: 18px; left: 18px; z-index: 9999;
  background: rgba(26,26,46,0.92); border: 1px solid #444;
  border-radius: 8px; padding: 12px 16px; color: #e0e0e0;
  font-family: sans-serif; font-size: 13px; line-height: 1.7;
  pointer-events: none;
}
#graph-legend b { display: block; margin-bottom: 4px; font-size: 14px; }
.leg-row { display: flex; align-items: center; gap: 8px; }
.leg-icon { width: 18px; height: 18px; flex-shrink: 0; }
</style>
<div id="graph-legend">
  <b>Node types</b>
  <div class="leg-row">
    <svg class="leg-icon" viewBox="0 0 18 18"><circle cx="9" cy="9" r="8" fill="#ff6b9d"/></svg>
    Hormone / signalling molecule
  </div>
  <div class="leg-row">
    <svg class="leg-icon" viewBox="0 0 18 18"><rect x="1" y="1" width="16" height="16" fill="#ffd166"/></svg>
    Nutrient / substance
  </div>
  <div class="leg-row">
    <svg class="leg-icon" viewBox="0 0 18 18"><polygon points="9,1 17,17 1,17" fill="#ef476f"/></svg>
    Condition / disease
  </div>
  <div class="leg-row">
    <svg class="leg-icon" viewBox="0 0 18 18"><polygon points="9,1 16,5 16,13 9,17 2,13 2,5" fill="#06d6a0"/></svg>
    Organ / tissue / cell
  </div>
  <div class="leg-row">
    <svg class="leg-icon" viewBox="0 0 18 18"><ellipse cx="9" cy="9" rx="8" ry="5" fill="#4a9eff"/></svg>
    Other concept
  </div>
</div>
"""
    html = out.read_text(encoding="utf-8")
    html = html.replace("</body>", legend_html + "\n</body>")
    out.write_text(html, encoding="utf-8")

    print(f"Graph written to {out.resolve()} ({len(net.nodes)} nodes, {len(net.edges)} edges)")


if __name__ == "__main__":
    main()
