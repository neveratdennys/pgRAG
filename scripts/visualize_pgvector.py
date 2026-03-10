#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import psycopg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an HTML visualization of pgvector chunk similarity."
    )
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--dbname", default="pgrag", help="PostgreSQL database name")
    parser.add_argument("--user", default="pgrag", help="PostgreSQL user")
    parser.add_argument(
        "--password",
        default=os.getenv("PGRAG_DB_PASSWORD"),
        help="PostgreSQL password (or use PGRAG_DB_PASSWORD env var)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.20,
        help="Minimum cosine similarity to draw an edge in graph view",
    )
    parser.add_argument(
        "--out",
        default="semantic_viz.html",
        help="Output HTML path",
    )
    return parser.parse_args()


def parse_vector_text(text: str) -> list[float]:
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    if not stripped:
        return []
    return [float(part) for part in stripped.split(",")]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for av, bv in zip(a, b):
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def color_for_source(source: str) -> str:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    idx = abs(hash(source)) % len(palette)
    return palette[idx]


def fetch_chunks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.password:
        raise ValueError(
            "No DB password provided. Set --password or PGRAG_DB_PASSWORD."
        )
    with psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        sslmode="disable",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source, title, chunk_index, LEFT(content, 220) AS preview, embedding::text
                FROM chunks
                ORDER BY source, chunk_index;
                """
            )
            rows = cur.fetchall()

    chunks: list[dict[str, Any]] = []
    for row in rows:
        chunk_id, source, title, chunk_index, preview, embedding_text = row
        chunks.append(
            {
                "id": int(chunk_id),
                "source": str(source),
                "title": str(title),
                "chunk_index": int(chunk_index),
                "preview": str(preview or "").replace("\n", " ").strip(),
                "embedding": parse_vector_text(str(embedding_text)),
            }
        )
    return chunks


def build_similarity(chunks: list[dict[str, Any]]) -> list[list[float]]:
    n = len(chunks)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = cosine_similarity(chunks[i]["embedding"], chunks[j]["embedding"])
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix


def build_html(chunks: list[dict[str, Any]], sim: list[list[float]], min_edge: float) -> str:
    nodes = []
    for c in chunks:
        label = f"{c['title']}#{c['chunk_index']}"
        nodes.append(
            {
                "id": c["id"],
                "label": label,
                "source": c["source"],
                "preview": c["preview"],
                "color": color_for_source(c["source"]),
            }
        )

    links = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            w = sim[i][j]
            if w >= min_edge:
                links.append(
                    {
                        "a": chunks[i]["id"],
                        "b": chunks[j]["id"],
                        "w": round(w, 4),
                    }
                )

    max_dim = len(chunks[0]["embedding"]) if chunks else 0

    matrix_rows = []
    for i, row in enumerate(sim):
        cells = "".join(
            f"<td class='hcell' data-v='{value:.6f}'>{value:.3f}</td>" for value in row
        )
        matrix_rows.append(f"<tr><th>#{i}</th>{cells}</tr>")
    matrix_html = "\n".join(matrix_rows)

    safe_nodes_json = json.dumps(nodes)
    safe_links_json = json.dumps(links)

    legend_sources = sorted({c["source"] for c in chunks})
    legend_html = "".join(
        f"<li><span class='dot' style='background:{color_for_source(src)}'></span>{html.escape(src)}</li>"
        for src in legend_sources
    )

    chunk_rows = []
    for idx, c in enumerate(chunks):
        chunk_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{c['id']}</td>"
            f"<td>{html.escape(c['source'])}</td>"
            f"<td>{html.escape(c['title'])}</td>"
            f"<td>{c['chunk_index']}</td>"
            f"<td>{html.escape(c['preview'])}</td>"
            "</tr>"
        )
    chunk_table_html = "\n".join(chunk_rows)

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>pgRAG Semantic Visualization</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; color: #222; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .meta {{ margin-bottom: 18px; color: #555; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .panel {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; vertical-align: top; }}
    .hcell {{ text-align: center; font-variant-numeric: tabular-nums; }}
    #graph {{ width: 100%; height: 520px; border: 1px solid #ddd; border-radius: 6px; }}
    .legend {{ list-style: none; margin: 8px 0 0; padding: 0; display: flex; gap: 12px; flex-wrap: wrap; }}
    .legend .dot {{ width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 6px; }}
    .foot {{ margin-top: 14px; color: #666; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>pgRAG Semantic Visualization</h1>
  <div class="meta">
    chunks: <b>{len(chunks)}</b> |
    embedding dimensions: <b>{max_dim}</b> |
    graph edge threshold: <b>{min_edge:.2f}</b>
  </div>

  <div class="grid">
    <div class="panel">
      <h2>Similarity Graph</h2>
      <svg id="graph" viewBox="0 0 920 520"></svg>
      <ul class="legend">{legend_html}</ul>
    </div>

    <div class="panel">
      <h2>Cosine Similarity Matrix</h2>
      <table>
        <thead>
          <tr><th>idx</th>{"".join(f"<th>#{i}</th>" for i in range(len(chunks)))}</tr>
        </thead>
        <tbody>
          {matrix_html}
        </tbody>
      </table>
    </div>
  </div>

  <div class="panel" style="margin-top:18px;">
    <h2>Chunk Metadata</h2>
    <table>
      <thead>
        <tr>
          <th>idx</th><th>chunk_id</th><th>source</th><th>title</th><th>chunk_index</th><th>preview</th>
        </tr>
      </thead>
      <tbody>{chunk_table_html}</tbody>
    </table>
  </div>

  <div class="foot">
    Graph layout is a lightweight force simulation driven by cosine similarity edge weights.
  </div>

<script>
const nodes = {safe_nodes_json};
const links = {safe_links_json};

const W = 920, H = 520;
const svg = document.getElementById("graph");

const nodeById = new Map(nodes.map(n => [n.id, n]));
nodes.forEach((n, i) => {{
  n.x = 120 + (i * 90) % (W - 240);
  n.y = 120 + (i * 70) % (H - 240);
  n.vx = 0;
  n.vy = 0;
}});

function step() {{
  const repulsion = 1700;
  const spring = 0.08;
  const damping = 0.88;

  for (let i = 0; i < nodes.length; i++) {{
    for (let j = i + 1; j < nodes.length; j++) {{
      const a = nodes[i], b = nodes[j];
      let dx = b.x - a.x, dy = b.y - a.y;
      const d2 = dx*dx + dy*dy + 1e-3;
      const force = repulsion / d2;
      const d = Math.sqrt(d2);
      dx /= d; dy /= d;
      a.vx -= force * dx; a.vy -= force * dy;
      b.vx += force * dx; b.vy += force * dy;
    }}
  }}

  links.forEach(l => {{
    const a = nodeById.get(l.a), b = nodeById.get(l.b);
    if (!a || !b) return;
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx*dx + dy*dy) + 1e-3;
    const target = 220 - (l.w * 120);
    const f = spring * (d - target) * Math.max(l.w, 0.08);
    const ux = dx / d, uy = dy / d;
    a.vx += f * ux; a.vy += f * uy;
    b.vx -= f * ux; b.vy -= f * uy;
  }});

  nodes.forEach(n => {{
    n.vx *= damping; n.vy *= damping;
    n.x += n.vx; n.y += n.vy;
    n.x = Math.max(22, Math.min(W - 22, n.x));
    n.y = Math.max(22, Math.min(H - 22, n.y));
  }});
}}

for (let i = 0; i < 220; i++) step();

function render() {{
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  links.forEach(l => {{
    const a = nodeById.get(l.a), b = nodeById.get(l.b);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
    line.setAttribute("stroke", "#9aa0a6");
    line.setAttribute("stroke-opacity", String(Math.min(0.85, Math.max(0.12, l.w))));
    line.setAttribute("stroke-width", String(0.8 + l.w * 3));
    svg.appendChild(line);
  }});

  nodes.forEach((n, i) => {{
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", n.x); c.setAttribute("cy", n.y);
    c.setAttribute("r", "8");
    c.setAttribute("fill", n.color);
    c.setAttribute("stroke", "#333");
    c.setAttribute("stroke-width", "0.8");
    g.appendChild(c);

    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", n.x + 10); t.setAttribute("y", n.y + 4);
    t.setAttribute("font-size", "11");
    t.textContent = `#${{i}} ${{n.label}}`;
    g.appendChild(t);

    svg.appendChild(g);
  }});
}}
render();

document.querySelectorAll(".hcell").forEach(td => {{
  const v = parseFloat(td.dataset.v || "0");
  const clamped = Math.max(-1, Math.min(1, v));
  const t = (clamped + 1) / 2; // [-1,1] -> [0,1]
  const r = Math.round(255 * (1 - t));
  const g = Math.round(230 * t + 25);
  const b = Math.round(255 * (1 - t));
  td.style.background = `rgb(${{r}},${{g}},${{b}})`;
}});
</script>
</body>
</html>"""


def main() -> int:
    args = parse_args()
    chunks = fetch_chunks(args)
    if not chunks:
        print("No rows found in chunks table.")
        return 1

    sim = build_similarity(chunks)
    output_path = Path(args.out).resolve()
    output_path.write_text(build_html(chunks, sim, args.min_edge), encoding="utf-8")

    print(f"wrote visualization: {output_path}")
    print(f"chunks={len(chunks)} dims={len(chunks[0]['embedding'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
