"""
Financial Pattern Engine — NetworkX Graph Anomaly Scorer
Anti-CSAM Trafficking Detection Prototype

Ingests data/transactions.json and scores every recipient node on four
anomaly dimensions:

  1. Star Topology      — many unique payers converging on one recipient
                          in a short time window with near-identical amounts
  2. Payment Clustering — structuring (amounts just below thresholds),
                          multi-hop layering chains
  3. Geo Mismatch       — domestic→foreign→domestic round-trip flows
  4. Platform Hopping   — same payer/recipient pair across ≥3 platforms

Each dimension produces a sub-score in [0, 1].  The composite risk score
is a weighted sum, also clamped to [0, 1].

Outputs
-------
  data/risk_scores.json     — per-node scores + evidence
  data/report.txt           — human-readable summary report
  data/graph.gexf           — full transaction graph (Gephi-compatible)
"""

from __future__ import annotations

import json
import math
import statistics
import textwrap
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import networkx as nx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data")
INPUT_FILE = DATA_DIR / "transactions.json"

WEIGHTS = {
    "star_topology":      0.35,
    "payment_clustering": 0.25,
    "geo_mismatch":       0.25,
    "platform_hopping":   0.15,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"

# Star topology: time window (hours) and min unique payers to trigger
STAR_WINDOW_HOURS   = 2
STAR_MIN_PAYERS     = 5
STAR_AMOUNT_CV_MAX  = 0.12   # coefficient of variation — tight cluster = suspicious

# Structuring thresholds (USD) — amounts just below these are suspicious
STRUCTURING_THRESHOLDS = [200, 500, 1_000, 3_000, 5_000, 10_000]
STRUCTURING_MARGIN_PCT = 0.08   # within 8% below threshold

# Geo: set of country prefixes considered "domestic"
DOMESTIC_PREFIXES = {"US"}

# Platform hopping: min distinct platforms in a payer→recipient pair
HOP_MIN_PLATFORMS = 3

# Risk score threshold for "high risk" in report
HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_transactions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Parse timestamps
    for tx in data:
        tx["_dt"] = datetime.fromisoformat(tx["timestamp"])
    return data


def build_graph(transactions: list[dict]) -> nx.MultiDiGraph:
    """
    Directed multigraph: node = actor, edge = transaction.
    Edge attributes mirror the transaction record.
    """
    G = nx.MultiDiGraph()
    for tx in transactions:
        G.add_edge(
            tx["payer_id"],
            tx["recipient_id"],
            tx_id       = tx["tx_id"],
            amount      = tx["amount_usd"],
            timestamp   = tx["timestamp"],
            dt          = tx["_dt"],
            platform    = tx["platform"],
            payer_geo   = tx["payer_geo"],
            recipient_geo = tx["recipient_geo"],
            is_trafficking = tx["is_trafficking"],
            pattern_tags   = tx["pattern_tags"],
        )
    return G

# ---------------------------------------------------------------------------
# Scorer helpers
# ---------------------------------------------------------------------------

def _cv(values: list[float]) -> float:
    """Coefficient of variation (std/mean). Returns 0 for single values."""
    if len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    if m == 0:
        return 0.0
    return statistics.stdev(values) / m


def _sigmoid(x: float, midpoint: float = 0.5, steepness: float = 10.0) -> float:
    """Maps a raw count/ratio to (0,1) with a sigmoid curve."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))

# ---------------------------------------------------------------------------
# Dimension 1 — Star Topology
# ---------------------------------------------------------------------------

def score_star_topology(
    recipient: str, G: nx.MultiDiGraph
) -> tuple[float, dict]:
    """
    Look for bursts where many unique payers send to this recipient within
    STAR_WINDOW_HOURS with tightly clustered amounts.
    """
    in_edges = [
        data for _, _, data in G.in_edges(recipient, data=True)
    ]
    if not in_edges:
        return 0.0, {}

    # Sort by datetime
    in_edges.sort(key=lambda e: e["dt"])

    best_score = 0.0
    best_evidence: dict = {}

    # Sliding window
    for i, anchor in enumerate(in_edges):
        window_end = anchor["dt"] + timedelta(hours=STAR_WINDOW_HOURS)
        window_edges = [e for e in in_edges[i:] if e["dt"] <= window_end]
        unique_payers = {e for e in window_edges}   # each edge is a unique dict
        payer_ids = list({
            src for src, _, data in G.in_edges(recipient, data=True)
            if data["dt"] >= anchor["dt"] and data["dt"] <= window_end
        })

        n_payers = len(payer_ids)
        if n_payers < STAR_MIN_PAYERS:
            continue

        amounts = [e["amount"] for e in window_edges]
        cv = _cv(amounts)
        amount_tightness = 1.0 - min(cv / STAR_AMOUNT_CV_MAX, 1.0)  # 1 = very tight

        # Score: payer count contribution + amount tightness
        payer_score  = _sigmoid(n_payers, midpoint=STAR_MIN_PAYERS, steepness=0.4)
        window_score = payer_score * 0.6 + amount_tightness * 0.4

        if window_score > best_score:
            best_score = window_score
            best_evidence = {
                "window_start": anchor["dt"].isoformat(),
                "window_end":   window_end.isoformat(),
                "unique_payers": n_payers,
                "amount_cv":    round(cv, 4),
                "mean_amount":  round(statistics.mean(amounts), 2),
            }

    return _clamp(best_score), best_evidence

# ---------------------------------------------------------------------------
# Dimension 2 — Payment Clustering (Structuring + Layering)
# ---------------------------------------------------------------------------

def score_payment_clustering(
    recipient: str, G: nx.MultiDiGraph, all_transactions: list[dict]
) -> tuple[float, dict]:
    """
    Two sub-signals:
      a) Structuring: fraction of incoming amounts just below a threshold
      b) Layering:    recipient also appears as a payer in a quick chain
    """
    in_edges = [data for _, _, data in G.in_edges(recipient, data=True)]
    amounts  = [e["amount"] for e in in_edges]
    if not amounts:
        return 0.0, {}

    # --- Structuring ---
    structured_count = 0
    for amt in amounts:
        for thresh in STRUCTURING_THRESHOLDS:
            lower = thresh * (1 - STRUCTURING_MARGIN_PCT)
            if lower <= amt < thresh:
                structured_count += 1
                break
    structuring_ratio = structured_count / len(amounts)

    # --- Layering: does this recipient quickly pass money on? ---
    out_edges = [data for _, _, data in G.out_edges(recipient, data=True)]
    layering_score = 0.0
    if in_edges and out_edges:
        in_times  = [e["dt"] for e in in_edges]
        out_times = [e["dt"] for e in out_edges]
        # Fraction of out-edges that occur within 4 hours of an in-edge
        quick_pass = 0
        for ot in out_times:
            if any(abs((ot - it).total_seconds()) < 4 * 3600 for it in in_times):
                quick_pass += 1
        layering_score = quick_pass / len(out_times)

    combined = structuring_ratio * 0.6 + layering_score * 0.4
    evidence = {
        "structured_txs":    structured_count,
        "total_incoming_txs": len(amounts),
        "structuring_ratio": round(structuring_ratio, 3),
        "layering_score":    round(layering_score, 3),
    }
    return _clamp(combined), evidence

# ---------------------------------------------------------------------------
# Dimension 3 — Geo Mismatch
# ---------------------------------------------------------------------------

def _is_domestic(geo: str) -> bool:
    return any(geo.startswith(p) for p in DOMESTIC_PREFIXES)


def score_geo_mismatch(
    recipient: str, G: nx.MultiDiGraph
) -> tuple[float, dict]:
    """
    Signals:
      a) Incoming from domestic payers, outgoing to foreign (or vice-versa)
      b) Recipient geo differs from majority of payer geos
      c) Fast round-trip: domestic→foreign→domestic within 24h
    """
    in_edges  = [data for _, _, data in G.in_edges(recipient, data=True)]
    out_edges = [data for _, _, data in G.out_edges(recipient, data=True)]

    if not in_edges:
        return 0.0, {}

    # Payer geo distribution
    payer_geos      = [e["payer_geo"] for e in in_edges]
    domestic_in     = sum(1 for g in payer_geos if _is_domestic(g))
    foreign_in      = len(payer_geos) - domestic_in

    # Recipient geo of this node's outgoing transactions
    recipient_out_geos = [e["recipient_geo"] for e in out_edges]
    foreign_out = sum(1 for g in recipient_out_geos if not _is_domestic(g))

    # Cross-border ratio: how often does money cross a border at this node?
    cross_border_in  = foreign_in  / len(in_edges)
    cross_border_out = (foreign_out / len(out_edges)) if out_edges else 0.0

    # Round-trip detection: domestic in → foreign out within 24h
    round_trips = 0
    for ie in in_edges:
        if not _is_domestic(ie["payer_geo"]):
            continue
        for oe in out_edges:
            if _is_domestic(oe["recipient_geo"]):
                continue
            gap_h = abs((oe["dt"] - ie["dt"]).total_seconds()) / 3600
            if gap_h <= 24:
                round_trips += 1
                break   # one round-trip per in-edge is enough

    round_trip_ratio = round_trips / len(in_edges)

    combined = (
        cross_border_in  * 0.30 +
        cross_border_out * 0.30 +
        round_trip_ratio * 0.40
    )
    evidence = {
        "domestic_payers":   domestic_in,
        "foreign_payers":    foreign_in,
        "foreign_out_txs":   foreign_out,
        "round_trips_24h":   round_trips,
        "cross_border_in_ratio":  round(cross_border_in, 3),
        "cross_border_out_ratio": round(cross_border_out, 3),
        "round_trip_ratio":       round(round_trip_ratio, 3),
    }
    return _clamp(combined), evidence

# ---------------------------------------------------------------------------
# Dimension 4 — Platform Hopping
# ---------------------------------------------------------------------------

def score_platform_hopping(
    recipient: str, G: nx.MultiDiGraph
) -> tuple[float, dict]:
    """
    For each unique payer→recipient pair, count distinct platforms used.
    Score rises with the max number of platforms seen in any single pair.
    Also flag if the recipient itself receives on many platforms overall.
    """
    # Per-payer platform sets
    payer_platforms: dict[str, set[str]] = defaultdict(set)
    for src, _, data in G.in_edges(recipient, data=True):
        payer_platforms[src].add(data["platform"])

    if not payer_platforms:
        return 0.0, {}

    max_platforms_per_payer = max(len(ps) for ps in payer_platforms.values())
    multi_platform_payers   = sum(
        1 for ps in payer_platforms.values() if len(ps) >= HOP_MIN_PLATFORMS
    )

    # Overall platform diversity for this recipient
    all_platforms = set().union(*payer_platforms.values())
    _all_known_platforms = 7   # matches PLATFORMS list in data generator
    platform_diversity = len(all_platforms) / _all_known_platforms

    pair_score      = _sigmoid(max_platforms_per_payer, midpoint=HOP_MIN_PLATFORMS, steepness=1.5)
    diversity_score = _clamp(platform_diversity)

    combined = pair_score * 0.70 + diversity_score * 0.30
    evidence = {
        "unique_payers":           len(payer_platforms),
        "max_platforms_per_payer": max_platforms_per_payer,
        "multi_platform_payers":   multi_platform_payers,
        "distinct_platforms_total": len(all_platforms),
        "platforms":               sorted(all_platforms),
    }
    return _clamp(combined), evidence

# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def score_recipient(
    recipient: str,
    G: nx.MultiDiGraph,
    transactions: list[dict],
) -> dict:
    s_star,  ev_star  = score_star_topology(recipient, G)
    s_clust, ev_clust = score_payment_clustering(recipient, G, transactions)
    s_geo,   ev_geo   = score_geo_mismatch(recipient, G)
    s_hop,   ev_hop   = score_platform_hopping(recipient, G)

    composite = _clamp(
        s_star  * WEIGHTS["star_topology"]      +
        s_clust * WEIGHTS["payment_clustering"] +
        s_geo   * WEIGHTS["geo_mismatch"]       +
        s_hop   * WEIGHTS["platform_hopping"]
    )

    return {
        "node_id":         recipient,
        "composite_score": round(composite, 4),
        "dimension_scores": {
            "star_topology":      round(s_star,  4),
            "payment_clustering": round(s_clust, 4),
            "geo_mismatch":       round(s_geo,   4),
            "platform_hopping":   round(s_hop,   4),
        },
        "evidence": {
            "star_topology":      ev_star,
            "payment_clustering": ev_clust,
            "geo_mismatch":       ev_geo,
            "platform_hopping":   ev_hop,
        },
        "in_degree":  G.in_degree(recipient),
        "out_degree": G.out_degree(recipient),
        "total_received_usd": round(
            sum(d["amount"] for _, _, d in G.in_edges(recipient, data=True)), 2
        ),
    }

# ---------------------------------------------------------------------------
# Graph-level metrics
# ---------------------------------------------------------------------------

def graph_summary(G: nx.MultiDiGraph) -> dict:
    return {
        "nodes":          G.number_of_nodes(),
        "edges":          G.number_of_edges(),
        "density":        round(nx.density(G), 6),
        "weakly_connected_components": nx.number_weakly_connected_components(G),
        "avg_in_degree":  round(
            sum(d for _, d in G.in_degree()) / max(G.number_of_nodes(), 1), 2
        ),
    }

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

RISK_LABEL = {
    "high":   "HIGH RISK",
    "medium": "MEDIUM RISK",
    "low":    "low risk",
}


def _risk_tier(score: float) -> str:
    if score >= HIGH_RISK_THRESHOLD:
        return "high"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def build_report(
    scored_nodes: list[dict],
    graph_meta: dict,
    transactions: list[dict],
) -> str:
    total       = len(scored_nodes)
    high_risk   = [n for n in scored_nodes if _risk_tier(n["composite_score"]) == "high"]
    medium_risk = [n for n in scored_nodes if _risk_tier(n["composite_score"]) == "medium"]

    # Ground-truth recall (only possible because data is synthetic)
    flagged_recipients = {
        tx["recipient_id"] for tx in transactions if tx["is_trafficking"]
    }
    tp = sum(1 for n in high_risk   if n["node_id"] in flagged_recipients)
    fp = sum(1 for n in high_risk   if n["node_id"] not in flagged_recipients)
    fn = sum(1 for n in scored_nodes
             if n["node_id"] in flagged_recipients
             and _risk_tier(n["composite_score"]) != "high")

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = (2 * precision * recall) / max(precision + recall, 1e-9)

    lines: list[str] = []
    w = 72

    def hr(char="─"):
        lines.append(char * w)

    def section(title: str):
        lines.append("")
        hr("═")
        lines.append(f"  {title}")
        hr("═")

    hr("█")
    lines.append("  ANTI-CSAM FINANCIAL PATTERN ENGINE — RISK REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    hr("█")

    section("GRAPH SUMMARY")
    lines.append(f"  Nodes (actors)     : {graph_meta['nodes']:,}")
    lines.append(f"  Edges (transactions): {graph_meta['edges']:,}")
    lines.append(f"  Avg in-degree      : {graph_meta['avg_in_degree']}")
    lines.append(f"  Weakly conn. comp. : {graph_meta['weakly_connected_components']:,}")
    lines.append(f"  Graph density      : {graph_meta['density']:.6f}")

    section("SCORING WEIGHTS")
    for dim, w_val in WEIGHTS.items():
        lines.append(f"  {dim:<25} {w_val*100:.0f}%")

    section("RISK TIER SUMMARY")
    lines.append(f"  Total recipient nodes scored : {total:,}")
    lines.append(f"  HIGH   (≥{HIGH_RISK_THRESHOLD:.0%})             : {len(high_risk):,}")
    lines.append(f"  MEDIUM ({MEDIUM_RISK_THRESHOLD:.0%}–{HIGH_RISK_THRESHOLD:.0%})         : {len(medium_risk):,}")
    lines.append(f"  Low    (<{MEDIUM_RISK_THRESHOLD:.0%})             : {total - len(high_risk) - len(medium_risk):,}")

    section("SYNTHETIC GROUND-TRUTH EVALUATION")
    lines.append("  (Available because data is synthetic — not applicable in production)")
    lines.append(f"  True positives  : {tp}")
    lines.append(f"  False positives : {fp}")
    lines.append(f"  False negatives : {fn}")
    lines.append(f"  Precision       : {precision:.2%}")
    lines.append(f"  Recall          : {recall:.2%}")
    lines.append(f"  F1 score        : {f1:.2%}")

    section(f"HIGH-RISK NODES  [{len(high_risk)} nodes]")
    if not high_risk:
        lines.append("  None detected.")
    for node in high_risk[:50]:   # cap report length
        hr()
        lines.append(f"  Node : {node['node_id']}")
        lines.append(f"  Composite score : {node['composite_score']:.4f}  ██ HIGH RISK")
        lines.append(f"  Transactions in : {node['in_degree']}  |  out : {node['out_degree']}")
        lines.append(f"  Total received  : ${node['total_received_usd']:,.2f}")
        lines.append("  Dimension scores:")
        for dim, score in node["dimension_scores"].items():
            bar = "▓" * int(score * 20)
            lines.append(f"    {dim:<25} {score:.4f}  {bar}")
        # Key evidence snippets
        ev = node["evidence"]
        if ev.get("star_topology"):
            e = ev["star_topology"]
            lines.append(
                f"  ★ Star: {e.get('unique_payers')} payers in window, "
                f"mean ${e.get('mean_amount'):.2f}, CV={e.get('amount_cv'):.3f}"
            )
        if ev.get("payment_clustering"):
            e = ev["payment_clustering"]
            lines.append(
                f"  ₿ Structuring: {e.get('structured_txs')}/{e.get('total_incoming_txs')} txs "
                f"({e.get('structuring_ratio'):.0%}), layering={e.get('layering_score'):.2f}"
            )
        if ev.get("geo_mismatch"):
            e = ev["geo_mismatch"]
            lines.append(
                f"  ✈ Geo: {e.get('foreign_payers')} foreign payers, "
                f"{e.get('round_trips_24h')} round-trips <24h"
            )
        if ev.get("platform_hopping"):
            e = ev["platform_hopping"]
            lines.append(
                f"  ↔ Hops: max {e.get('max_platforms_per_payer')} platforms/payer, "
                f"platforms={e.get('platforms')}"
            )

    if len(high_risk) > 50:
        lines.append(f"  … and {len(high_risk) - 50} more (see risk_scores.json)")

    section(f"MEDIUM-RISK NODES  [{len(medium_risk)} nodes — top 20]")
    for node in medium_risk[:20]:
        lines.append(
            f"  {node['node_id']}  score={node['composite_score']:.4f}  "
            f"in={node['in_degree']} out={node['out_degree']}  "
            f"${node['total_received_usd']:,.2f}"
        )

    lines.append("")
    hr("█")
    lines.append("  END OF REPORT")
    hr("█")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Financial Pattern Engine — loading data …")
    transactions = load_transactions(INPUT_FILE)
    print(f"  Loaded {len(transactions):,} transactions")

    print("Building transaction graph …")
    G = build_graph(transactions)
    meta = graph_summary(G)
    print(f"  {meta['nodes']:,} nodes  |  {meta['edges']:,} edges")

    # Score every node that receives money (has in-edges)
    recipients = [n for n in G.nodes() if G.in_degree(n) > 0]
    print(f"Scoring {len(recipients):,} recipient nodes …")

    scored: list[dict] = []
    for i, rec in enumerate(recipients):
        scored.append(score_recipient(rec, G, transactions))
        if (i + 1) % 100 == 0:
            print(f"  … {i+1}/{len(recipients)}")

    scored.sort(key=lambda n: n["composite_score"], reverse=True)

    # Write outputs
    DATA_DIR.mkdir(exist_ok=True)

    scores_path = DATA_DIR / "risk_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2)
    print(f"  Saved {len(scored):,} node scores → {scores_path}")

    report_path = DATA_DIR / "report.txt"
    report = build_report(scored, meta, transactions)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved report → {report_path}")

    graph_path = DATA_DIR / "graph.gexf"
    nx.write_gexf(G, str(graph_path))
    print(f"  Saved graph  → {graph_path}")

    # Print top 10 to console
    print("\n── Top 10 highest-risk recipients ──")
    print(f"  {'Node':<20} {'Score':>7}  {'Star':>6}  {'Clust':>6}  {'Geo':>6}  {'Hop':>6}  {'In$':>10}")
    print("  " + "─" * 68)
    for node in scored[:10]:
        d = node["dimension_scores"]
        print(
            f"  {node['node_id']:<20} {node['composite_score']:>7.4f}"
            f"  {d['star_topology']:>6.3f}"
            f"  {d['payment_clustering']:>6.3f}"
            f"  {d['geo_mismatch']:>6.3f}"
            f"  {d['platform_hopping']:>6.3f}"
            f"  ${node['total_received_usd']:>9,.2f}"
        )

    # Print summary stats
    high   = sum(1 for n in scored if n["composite_score"] >= HIGH_RISK_THRESHOLD)
    medium = sum(1 for n in scored if MEDIUM_RISK_THRESHOLD <= n["composite_score"] < HIGH_RISK_THRESHOLD)
    low    = len(scored) - high - medium
    print(f"\n── Risk tier summary ──")
    print(f"  HIGH   (≥{HIGH_RISK_THRESHOLD:.0%}): {high:>4}")
    print(f"  MEDIUM ({MEDIUM_RISK_THRESHOLD:.0%}–{HIGH_RISK_THRESHOLD:.0%}): {medium:>4}")
    print(f"  Low    (<{MEDIUM_RISK_THRESHOLD:.0%}): {low:>4}")
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
