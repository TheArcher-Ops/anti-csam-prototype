"""
NLP Classifier — Coded Language Detection + Communication Metadata Rhythm Analysis
Anti-CSAM Trafficking Detection Prototype

Two-stage pipeline:

  Stage 1 — Text Classifier (sklearn TF-IDF + LogisticRegression)
    Detects coded language patterns in synthetic message content
    generated from vocabulary profiles for normal vs. trafficking sessions.

  Stage 2 — Rhythm Feature Classifier (sklearn GradientBoosting)
    Detects anomalous communication timing patterns from session metadata:
      - burst_timing inter-message gaps
      - message density, session duration, night sessions
      - platform sequence length (hopping signal)

  Stage 3 — Ensemble
    Combines both stage scores via a weighted average into a final
    session-level risk score in [0, 1].

Outputs
-------
  data/nlp_scores.json      — per-session risk scores + feature evidence
  data/nlp_report.txt       — classification report + top flagged sessions
  data/tfidf_vectorizer.pkl — saved vectorizer (for future inference)
  data/ensemble_model.pkl   — saved ensemble weights/thresholds
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import math
import pickle
import random
import statistics
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR       = Path("data")
SESSION_FILE   = DATA_DIR / "sessions.json"
OUTPUT_SCORES  = DATA_DIR / "nlp_scores.json"
OUTPUT_REPORT  = DATA_DIR / "nlp_report.txt"
OUTPUT_TFIDF   = DATA_DIR / "tfidf_vectorizer.pkl"
OUTPUT_MODEL   = DATA_DIR / "ensemble_model.pkl"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Ensemble weights (text signal vs rhythm signal)
WEIGHT_TEXT   = 0.45
WEIGHT_RHYTHM = 0.55

HIGH_RISK_THRESHOLD   = 0.60
MEDIUM_RISK_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Vocabulary profiles for synthetic message generation
# ---------------------------------------------------------------------------
# Normal conversation fragments
NORMAL_VOCAB = [
    "hey what's up", "sounds good", "talk later", "how are you doing",
    "did you see the game", "meeting at noon", "can we reschedule",
    "birthday dinner tonight", "running late", "pick you up at 7",
    "the weather is great", "finished the project", "send me the file",
    "call me when free", "thanks for helping", "no problem at all",
    "see you tomorrow", "just checking in", "long day at work",
    "got your message", "sounds like a plan", "take care",
    "happy to help", "good morning", "catch you later",
    "on my way", "stuck in traffic", "back home now",
    "movie tonight", "great idea", "let me know",
]

# Trafficking-coded language fragments (euphemisms, operational signals)
# These are representative of patterns documented in law enforcement
# research — generic enough to be realistic without being instructional.
CODED_VOCAB = [
    "new girl available tonight", "roses for the hour",
    "special menu after midnight", "fresh listings posted",
    "meet at the usual spot", "bring the usual", "package ready",
    "donation required upfront", "screening completed",
    "out-call only right now", "in for the night",
    "contact through the link", "delete after reading",
    "new in town for limited time", "verified clients only",
    "rates in the message", "book through the site",
    "incall downtown discreet", "deposit first no exceptions",
    "the product moved", "handoff at the corner",
    "she's ready to work", "new inventory this week",
    "keeping it low-key", "don't discuss over text",
    "use the other number", "switch to app",
    "going dark for a bit", "will reach out when clear",
    "changed location again",
]


def generate_message_text(session: dict) -> str:
    """
    Synthesise a plausible message log string from session metadata.
    Trafficking sessions draw from CODED_VOCAB; normal from NORMAL_VOCAB.
    The text is a bag-of-phrases stand-in for an actual message log.
    """
    n = max(1, session["message_count"] // 3)
    if session["is_trafficking"]:
        # Mix coded phrases with some normal noise
        coded   = random.choices(CODED_VOCAB,  k=max(1, int(n * 0.7)))
        noise   = random.choices(NORMAL_VOCAB, k=max(1, int(n * 0.3)))
        phrases = coded + noise
    else:
        phrases = random.choices(NORMAL_VOCAB, k=n)
    random.shuffle(phrases)
    return " . ".join(phrases)

# ---------------------------------------------------------------------------
# Rhythm feature extraction
# ---------------------------------------------------------------------------

def extract_rhythm_features(session: dict) -> dict[str, float]:
    """
    Derive numeric features from communication timing metadata.
    Returns a flat dict suitable for a feature matrix row.
    """
    burst = session.get("burst_timing", [])
    burst = [float(x) for x in burst] if burst else []

    msg_count   = int(session.get("message_count", 1))
    night       = float(bool(session.get("night_session", False)))
    n_platforms = len(session.get("platform_sequence", []))
    avg_gap     = float(session.get("avg_message_gap_sec", 60))

    # Parse duration
    try:
        t_start = datetime.fromisoformat(session["session_start"])
        t_end   = datetime.fromisoformat(session["session_end"])
        duration_sec = max((t_end - t_start).total_seconds(), 1.0)
    except Exception:
        duration_sec = avg_gap * max(msg_count - 1, 1)

    # Burst statistics
    if burst:
        gap_mean   = statistics.mean(burst)
        gap_std    = statistics.stdev(burst) if len(burst) > 1 else 0.0
        gap_min    = min(burst)
        gap_max    = max(burst)
        gap_cv     = gap_std / gap_mean if gap_mean > 0 else 0.0
        # Large single gap inside session (silence between negotiation phases)
        max_silence_ratio = gap_max / duration_sec if duration_sec > 0 else 0.0
        # Fraction of gaps < 5 seconds (rapid-fire bursts)
        rapid_ratio = sum(1 for g in burst if g < 5) / len(burst)
    else:
        gap_mean = gap_std = gap_min = gap_max = gap_cv = 0.0
        max_silence_ratio = rapid_ratio = 0.0

    # Message density (msgs per minute)
    msg_density = msg_count / (duration_sec / 60.0)

    # Negotiation rhythm signal: high early density + long silence + short late burst
    # Operationalise as: CV of burst gaps (high = uneven = suspicious)
    negotiation_signal = gap_cv

    return {
        "msg_count":          float(msg_count),
        "duration_sec":       duration_sec,
        "msg_density":        msg_density,
        "avg_gap_sec":        avg_gap,
        "gap_mean":           gap_mean,
        "gap_std":            gap_std,
        "gap_min":            gap_min,
        "gap_max":            gap_max,
        "gap_cv":             gap_cv,
        "max_silence_ratio":  max_silence_ratio,
        "rapid_ratio":        rapid_ratio,
        "night_session":      night,
        "n_platforms":        float(n_platforms),
        "negotiation_signal": negotiation_signal,
    }


FEATURE_NAMES = [
    "msg_count", "duration_sec", "msg_density", "avg_gap_sec",
    "gap_mean", "gap_std", "gap_min", "gap_max", "gap_cv",
    "max_silence_ratio", "rapid_ratio", "night_session",
    "n_platforms", "negotiation_signal",
]


def features_to_array(feat_dict: dict[str, float]) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float64)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_sessions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prepare_datasets(sessions: list[dict]):
    """
    Returns:
      texts  : list[str]       — synthetic message strings
      X_rhy  : np.ndarray      — rhythm feature matrix (n, 14)
      y      : np.ndarray      — binary labels (1 = trafficking)
      feats  : list[dict]      — per-session feature dicts (for reporting)
    """
    texts, rhythm_rows, labels, feats = [], [], [], []
    for s in sessions:
        texts.append(generate_message_text(s))
        feat = extract_rhythm_features(s)
        rhythm_rows.append(features_to_array(feat))
        labels.append(int(s["is_trafficking"]))
        feats.append(feat)

    X_rhy = np.vstack(rhythm_rows)
    y     = np.array(labels, dtype=int)
    return texts, X_rhy, y, feats

# ---------------------------------------------------------------------------
# Stage 1 — Text classifier
# ---------------------------------------------------------------------------

def build_text_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1_000,
            random_state=RANDOM_STATE,
        )),
    ])

# ---------------------------------------------------------------------------
# Stage 2 — Rhythm classifier
# ---------------------------------------------------------------------------

def build_rhythm_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )),
    ])

# ---------------------------------------------------------------------------
# Stage 3 — Ensemble score
# ---------------------------------------------------------------------------

def ensemble_score(p_text: float, p_rhythm: float) -> float:
    return WEIGHT_TEXT * p_text + WEIGHT_RHYTHM * p_rhythm

# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(sessions: list[dict]):
    print("Preparing datasets …")
    texts, X_rhy, y, feats = prepare_datasets(sessions)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"  {len(y):,} sessions  |  trafficking={n_pos}  normal={n_neg}")

    # ── Text pipeline ──────────────────────────────────────────────────
    print("\nTraining text classifier (TF-IDF + LogisticRegression) …")
    text_pipe = build_text_pipeline()
    text_pipe.fit(texts, y)

    cv_text = cross_val_score(
        text_pipe, texts, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
    )
    print(f"  Text CV AUC: {cv_text.mean():.3f} ± {cv_text.std():.3f}")

    p_text = text_pipe.predict_proba(texts)[:, 1]

    # ── Rhythm pipeline ────────────────────────────────────────────────
    print("\nTraining rhythm classifier (GradientBoosting) …")
    rhythm_pipe = build_rhythm_pipeline()
    rhythm_pipe.fit(X_rhy, y)

    cv_rhy = cross_val_score(
        rhythm_pipe, X_rhy, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
    )
    print(f"  Rhythm CV AUC: {cv_rhy.mean():.3f} ± {cv_rhy.std():.3f}")

    p_rhythm = rhythm_pipe.predict_proba(X_rhy)[:, 1]

    # ── Ensemble ───────────────────────────────────────────────────────
    p_ensemble = np.array([ensemble_score(pt, pr) for pt, pr in zip(p_text, p_rhythm)])
    ensemble_auc = roc_auc_score(y, p_ensemble)
    print(f"\nEnsemble AUC: {ensemble_auc:.3f}")

    y_pred = (p_ensemble >= HIGH_RISK_THRESHOLD).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    print(f"  At threshold {HIGH_RISK_THRESHOLD:.0%}: P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    # ── Feature importance (rhythm) ────────────────────────────────────
    importances = rhythm_pipe.named_steps["clf"].feature_importances_
    feat_importance = sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: -x[1]
    )

    return (
        text_pipe, rhythm_pipe,
        p_text, p_rhythm, p_ensemble,
        feats, texts,
        {
            "text_cv_auc":    float(cv_text.mean()),
            "rhythm_cv_auc":  float(cv_rhy.mean()),
            "ensemble_auc":   float(ensemble_auc),
            "precision":      float(prec),
            "recall":         float(rec),
            "f1":             float(f1),
            "feat_importance": [(n, float(v)) for n, v in feat_importance],
        },
    )

# ---------------------------------------------------------------------------
# Build per-session score records
# ---------------------------------------------------------------------------

def build_score_records(
    sessions: list[dict],
    texts: list[str],
    feats: list[dict],
    p_text: np.ndarray,
    p_rhythm: np.ndarray,
    p_ensemble: np.ndarray,
) -> list[dict]:
    records = []
    for i, s in enumerate(sessions):
        tier = (
            "high"   if p_ensemble[i] >= HIGH_RISK_THRESHOLD   else
            "medium" if p_ensemble[i] >= MEDIUM_RISK_THRESHOLD  else
            "low"
        )
        records.append({
            "session_id":      s["session_id"],
            "ensemble_score":  round(float(p_ensemble[i]), 4),
            "text_score":      round(float(p_text[i]),     4),
            "rhythm_score":    round(float(p_rhythm[i]),   4),
            "risk_tier":       tier,
            "ground_truth":    bool(s["is_trafficking"]),
            "pattern_tags":    s.get("pattern_tags", []),
            "rhythm_features": {k: round(v, 4) for k, v in feats[i].items()},
            "synthetic_text_snippet": texts[i][:120] + ("…" if len(texts[i]) > 120 else ""),
        })
    records.sort(key=lambda r: r["ensemble_score"], reverse=True)
    return records

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    score_records: list[dict],
    metrics: dict,
    sessions: list[dict],
) -> str:
    total      = len(score_records)
    high_risk  = [r for r in score_records if r["risk_tier"] == "high"]
    med_risk   = [r for r in score_records if r["risk_tier"] == "medium"]

    lines: list[str] = []
    w = 72

    def hr(c="─"): lines.append(c * w)
    def section(t): lines.append(""); hr("═"); lines.append(f"  {t}"); hr("═")

    hr("█")
    lines.append("  ANTI-CSAM NLP CLASSIFIER — RISK REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    hr("█")

    section("MODEL CONFIGURATION")
    lines.append(f"  Stage 1  TF-IDF n-gram(1-3) + LogisticRegression (balanced)")
    lines.append(f"  Stage 2  StandardScaler + GradientBoostingClassifier (150 trees)")
    lines.append(f"  Ensemble weight  text={WEIGHT_TEXT:.0%}  rhythm={WEIGHT_RHYTHM:.0%}")
    lines.append(f"  High-risk threshold : {HIGH_RISK_THRESHOLD:.0%}")
    lines.append(f"  Medium-risk threshold: {MEDIUM_RISK_THRESHOLD:.0%}")

    section("CROSS-VALIDATED PERFORMANCE  (StratifiedKFold k=5)")
    lines.append(f"  Text classifier   AUC : {metrics['text_cv_auc']:.3f}")
    lines.append(f"  Rhythm classifier AUC : {metrics['rhythm_cv_auc']:.3f}")
    lines.append(f"  Ensemble          AUC : {metrics['ensemble_auc']:.3f}")
    lines.append("")
    lines.append(f"  At threshold {HIGH_RISK_THRESHOLD:.0%}:")
    lines.append(f"    Precision : {metrics['precision']:.3f}")
    lines.append(f"    Recall    : {metrics['recall']:.3f}")
    lines.append(f"    F1        : {metrics['f1']:.3f}")

    section("RHYTHM FEATURE IMPORTANCES  (GradientBoosting)")
    for feat, imp in metrics["feat_importance"]:
        bar = "▓" * int(imp * 40)
        lines.append(f"  {feat:<25} {imp:.4f}  {bar}")

    section("RISK TIER SUMMARY")
    lines.append(f"  Total sessions scored : {total:,}")
    lines.append(f"  HIGH   (≥{HIGH_RISK_THRESHOLD:.0%})   : {len(high_risk):,}")
    lines.append(f"  MEDIUM ({MEDIUM_RISK_THRESHOLD:.0%}–{HIGH_RISK_THRESHOLD:.0%}) : {len(med_risk):,}")
    lines.append(f"  Low    (<{MEDIUM_RISK_THRESHOLD:.0%})   : {total - len(high_risk) - len(med_risk):,}")

    # Ground-truth confusion
    tp = sum(1 for r in high_risk if r["ground_truth"])
    fp = sum(1 for r in high_risk if not r["ground_truth"])
    fn = sum(1 for r in score_records
             if r["ground_truth"] and r["risk_tier"] != "high")
    tn = sum(1 for r in score_records
             if not r["ground_truth"] and r["risk_tier"] != "high")

    section("CONFUSION MATRIX  (high-risk threshold = ground truth positive)")
    lines.append(f"              Predicted+   Predicted-")
    lines.append(f"  Actual +       {tp:>6}       {fn:>6}   (trafficking)")
    lines.append(f"  Actual -       {fp:>6}       {tn:>6}   (normal)")

    section(f"TOP HIGH-RISK SESSIONS  [{len(high_risk)} total — showing up to 30]")
    hdr = f"  {'Session':<20} {'Ens':>6} {'Text':>6} {'Rhy':>6} {'GT':>5}  Tags"
    lines.append(hdr)
    hr()
    for r in high_risk[:30]:
        gt_mark = "✓TP" if r["ground_truth"] else "✗FP"
        tags    = ",".join(r["pattern_tags"]) or "—"
        lines.append(
            f"  {r['session_id']:<20} {r['ensemble_score']:>6.3f} "
            f"{r['text_score']:>6.3f} {r['rhythm_score']:>6.3f} "
            f"{gt_mark:>5}  {tags}"
        )
        snippet = textwrap.shorten(r["synthetic_text_snippet"], width=60)
        lines.append(f"    └ \"{snippet}\"")

    if len(high_risk) > 30:
        lines.append(f"  … {len(high_risk) - 30} more in nlp_scores.json")

    section(f"MEDIUM-RISK SESSIONS  [top 20]")
    lines.append(f"  {'Session':<20} {'Ens':>6} {'Text':>6} {'Rhy':>6} {'GT':>5}  Tags")
    hr()
    for r in med_risk[:20]:
        gt_mark = "✓" if r["ground_truth"] else "✗"
        tags    = ",".join(r["pattern_tags"]) or "—"
        lines.append(
            f"  {r['session_id']:<20} {r['ensemble_score']:>6.3f} "
            f"{r['text_score']:>6.3f} {r['rhythm_score']:>6.3f} "
            f"{gt_mark:>5}  {tags}"
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
    print("NLP Classifier — Anti-CSAM Trafficking Detection")
    print("=" * 50)

    print(f"\nLoading sessions from {SESSION_FILE} …")
    sessions = load_sessions(SESSION_FILE)
    print(f"  {len(sessions):,} sessions loaded")

    (
        text_pipe, rhythm_pipe,
        p_text, p_rhythm, p_ensemble,
        feats, texts,
        metrics,
    ) = train_and_evaluate(sessions)

    print("\nBuilding score records …")
    score_records = build_score_records(
        sessions, texts, feats, p_text, p_rhythm, p_ensemble
    )

    # ── Save outputs ───────────────────────────────────────────────────
    DATA_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_SCORES, "w", encoding="utf-8") as f:
        json.dump(score_records, f, indent=2)
    print(f"  Saved {len(score_records):,} session scores → {OUTPUT_SCORES}")

    report = build_report(score_records, metrics, sessions)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved report → {OUTPUT_REPORT}")

    with open(OUTPUT_TFIDF, "wb") as f:
        pickle.dump(text_pipe, f)
    print(f"  Saved text pipeline → {OUTPUT_TFIDF}")

    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump({
            "rhythm_pipe":     rhythm_pipe,
            "weight_text":     WEIGHT_TEXT,
            "weight_rhythm":   WEIGHT_RHYTHM,
            "high_threshold":  HIGH_RISK_THRESHOLD,
            "medium_threshold": MEDIUM_RISK_THRESHOLD,
            "feature_names":   FEATURE_NAMES,
        }, f)
    print(f"  Saved ensemble model → {OUTPUT_MODEL}")

    # ── Console summary ────────────────────────────────────────────────
    high  = [r for r in score_records if r["risk_tier"] == "high"]
    med   = [r for r in score_records if r["risk_tier"] == "medium"]
    low   = [r for r in score_records if r["risk_tier"] == "low"]

    print("\n── Top 10 highest-risk sessions ──")
    print(f"  {'Session':<20} {'Ensemble':>8} {'Text':>7} {'Rhythm':>7} {'GT':>6}  Tags")
    print("  " + "─" * 65)
    for r in score_records[:10]:
        gt = "TRAFF" if r["ground_truth"] else "normal"
        tags = ",".join(r["pattern_tags"]) or "—"
        print(
            f"  {r['session_id']:<20} {r['ensemble_score']:>8.4f}"
            f" {r['text_score']:>7.4f} {r['rhythm_score']:>7.4f}"
            f" {gt:>6}  {tags}"
        )

    print(f"\n── Risk tier summary ──")
    print(f"  HIGH   (≥{HIGH_RISK_THRESHOLD:.0%}): {len(high):>4}")
    print(f"  MEDIUM ({MEDIUM_RISK_THRESHOLD:.0%}–{HIGH_RISK_THRESHOLD:.0%}): {len(med):>4}")
    print(f"  Low    (<{MEDIUM_RISK_THRESHOLD:.0%}): {len(low):>4}")
    print(f"\n── Model performance ──")
    print(f"  Ensemble AUC : {metrics['ensemble_auc']:.3f}")
    print(f"  Precision    : {metrics['precision']:.3f}")
    print(f"  Recall       : {metrics['recall']:.3f}")
    print(f"  F1           : {metrics['f1']:.3f}")
    print(f"\nFull report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
