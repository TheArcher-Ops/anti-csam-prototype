"""
Synthetic Data Generator for Anti-CSAM Trafficking Detection Prototype.

Produces:
  - Transaction records  (payer_id, recipient_id, amount, timestamp, platform, geo)
  - Communication metadata (session_id, message_count, burst_timing, platform_sequence)

15% of records are injected with realistic trafficking patterns.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import random
import uuid
import json
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_TRANSACTIONS = 2000
NUM_SESSIONS = 1500
TRAFFICKING_RATE = 0.15          # 15% flagged records
OUTPUT_DIR = Path("data")

PLATFORMS = ["CashApp", "Venmo", "Zelle", "PayPal", "Crypto", "MoneyGram", "Western Union"]
COMMS_PLATFORMS = ["Telegram", "Signal", "WhatsApp", "Discord", "Kik", "Wickr", "SMS"]

# Geo pools — (country_code, timezone_offset_hours)
GEO_DOMESTIC = [
    ("US-CA", -8), ("US-NY", -5), ("US-TX", -6), ("US-FL", -5),
    ("US-IL", -6), ("US-WA", -8), ("US-GA", -5), ("US-OH", -5),
]
GEO_FOREIGN = [
    ("MX", -6), ("PH", +8), ("TH", +7), ("RO", +2), ("NG", +1),
    ("CO", -5), ("VN", +7), ("UA", +2), ("BD", +6), ("KH", +7),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def jitter(value: float, pct: float = 0.1) -> float:
    """Add ±pct% noise to a value."""
    return value * (1 + random.uniform(-pct, pct))

def format_ts(dt: datetime) -> str:
    return dt.isoformat()

# ---------------------------------------------------------------------------
# Normal transaction builder
# ---------------------------------------------------------------------------

def build_normal_transaction(
    start: datetime, end: datetime, known_users: list[str]
) -> dict:
    payer = random.choice(known_users)
    recipient = random.choice([u for u in known_users if u != payer])
    return {
        "tx_id": random_id("TX"),
        "payer_id": payer,
        "recipient_id": recipient,
        "amount_usd": round(random.uniform(5, 500), 2),
        "timestamp": format_ts(random_timestamp(start, end)),
        "platform": random.choice(PLATFORMS[:4]),   # domestic platforms
        "payer_geo": random.choice(GEO_DOMESTIC)[0],
        "recipient_geo": random.choice(GEO_DOMESTIC)[0],
        "is_trafficking": False,
        "pattern_tags": [],
    }

# ---------------------------------------------------------------------------
# Trafficking pattern builders
# ---------------------------------------------------------------------------

def inject_star_topology(
    start: datetime, end: datetime, known_users: list[str]
) -> list[dict]:
    """
    One hub recipient receives micro-payments from many unique payers
    in a short time window (< 2 hours).
    """
    hub = random_id("HUB")
    window_start = random_timestamp(start, end - timedelta(hours=2))
    window_end = window_start + timedelta(hours=random.uniform(0.5, 2))
    n_payers = random.randint(6, 15)
    records = []
    for _ in range(n_payers):
        payer = random_id("ANON")
        records.append({
            "tx_id": random_id("TX"),
            "payer_id": payer,
            "recipient_id": hub,
            "amount_usd": round(jitter(49.99, 0.15), 2),   # near-identical amounts
            "timestamp": format_ts(random_timestamp(window_start, window_end)),
            "platform": random.choice(PLATFORMS),
            "payer_geo": random.choice(GEO_DOMESTIC)[0],
            "recipient_geo": random.choice(GEO_DOMESTIC)[0],
            "is_trafficking": True,
            "pattern_tags": ["star_topology"],
        })
    return records


def inject_payment_clustering(
    start: datetime, end: datetime, known_users: list[str]
) -> list[dict]:
    """
    Payments just below round-number reporting thresholds (structuring).
    Multiple hops through intermediary accounts.
    """
    actors = [random_id("ACT") for _ in range(3)]
    records = []
    threshold = random.choice([200, 500, 1000, 3000, 10000])
    hop_ts = random_timestamp(start, end - timedelta(hours=6))
    for i in range(len(actors) - 1):
        hop_ts += timedelta(minutes=random.randint(5, 90))
        records.append({
            "tx_id": random_id("TX"),
            "payer_id": actors[i],
            "recipient_id": actors[i + 1],
            "amount_usd": round(threshold - random.uniform(0.01, threshold * 0.05), 2),
            "timestamp": format_ts(hop_ts),
            "platform": random.choice(PLATFORMS),
            "payer_geo": random.choice(GEO_DOMESTIC)[0],
            "recipient_geo": random.choice(GEO_DOMESTIC)[0],
            "is_trafficking": True,
            "pattern_tags": ["payment_clustering", "structuring"],
        })
    return records


def inject_geo_mismatch(
    start: datetime, end: datetime, known_users: list[str]
) -> list[dict]:
    """
    Domestic payer → foreign recipient, rapidly followed by a foreign
    payer → domestic recipient cashout (round-trip mismatch).
    """
    domestic_user = random.choice(known_users)
    foreign_actor = random_id("FORN")
    cashout_actor = random_id("CSH")
    t0 = random_timestamp(start, end - timedelta(hours=4))
    amount = round(random.uniform(100, 2000), 2)
    records = [
        {
            "tx_id": random_id("TX"),
            "payer_id": domestic_user,
            "recipient_id": foreign_actor,
            "amount_usd": amount,
            "timestamp": format_ts(t0),
            "platform": random.choice(PLATFORMS),
            "payer_geo": random.choice(GEO_DOMESTIC)[0],
            "recipient_geo": random.choice(GEO_FOREIGN)[0],
            "is_trafficking": True,
            "pattern_tags": ["geo_mismatch"],
        },
        {
            "tx_id": random_id("TX"),
            "payer_id": foreign_actor,
            "recipient_id": cashout_actor,
            "amount_usd": round(amount * random.uniform(0.85, 0.98), 2),
            "timestamp": format_ts(t0 + timedelta(minutes=random.randint(15, 180))),
            "platform": random.choice(PLATFORMS[4:]),  # crypto / wire
            "payer_geo": random.choice(GEO_FOREIGN)[0],
            "recipient_geo": random.choice(GEO_DOMESTIC)[0],
            "is_trafficking": True,
            "pattern_tags": ["geo_mismatch", "cashout"],
        },
    ]
    return records


def inject_platform_hopping(
    start: datetime, end: datetime, known_users: list[str]
) -> list[dict]:
    """
    Same economic relationship transacted across ≥3 different platforms
    within 24 hours to fragment the money trail.
    """
    payer = random.choice(known_users)
    recipient = random_id("RECIP")
    t0 = random_timestamp(start, end - timedelta(hours=24))
    platforms_used = random.sample(PLATFORMS, k=random.randint(3, len(PLATFORMS)))
    records = []
    for i, plat in enumerate(platforms_used):
        records.append({
            "tx_id": random_id("TX"),
            "payer_id": payer,
            "recipient_id": recipient,
            "amount_usd": round(jitter(75.0, 0.2), 2),
            "timestamp": format_ts(t0 + timedelta(hours=i * random.uniform(1, 5))),
            "platform": plat,
            "payer_geo": random.choice(GEO_DOMESTIC)[0],
            "recipient_geo": random.choice(GEO_DOMESTIC)[0],
            "is_trafficking": True,
            "pattern_tags": ["platform_hopping"],
        })
    return records

# ---------------------------------------------------------------------------
# Normal session builder
# ---------------------------------------------------------------------------

def build_normal_session(start: datetime, end: datetime) -> dict:
    session_start = random_timestamp(start, end - timedelta(hours=1))
    duration_min = random.uniform(2, 60)
    msg_count = random.randint(3, 40)
    return {
        "session_id": random_id("SES"),
        "user_a": random_id("USR"),
        "user_b": random_id("USR"),
        "session_start": format_ts(session_start),
        "session_end": format_ts(session_start + timedelta(minutes=duration_min)),
        "message_count": msg_count,
        # burst_timing: list of inter-message gaps (seconds)
        "burst_timing": _normal_burst(msg_count, duration_min),
        "platform_sequence": [random.choice(COMMS_PLATFORMS)],
        "avg_message_gap_sec": round(duration_min * 60 / max(msg_count - 1, 1), 1),
        "night_session": _is_night(session_start),
        "is_trafficking": False,
        "pattern_tags": [],
    }


def _normal_burst(msg_count: int, duration_min: float) -> list[float]:
    """Roughly uniform gaps with some noise."""
    if msg_count <= 1:
        return []
    avg_gap = (duration_min * 60) / (msg_count - 1)
    return [round(max(1, jitter(avg_gap, 0.4)), 1) for _ in range(msg_count - 1)]


def _is_night(dt: datetime) -> bool:
    return dt.hour < 6 or dt.hour >= 22

# ---------------------------------------------------------------------------
# Trafficking session pattern builders
# ---------------------------------------------------------------------------

def inject_coded_language_session(start: datetime, end: datetime) -> dict:
    """
    Short, high-frequency bursts suggesting negotiation, followed by
    a long silence, then a brief confirmation burst.
    """
    t0 = random_timestamp(start, end - timedelta(hours=3))
    negotiation_msgs = random.randint(8, 20)
    silence_min = random.uniform(10, 60)
    confirm_msgs = random.randint(2, 5)

    neg_gaps = [round(random.uniform(1, 8), 1) for _ in range(negotiation_msgs - 1)]
    conf_gaps = [round(random.uniform(0.5, 3), 1) for _ in range(confirm_msgs - 1)]
    all_gaps = neg_gaps + [silence_min * 60] + conf_gaps

    total_msgs = negotiation_msgs + confirm_msgs
    duration_min = sum(all_gaps) / 60

    return {
        "session_id": random_id("SES"),
        "user_a": random_id("USR"),
        "user_b": random_id("USR"),
        "session_start": format_ts(t0),
        "session_end": format_ts(t0 + timedelta(minutes=duration_min)),
        "message_count": total_msgs,
        "burst_timing": all_gaps,
        "platform_sequence": [random.choice(COMMS_PLATFORMS)],
        "avg_message_gap_sec": round(sum(all_gaps) / max(len(all_gaps), 1), 1),
        "night_session": _is_night(t0),
        "is_trafficking": True,
        "pattern_tags": ["coded_language_rhythm", "negotiation_burst"],
    }


def inject_platform_hop_session(start: datetime, end: datetime) -> dict:
    """
    Communication migrates across ≥3 platforms within one logical session,
    suggesting evasion of monitoring on any single platform.
    """
    t0 = random_timestamp(start, end - timedelta(hours=2))
    n_platforms = random.randint(3, len(COMMS_PLATFORMS))
    platform_seq = random.sample(COMMS_PLATFORMS, k=n_platforms)
    msg_count = random.randint(n_platforms * 2, n_platforms * 8)
    duration_min = random.uniform(20, 120)

    return {
        "session_id": random_id("SES"),
        "user_a": random_id("USR"),
        "user_b": random_id("USR"),
        "session_start": format_ts(t0),
        "session_end": format_ts(t0 + timedelta(minutes=duration_min)),
        "message_count": msg_count,
        "burst_timing": _normal_burst(msg_count, duration_min),
        "platform_sequence": platform_seq,
        "avg_message_gap_sec": round(duration_min * 60 / max(msg_count - 1, 1), 1),
        "night_session": _is_night(t0),
        "is_trafficking": True,
        "pattern_tags": ["platform_hopping_comms"],
    }


def inject_late_night_burst_session(start: datetime, end: datetime) -> dict:
    """
    Very late-night session (01:00–04:00 local) with unusually high
    message density in short windows — matches operational coordination.
    """
    base_date = random_timestamp(start, end).replace(hour=0, minute=0, second=0)
    t0 = base_date + timedelta(hours=random.uniform(1, 4))
    msg_count = random.randint(30, 80)
    duration_min = random.uniform(5, 20)   # many messages, short window

    return {
        "session_id": random_id("SES"),
        "user_a": random_id("USR"),
        "user_b": random_id("USR"),
        "session_start": format_ts(t0),
        "session_end": format_ts(t0 + timedelta(minutes=duration_min)),
        "message_count": msg_count,
        "burst_timing": [round(random.uniform(0.5, 5), 1) for _ in range(msg_count - 1)],
        "platform_sequence": [random.choice(COMMS_PLATFORMS)],
        "avg_message_gap_sec": round(duration_min * 60 / max(msg_count - 1, 1), 1),
        "night_session": True,
        "is_trafficking": True,
        "pattern_tags": ["late_night_burst"],
    }

# ---------------------------------------------------------------------------
# Main generators
# ---------------------------------------------------------------------------

TRAFFICKING_TX_PATTERNS = [
    inject_star_topology,
    inject_payment_clustering,
    inject_geo_mismatch,
    inject_platform_hopping,
]

TRAFFICKING_SESSION_PATTERNS = [
    inject_coded_language_session,
    inject_platform_hop_session,
    inject_late_night_burst_session,
]


def generate_transactions(n: int, start: datetime, end: datetime) -> list[dict]:
    known_users = [random_id("USR") for _ in range(200)]
    records: list[dict] = []

    trafficking_target = math.ceil(n * TRAFFICKING_RATE)
    normal_target = n - trafficking_target

    # Normal records
    while len(records) < normal_target:
        records.append(build_normal_transaction(start, end, known_users))

    # Trafficking records — inject pattern groups until we hit quota
    while sum(1 for r in records if r["is_trafficking"]) < trafficking_target:
        pattern_fn = random.choice(TRAFFICKING_TX_PATTERNS)
        batch = pattern_fn(start, end, known_users)
        records.extend(batch)

    # Trim to exactly n, preserving label ratio approximately
    random.shuffle(records)
    return records[:n]


def generate_sessions(n: int, start: datetime, end: datetime) -> list[dict]:
    records: list[dict] = []

    trafficking_target = math.ceil(n * TRAFFICKING_RATE)
    normal_target = n - trafficking_target

    while len(records) < normal_target:
        records.append(build_normal_session(start, end))

    while sum(1 for r in records if r["is_trafficking"]) < trafficking_target:
        pattern_fn = random.choice(TRAFFICKING_SESSION_PATTERNS)
        records.append(pattern_fn(start, end))

    random.shuffle(records)
    return records[:n]

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_json(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved {len(records):,} records → {path}")


def save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Flatten list fields to JSON strings for CSV compatibility
    flat = []
    for r in records:
        row = dict(r)
        for k, v in row.items():
            if isinstance(v, list):
                row[k] = json.dumps(v)
        flat.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)
    print(f"  Saved {len(records):,} records → {path}")


def print_stats(label: str, records: list[dict]) -> None:
    total = len(records)
    flagged = sum(1 for r in records if r["is_trafficking"])
    tags: dict[str, int] = {}
    for r in records:
        for t in r.get("pattern_tags", []):
            tags[t] = tags.get(t, 0) + 1
    print(f"\n{label} ({total:,} total)")
    print(f"  Normal:     {total - flagged:,} ({(total - flagged)/total*100:.1f}%)")
    print(f"  Trafficking:{flagged:,} ({flagged/total*100:.1f}%)")
    if tags:
        print("  Pattern tag breakdown:")
        for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
            print(f"    {tag:<30} {count:>5}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    start = datetime(2024, 1, 1)
    end   = datetime(2024, 12, 31)

    print("Generating synthetic anti-CSAM detection dataset …")
    print(f"  Period : {start.date()} → {end.date()}")
    print(f"  Trafficking injection rate: {TRAFFICKING_RATE*100:.0f}%\n")

    txs = generate_transactions(NUM_TRANSACTIONS, start, end)
    ses = generate_sessions(NUM_SESSIONS, start, end)

    print_stats("Transaction Records", txs)
    print_stats("Communication Sessions", ses)

    print("\nWriting output files …")
    save_json(txs, OUTPUT_DIR / "transactions.json")
    save_csv(txs,  OUTPUT_DIR / "transactions.csv")
    save_json(ses, OUTPUT_DIR / "sessions.json")
    save_csv(ses,  OUTPUT_DIR / "sessions.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
