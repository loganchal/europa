#!/usr/bin/env python3
from __future__ import annotations
import glob
import json
import os
from collections import Counter

EXPECTED_SHARDS = 128
EXPECTED_TOTAL = 8_229_721

files = sorted(glob.glob("ewald-d9-results/**/d9-shard-*.json", recursive=True))
if len(files) != EXPECTED_SHARDS:
    raise SystemExit(f"expected {EXPECTED_SHARDS} shard files, found {len(files)}")

rows = [json.load(open(path)) for path in files]
rows.sort(key=lambda r: r["shard"])
if [r["shard"] for r in rows] != list(range(EXPECTED_SHARDS)):
    raise SystemExit("shard indices are not exactly 0..127")
if any(r["nshards"] != EXPECTED_SHARDS for r in rows):
    raise SystemExit("inconsistent nshards")
if any(r.get("limited") for r in rows):
    raise SystemExit("a shard was run in limited mode")

cursor = 0
for row in rows:
    if row["global_start"] != cursor:
        raise SystemExit(f"coverage gap or overlap before shard {row['shard']}")
    if row["global_end"] - row["global_start"] != row["assigned_total"]:
        raise SystemExit(f"bad assigned count in shard {row['shard']}")
    cursor = row["global_end"]
if cursor != EXPECTED_TOTAL:
    raise SystemExit(f"coverage ends at {cursor}, expected {EXPECTED_TOTAL}")

total_assigned = sum(r["assigned_total"] for r in rows)
total_hits = sum(r["processed_query_hits"] for r in rows)
total_raw_trivial = sum(r["raw_trivial_standard_basis_records"] for r in rows)
if total_assigned != EXPECTED_TOTAL:
    raise SystemExit("assigned total mismatch")
if total_hits + total_raw_trivial != EXPECTED_TOTAL:
    raise SystemExit("query partition mismatch")
if not all(r.get("query_self_test") for r in rows):
    raise SystemExit("query self-test missing")

failures = []
anomalies = []
methods = Counter()
min_E = None
min_ids = []
for row in rows:
    failures.extend(row.get("failures", []))
    anomalies.extend(row.get("normal_form_anomalies", []))
    methods.update(row.get("methods", {}))
    m = row.get("min_E_query_hits")
    if m is None:
        continue
    if min_E is None or m < min_E:
        min_E = m
        min_ids = list(row.get("min_ids", []))
    elif m == min_E:
        min_ids.extend(row.get("min_ids", []))

summary = {
    "dimension": 9,
    "database_records": EXPECTED_TOTAL,
    "shards": EXPECTED_SHARDS,
    "coverage_start": 0,
    "coverage_end": cursor,
    "raw_trivial_standard_basis_records": total_raw_trivial,
    "records_examined_from_query_superset": total_hits,
    "records_with_exact_verified_basis_in_query_superset": total_hits - len(failures) - len(anomalies),
    "basis_methods": dict(methods),
    "minimum_symmetric_point_count_among_query_hits": min_E,
    "minimum_ids": sorted(set(min_ids)),
    "heuristic_failures": failures,
    "normal_coordinate_anomalies": anomalies,
    "n_heuristic_failures": len(failures),
    "n_normal_coordinate_anomalies": len(anomalies),
    "total_elapsed_worker_seconds": sum(r.get("elapsed_sec", 0.0) for r in rows),
    "certificate": {
        "shard_indices_exact": True,
        "contiguous_global_coverage": True,
        "assigned_total_exact": True,
        "query_partition_exact": True,
        "query_self_test_all_shards": True,
    },
}

os.makedirs("ewald-d9-summary", exist_ok=True)
with open("ewald-d9-summary/summary.json", "w") as f:
    json.dump(summary, f, separators=(",", ":"))
with open("ewald-d9-summary/shards.json", "w") as f:
    json.dump(rows, f, separators=(",", ":"))

compact = {k:v for k,v in summary.items() if k not in ("heuristic_failures", "normal_coordinate_anomalies")}
print("CENSUS_SUMMARY", json.dumps(compact, separators=(",", ":")), flush=True)
if failures:
    print("CANDIDATE_FAILURE_IDS", [x["id"] for x in failures], flush=True)
if anomalies:
    print("ANOMALY_IDS", [x["id"] for x in anomalies], flush=True)
