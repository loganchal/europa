#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from bson import BSON
from pymongo import MongoClient

URI = os.environ.get(
    "POLYDB_URI",
    "mongodb://polymake:database@db.polymake.org/?authSource=admin&tls=true",
)
COLLECTION = "Polytopes.Lattice.SmoothReflexive"


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "dim9-stats.json")
    started = time.time()
    client = MongoClient(
        URI,
        serverSelectionTimeoutMS=60_000,
        connectTimeoutMS=60_000,
        socketTimeoutMS=600_000,
        directConnection=True,
    )
    coll = client["polydb"][COLLECTION]
    indexes = {name: dict(spec) for name, spec in coll.index_information().items()}
    facet_values = sorted(coll.distinct("N_FACETS", {"DIM": 9}))
    groups = []
    for nf in facet_values:
        t0 = time.time()
        query = {"DIM": 9, "N_FACETS": nf}
        count = coll.count_documents(query)
        first = coll.find_one(query, projection={"_id": 1}, sort=[("_id", 1)])
        last = coll.find_one(query, projection={"_id": 1}, sort=[("_id", -1)])
        groups.append({
            "n_facets": nf,
            "count": count,
            "first_id": first["_id"] if first else None,
            "last_id": last["_id"] if last else None,
            "query_seconds": time.time() - t0,
        })

    sample_n = 10000
    query = {"DIM": 9}
    cursor = coll.find(
        query,
        projection={"_id": 1, "FACETS": 1, "N_FACETS": 1},
        sort=[("_id", 1)],
        batch_size=2000,
        limit=sample_n,
        no_cursor_timeout=False,
    )
    bytes_bson = 0
    rows = 0
    t0 = time.time()
    normal_abs_max = 0
    leading_constants = set()
    initial_standard_count = 0
    malformed = []
    for doc in cursor:
        rows += 1
        bytes_bson += len(BSON.encode(doc))
        facets = doc["FACETS"]
        normals = [tuple(int(a) for a in row[1:]) for row in facets]
        leading_constants.update(int(row[0]) for row in facets)
        normal_abs_max = max(normal_abs_max, *(abs(a) for u in normals for a in u))
        expected = {tuple(-1 if i == j else 0 for j in range(9)) for i in range(9)}
        if expected.issubset(set(normals)):
            initial_standard_count += 1
        elif len(malformed) < 20:
            malformed.append(doc["_id"])
    stream_seconds = time.time() - t0

    result = {
        "dimension": 9,
        "indexes": indexes,
        "n_facets_groups": groups,
        "group_count_sum": sum(g["count"] for g in groups),
        "sample": {
            "rows": rows,
            "bson_bytes": bytes_bson,
            "mean_bson_bytes": bytes_bson / rows if rows else None,
            "stream_seconds": stream_seconds,
            "rows_per_second": rows / stream_seconds if stream_seconds else None,
            "leading_constants": sorted(leading_constants),
            "maximum_abs_normal_coordinate": normal_abs_max,
            "contains_negative_standard_basis_count": initial_standard_count,
            "malformed_ids": malformed,
        },
        "elapsed_seconds": time.time() - started,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
