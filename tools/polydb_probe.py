#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import ssl
import sys
import time
from pathlib import Path

from bson import json_util
from pymongo import MongoClient

URI = os.environ.get(
    "POLYDB_URI",
    "mongodb://polymake:database@db.polymake.org/?authSource=admin&tls=true",
)
COLLECTION = "Polytopes.Lattice.SmoothReflexive"


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "polydb-probe.json")
    started = time.time()
    client = MongoClient(
        URI,
        serverSelectionTimeoutMS=60_000,
        connectTimeoutMS=60_000,
        socketTimeoutMS=180_000,
        directConnection=True,
        tlsAllowInvalidCertificates=False,
    )
    ping = client.admin.command("ping")
    coll = client["polydb"][COLLECTION]
    result: dict = {
        "ping": ping,
        "collection": COLLECTION,
        "dimensions": coll.distinct("DIM"),
        "count_dim8": coll.count_documents({"DIM": 8}),
        "count_dim9": coll.count_documents({"DIM": 9}),
    }
    projection = {
        "_id": 1,
        "DIM": 1,
        "CONE_DIM": 1,
        "N_VERTICES": 1,
        "VERTICES": 1,
        "N_FACETS": 1,
        "FACETS": 1,
        "SMOOTH": 1,
        "REFLEXIVE": 1,
        "SIMPLE": 1,
        "SIMPLICIAL": 1,
        "_attrs": 1,
    }
    result["sample_dim8"] = coll.find_one({"DIM": 8}, projection=projection)
    result["sample_dim9"] = coll.find_one({"DIM": 9}, projection=projection)
    result["sample_dim9_keys"] = sorted(result["sample_dim9"].keys()) if result["sample_dim9"] else []
    result["elapsed_seconds"] = time.time() - started
    output.write_text(json_util.dumps(result, indent=2), encoding="utf-8")
    print(json_util.dumps({
        "count_dim8": result["count_dim8"],
        "count_dim9": result["count_dim9"],
        "dimensions": result["dimensions"],
        "sample_dim9_id": result["sample_dim9"].get("_id") if result["sample_dim9"] else None,
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
