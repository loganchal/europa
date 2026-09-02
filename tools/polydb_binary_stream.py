#!/usr/bin/env python3
"""Write exact dimension-9 facet records from polyDB to stdout in binary form."""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time

from pymongo import MongoClient

DIM = 9
MAGIC = b"EWALD9\0\0"
URI = os.environ.get(
    "POLYDB_URI",
    "mongodb://polymake:database@db.polymake.org/?authSource=admin&tls=true",
)
COLLECTION = "Polytopes.Lattice.SmoothReflexive"
NEG_E = {tuple(-1 if i == j else 0 for j in range(DIM)) for i in range(DIM)}


def parse_ranges(spec: str) -> list[tuple[int, int, int]]:
    result = []
    for token in spec.split(","):
        if not token.strip():
            continue
        nf, lo, hi = map(int, token.split(":"))
        if not 10 <= nf <= 26 or lo < 0 or hi <= lo:
            raise ValueError(f"bad range {token}")
        result.append((nf, lo, hi))
    if not result:
        raise ValueError("empty ranges")
    return result


def poly_id(n_facets: int, serial: int) -> str:
    return f"F.9D.f{n_facets}.{serial:07d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--ranges", required=True)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--buffer-bytes", type=int, default=1 << 20)
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args()

    ranges = parse_ranges(args.ranges)
    expected = sum(hi - lo for _, lo, hi in ranges)
    out = sys.stdout.buffer
    out.write(MAGIC)
    out.write(struct.pack("<Q", expected))

    client = MongoClient(
        URI,
        serverSelectionTimeoutMS=60_000,
        connectTimeoutMS=60_000,
        socketTimeoutMS=1_800_000,
        directConnection=True,
    )
    coll = client["polydb"][COLLECTION]
    buffer = bytearray()
    sent = 0
    started = time.time()

    for n_facets, lo, hi in ranges:
        lower = poly_id(n_facets, lo)
        upper = poly_id(n_facets, hi)
        cursor = coll.find(
            {"_id": {"$gte": lower, "$lt": upper}, "DIM": DIM},
            projection={"_id": 1, "FACETS": 1, "N_FACETS": 1},
            sort=[("_id", 1)],
            batch_size=args.batch_size,
        )
        expected_serial = lo
        prefix = f"F.9D.f{n_facets}."
        for doc in cursor:
            identifier = doc["_id"]
            if not identifier.startswith(prefix):
                raise AssertionError(f"unexpected ID {identifier}")
            serial = int(identifier[len(prefix):])
            if serial != expected_serial:
                raise AssertionError(
                    f"noncontiguous range: expected {expected_serial}, got {serial}"
                )
            expected_serial += 1
            facets = doc["FACETS"]
            if int(doc["N_FACETS"]) != n_facets or len(facets) != n_facets:
                raise AssertionError(f"facet count mismatch at {identifier}")
            normals = []
            for row in facets:
                if len(row) != DIM + 1 or int(row[0]) != 1:
                    raise AssertionError(f"invalid facet at {identifier}: {row}")
                normal = tuple(int(a) for a in row[1:])
                if any(a < -32767 or a > 32767 for a in normal):
                    raise AssertionError(f"normal outside int16 at {identifier}")
                normals.append(normal)
            if not NEG_E.issubset(set(normals)):
                raise AssertionError(f"coordinate facets absent at {identifier}")

            buffer += struct.pack("<BI", n_facets, serial)
            for normal in normals:
                buffer += struct.pack("<9h", *normal)
            sent += 1
            if len(buffer) >= args.buffer_bytes:
                out.write(buffer)
                buffer.clear()
            if sent % args.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"{args.shard}: streamed={sent}/{expected} "
                    f"rate={sent/elapsed:.1f}/s last={identifier}",
                    file=sys.stderr,
                    flush=True,
                )
        if expected_serial != hi:
            raise AssertionError(
                f"range {n_facets}:{lo}:{hi} ended at {expected_serial}, expected {hi}"
            )

    if buffer:
        out.write(buffer)
    out.flush()
    if sent != expected:
        raise AssertionError(f"sent {sent}, expected {expected}")
    print(
        f"{args.shard}: stream complete records={sent} seconds={time.time()-started:.3f}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
