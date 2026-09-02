#!/usr/bin/env python3
"""Exact sharded Ewald-set scan of the dimension-9 polyDB classification.

A polyDB FACETS row is [1,a_1,...,a_9] and represents 1+a.x >= 0.
For x and -x to both lie in P, it is necessary and sufficient that
|a.x| <= 1 for every facet.  Every classified representative contains the
nine normals -e_i, so the complete search box is {-1,0,1}^9 (19683 points).
Membership is represented by Python integer bitsets and uses exact dot products.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pymongo import MongoClient

DIM = 9
VALUES = (-1, 0, 1)
X = np.asarray(list(itertools.product(VALUES, repeat=DIM)), dtype=np.int16)
N_X = int(X.shape[0])
ALL_MASK = (1 << N_X) - 1
ZERO = (0,) * DIM
ZERO_INDEX = next(i for i, row in enumerate(X) if tuple(map(int, row)) == ZERO)
NEG_E = {tuple(-1 if i == j else 0 for j in range(DIM)) for i in range(DIM)}
URI = os.environ.get(
    "POLYDB_URI",
    "mongodb://polymake:database@db.polymake.org/?authSource=admin&tls=true",
)
COLLECTION = "Polytopes.Lattice.SmoothReflexive"


def canonical_normal(u: tuple[int, ...]) -> tuple[int, ...]:
    for a in u:
        if a < 0:
            return tuple(-x for x in u)
        if a > 0:
            return u
    raise ValueError("zero facet normal")


class RayMasks:
    def __init__(self) -> None:
        self.cache: dict[tuple[int, ...], int] = {}

    def __getitem__(self, raw: tuple[int, ...]) -> int:
        u = canonical_normal(raw)
        found = self.cache.get(u)
        if found is not None:
            return found
        vector = np.asarray(u, dtype=np.int16)
        valid = np.abs(X @ vector) <= 1
        packed = np.packbits(valid, bitorder="little")
        mask = int.from_bytes(packed.tobytes(), "little") & ALL_MASK
        self.cache[u] = mask
        return mask


def parse_ranges(spec: str) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        nf, lo, hi = map(int, token.split(":"))
        if nf < 10 or nf > 26 or lo < 0 or hi <= lo:
            raise ValueError(f"invalid range {token!r}")
        ranges.append((nf, lo, hi))
    if not ranges:
        raise ValueError("no ranges")
    return ranges


def id_for(n_facets: int, serial: int) -> str:
    return f"F.9D.f{n_facets}.{serial:07d}"


def iter_bits(mask: int) -> Iterable[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def points_from_mask(mask: int) -> list[tuple[int, ...]]:
    return [tuple(map(int, X[i])) for i in iter_bits(mask)]


def parity_code(point_index: int) -> int:
    code = 0
    for a in X[point_index]:
        code = (code << 1) | int(a != 0)
    return code


PARITY = [parity_code(i) for i in range(N_X)]


def gf2_basis_from_mask(mask: int) -> tuple[int, list[int]]:
    pivots = [0] * DIM
    chosen: list[int] = []
    rank = 0
    for point_index in iter_bits(mask):
        v = PARITY[point_index]
        if v == 0:
            continue
        original = v
        for bit in range(DIM - 1, -1, -1):
            if not ((v >> bit) & 1):
                continue
            if pivots[bit]:
                v ^= pivots[bit]
            else:
                pivots[bit] = v
                chosen.append(point_index)
                rank += 1
                break
        if rank == DIM:
            return rank, chosen
    return rank, chosen


def determinant(columns: Sequence[Sequence[int]]) -> int:
    n = len(columns)
    a = [[int(columns[j][i]) for j in range(n)] for i in range(n)]
    sign = 1
    previous = 1
    for k in range(n - 1):
        pivot_row = next((r for r in range(k, n) if a[r][k]), None)
        if pivot_row is None:
            return 0
        if pivot_row != k:
            a[k], a[pivot_row] = a[pivot_row], a[k]
            sign = -sign
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = a[i][j] * pivot - a[i][k] * a[k][j]
                if numerator % previous:
                    raise ArithmeticError("non-exact Bareiss division")
                a[i][j] = numerator // previous
        previous = pivot
        for i in range(k + 1, n):
            a[i][k] = 0
    return sign * a[n - 1][n - 1]


def antipodal_representatives(points: Sequence[tuple[int, ...]]) -> list[tuple[int, ...]]:
    result = []
    for point in points:
        if not any(point):
            continue
        first = next(a for a in point if a)
        if first > 0:
            result.append(point)
    return result


def rank_mod_prime(points: Sequence[tuple[int, ...]], p: int) -> int:
    pivots: list[list[int] | None] = [None] * DIM
    rank = 0
    for point in points:
        row = [a % p for a in point]
        for col in range(DIM):
            if row[col] == 0:
                continue
            if pivots[col] is None:
                inv = pow(row[col], -1, p)
                row = [(a * inv) % p for a in row]
                pivots[col] = row
                rank += 1
                break
            factor = row[col]
            pivot = pivots[col]
            row = [(a - factor * b) % p for a, b in zip(row, pivot)]
        if rank == DIM:
            return rank
    return rank


def basis_probe(points: Sequence[tuple[int, ...]], seed: str, max_trials: int) -> dict:
    reps = antipodal_representatives(points)
    ranks = {str(p): rank_mod_prime(reps, p) for p in (2, 3, 5, 7)}
    obstruction = next((p for p, rank in ranks.items() if rank < DIM), None)
    result = {
        "antipodal_nonzero_count": len(reps),
        "ranks_mod_primes": ranks,
        "prime_divisibility_obstruction": int(obstruction) if obstruction else None,
        "trials": 0,
        "has_unimodular_basis": False,
        "basis": None,
        "sampled_determinant_gcd": 0,
        "minimum_nonzero_abs_det_seen": None,
    }
    if obstruction or len(reps) < DIM:
        return result

    rng = random.Random(seed)
    best: int | None = None
    gcd_seen = 0
    for trial in range(1, max_trials + 1):
        cols = rng.sample(reps, DIM)
        det = determinant(cols)
        result["trials"] = trial
        if det:
            ad = abs(det)
            gcd_seen = math.gcd(gcd_seen, ad)
            if best is None or ad < best:
                best = ad
            if ad == 1:
                result["has_unimodular_basis"] = True
                result["basis"] = [list(v) for v in cols]
                break
    result["sampled_determinant_gcd"] = gcd_seen
    result["minimum_nonzero_abs_det_seen"] = best
    return result


def scan(args: argparse.Namespace) -> None:
    ranges = parse_ranges(args.ranges)
    expected_records = sum(hi - lo for _, lo, hi in ranges)
    started = time.time()
    client = MongoClient(
        URI,
        serverSelectionTimeoutMS=60_000,
        connectTimeoutMS=60_000,
        socketTimeoutMS=1_800_000,
        directConnection=True,
    )
    coll = client["polydb"][COLLECTION]
    masks = RayMasks()
    histogram: Counter[int] = Counter()
    facet_histogram: Counter[int] = Counter()
    top: list[tuple[int, str, dict]] = []
    gf2_deficient: list[dict] = []
    scanned = 0
    last_id = None

    for n_facets, lo, hi in ranges:
        lower = id_for(n_facets, lo)
        upper = id_for(n_facets, hi)
        query = {"_id": {"$gte": lower, "$lt": upper}, "DIM": DIM}
        cursor = coll.find(
            query,
            projection={"_id": 1, "FACETS": 1, "N_FACETS": 1},
            sort=[("_id", 1)],
            batch_size=args.batch_size,
            no_cursor_timeout=False,
        )
        range_count = 0
        for doc in cursor:
            poly_id = doc["_id"]
            facets = doc["FACETS"]
            if int(doc["N_FACETS"]) != n_facets or len(facets) != n_facets:
                raise AssertionError(f"facet-count mismatch at {poly_id}")
            normals: list[tuple[int, ...]] = []
            for row in facets:
                if len(row) != DIM + 1 or int(row[0]) != 1:
                    raise AssertionError(f"non-reflexive facet row at {poly_id}: {row}")
                normals.append(tuple(int(a) for a in row[1:]))
            if not NEG_E.issubset(set(normals)):
                raise AssertionError(f"normalized coordinate facets absent at {poly_id}")

            ewald = ALL_MASK
            for normal in normals:
                ewald &= masks[normal]
            n_points = ewald.bit_count()
            if not ((ewald >> ZERO_INDEX) & 1) or n_points % 2 != 1:
                raise AssertionError(f"invalid symmetric point mask at {poly_id}")

            gf2_rank, gf2_chosen = gf2_basis_from_mask(ewald)
            if gf2_rank < DIM:
                gf2_deficient.append({
                    "id": poly_id,
                    "n_facets": n_facets,
                    "n_ewald_points": n_points,
                    "gf2_rank": gf2_rank,
                    "facets": [[1, *u] for u in normals],
                    "ewald_mask_hex": hex(ewald),
                })

            record = {
                "id": poly_id,
                "n_facets": n_facets,
                "n_ewald_points": n_points,
                "gf2_rank": gf2_rank,
                "gf2_basis_points": [list(map(int, X[i])) for i in gf2_chosen],
                "facets": [[1, *u] for u in normals],
                "ewald_mask_hex": hex(ewald),
            }
            heapq.heappush(top, (-n_points, poly_id, record))
            if len(top) > args.top_k:
                heapq.heappop(top)

            histogram[n_points] += 1
            facet_histogram[n_facets] += 1
            scanned += 1
            range_count += 1
            last_id = poly_id
            if scanned % args.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"{args.shard}: scanned={scanned}/{expected_records} "
                    f"rate={scanned/elapsed:.1f}/s min={min(histogram)} "
                    f"cached_normals={len(masks.cache)} last={poly_id}",
                    flush=True,
                )
        if range_count != hi - lo:
            raise AssertionError(
                f"range {n_facets}:{lo}:{hi} returned {range_count}, expected {hi-lo}"
            )

    retained = [item[2] for item in top]
    retained.sort(key=lambda r: (r["n_ewald_points"], r["id"]))
    for record in retained:
        point_mask = int(record["ewald_mask_hex"], 16)
        points = points_from_mask(point_mask)
        record["ewald_points"] = [list(v) for v in points]
        record["basis_probe"] = basis_probe(points, record["id"], args.basis_trials)

    for record in gf2_deficient:
        point_mask = int(record["ewald_mask_hex"], 16)
        record["ewald_points"] = [list(v) for v in points_from_mask(point_mask)]

    result = {
        "dimension": DIM,
        "shard": args.shard,
        "ranges": ranges,
        "complete": scanned == expected_records,
        "expected_records": expected_records,
        "polytopes_scanned": scanned,
        "last_id": last_id,
        "candidate_box": [[-1, 1] for _ in range(DIM)],
        "candidate_box_size": N_X,
        "distinct_normal_constraints": len(masks.cache),
        "minimum_ewald_points": min(histogram) if histogram else None,
        "ewald_point_count_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "facet_count_histogram": {str(k): facet_histogram[k] for k in sorted(facet_histogram)},
        "gf2_deficient_count": len(gf2_deficient),
        "gf2_deficient": gf2_deficient,
        "smallest_ewald_sets": retained,
        "elapsed_seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{args.shard}: complete={result['complete']} scanned={scanned} "
        f"min={result['minimum_ewald_points']} gf2_deficient={len(gf2_deficient)} "
        f"seconds={result['elapsed_seconds']:.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--ranges", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--basis-trials", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args()
    scan(args)


if __name__ == "__main__":
    main()
