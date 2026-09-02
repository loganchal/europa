#!/usr/bin/env python3
"""Scan the encoded dimension-8 smooth-Fano classification for small Ewald sets.

The database stores the ray generators U of a smooth Fano polytope Q.  Its
polar monotone polytope P has symmetric lattice points

    E(P) = {x in Z^8 : |<u,x>| <= 1 for every u in U}.

Every encoded representative begins with the standard basis, so every such x
lies in {-1,0,1}^8.  Ray constraints are cached as 6561-bit Python integers.
All arithmetic used for membership and determinant checks is integral.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DIM = 8
BLOCK_SIZE = 7498
VALUES = (-1, 0, 1)
X = np.asarray(list(itertools.product(VALUES, repeat=DIM)), dtype=np.int16)
N_X = int(X.shape[0])
ALL_MASK = (1 << N_X) - 1
INDEX = {tuple(map(int, x)): i for i, x in enumerate(X)}
ZERO_INDEX = INDEX[(0,) * DIM]
EYE = tuple(tuple(1 if i == j else 0 for j in range(DIM)) for i in range(DIM))


def integer_to_sequence(n: int, base: int) -> list[int]:
    """Magma's IntegerToSequence(n, base): least-significant digit first."""
    if n < 0 or base < 2:
        raise ValueError("invalid encoded integer or base")
    out: list[int] = []
    while n:
        n, r = divmod(n, base)
        out.append(r)
    return out or [0]


def decode_record(text: str, base: int) -> list[tuple[int, ...]]:
    digits = integer_to_sequence(int(text), base)
    if len(digits) < 2:
        raise ValueError("short record")
    dim, shift = digits[0], digits[1]
    if dim != DIM:
        raise ValueError(f"unexpected dimension {dim}")
    coeffs = [a - shift for a in digits[2:]]
    if len(coeffs) % dim:
        raise ValueError("coefficient count is not divisible by dimension")
    vertices = [tuple(coeffs[i : i + dim]) for i in range(0, len(coeffs), dim)]
    if tuple(vertices[:DIM]) != EYE:
        raise ValueError("database representative does not begin with the standard basis")
    return vertices


class RayMasks:
    def __init__(self) -> None:
        self.cache: dict[tuple[int, ...], int] = {}

    def __getitem__(self, ray: tuple[int, ...]) -> int:
        found = self.cache.get(ray)
        if found is not None:
            return found
        u = np.asarray(ray, dtype=np.int16)
        valid = np.abs(X @ u) <= 1
        packed = np.packbits(valid, bitorder="little")
        mask = int.from_bytes(packed.tobytes(), byteorder="little", signed=False)
        self.cache[ray] = mask
        return mask


def ewald_mask(vertices: Sequence[tuple[int, ...]], masks: RayMasks) -> int:
    result = ALL_MASK
    # The first DIM rays are e_i and are already enforced by X in {-1,0,1}^8.
    for ray in vertices[DIM:]:
        result &= masks[ray]
    return result


def iter_set_bits(mask: int) -> Iterable[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def points_from_mask(mask: int) -> list[tuple[int, ...]]:
    return [tuple(map(int, X[i])) for i in iter_set_bits(mask)]


def antipodal_representatives(points: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
    reps: list[tuple[int, ...]] = []
    for v in points:
        if not any(v):
            continue
        first = next(a for a in v if a)
        if first > 0:
            reps.append(v)
    return reps


def determinant(columns: Sequence[Sequence[int]]) -> int:
    """Exact Bareiss determinant; columns must form an 8 by 8 matrix."""
    n = len(columns)
    if n == 0:
        return 1
    a = [[int(columns[j][i]) for j in range(n)] for i in range(n)]
    sign = 1
    prev = 1
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
                a[i][j] = (a[i][j] * pivot - a[i][k] * a[k][j]) // prev
        prev = pivot
        for i in range(k + 1, n):
            a[i][k] = 0
    return sign * a[n - 1][n - 1]


def basis_probe(
    points: Sequence[tuple[int, ...]],
    seed: int,
    combination_limit: int = 2_000_000,
    random_trials: int = 40_000,
) -> dict:
    """Exact when all 8-subsets fit under the limit; otherwise a one-sided probe."""
    reps = antipodal_representatives(points)
    m = len(reps)
    result = {
        "antipodal_nonzero_count": m,
        "method": "none",
        "subsets_checked": 0,
        "has_unimodular_basis": False,
        "exhaustive": False,
        "basis": None,
        "minimum_nonzero_abs_det_seen": None,
    }
    if m < DIM:
        result["method"] = "cardinality"
        result["exhaustive"] = True
        return result

    total = math.comb(m, DIM)
    best: int | None = None
    if total <= combination_limit:
        result["method"] = "all-combinations"
        result["exhaustive"] = True
        iterator = itertools.combinations(reps, DIM)
    else:
        result["method"] = "deterministic-random-combinations"
        rng = random.Random(seed)

        def draws() -> Iterable[tuple[tuple[int, ...], ...]]:
            seen: set[tuple[int, ...]] = set()
            trials = min(random_trials, total)
            while len(seen) < trials:
                ids = tuple(sorted(rng.sample(range(m), DIM)))
                if ids in seen:
                    continue
                seen.add(ids)
                yield tuple(reps[i] for i in ids)

        iterator = draws()

    for cols in iterator:
        det = determinant(cols)
        result["subsets_checked"] += 1
        if det:
            ad = abs(det)
            if best is None or ad < best:
                best = ad
            if ad == 1:
                result["has_unimodular_basis"] = True
                result["basis"] = [list(v) for v in cols]
                result["minimum_nonzero_abs_det_seen"] = 1
                return result
    result["minimum_nonzero_abs_det_seen"] = best
    return result


def parse_blocks(spec: str, data_dir: Path) -> list[int]:
    if spec == "all":
        return sorted(int(p.name[5:]) for p in data_dir.glob("block*") if p.name[5:].isdigit())
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = map(int, token.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    return sorted(out)


def scan(data_dir: Path, blocks: Sequence[int], top_k: int, output: Path) -> None:
    masks = RayMasks()
    counts: Counter[int] = Counter()
    vertex_counts: Counter[int] = Counter()
    top: list[tuple[int, int, dict]] = []
    per_block: list[dict] = []
    total = 0

    for block in blocks:
        path = data_dir / f"block{block}"
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            raise ValueError(f"empty block: {path}")
        base = int(lines[0])
        block_min = None
        block_min_ids: list[int] = []
        for offset, encoded in enumerate(lines[1:], start=1):
            poly_id = block * BLOCK_SIZE + offset
            vertices = decode_record(encoded, base)
            mask = ewald_mask(vertices, masks)
            n_points = mask.bit_count()
            if not (mask >> ZERO_INDEX) & 1:
                raise AssertionError("zero missing from Ewald set")
            if n_points % 2 != 1:
                raise AssertionError("Ewald set is not centrally symmetric")
            total += 1
            counts[n_points] += 1
            vertex_counts[len(vertices)] += 1

            if block_min is None or n_points < block_min:
                block_min = n_points
                block_min_ids = [poly_id]
            elif n_points == block_min:
                block_min_ids.append(poly_id)

            record = {
                "id": poly_id,
                "block": block,
                "offset": offset,
                "base": base,
                "n_vertices": len(vertices),
                "n_ewald_points": n_points,
                "ewald_mask_hex": hex(mask),
                "vertices": [list(v) for v in vertices],
            }
            heapq.heappush(top, (-n_points, -poly_id, record))
            if len(top) > top_k:
                heapq.heappop(top)

        per_block.append(
            {
                "block": block,
                "records": len(lines) - 1,
                "minimum_ewald_points": block_min,
                "minimum_ids": block_min_ids[:100],
                "minimum_id_count": len(block_min_ids),
            }
        )
        print(
            f"block {block:3d}: {len(lines)-1:4d} records, "
            f"min |E|={block_min}, distinct rays={len(masks.cache)}",
            flush=True,
        )

    retained = [entry[2] for entry in top]
    retained.sort(key=lambda r: (r["n_ewald_points"], r["id"]))
    for record in retained:
        mask = int(record["ewald_mask_hex"], 16)
        points = points_from_mask(mask)
        record["ewald_points"] = [list(v) for v in points]
        coordinate_basis = True
        for i in range(DIM):
            e = tuple(1 if i == j else 0 for j in range(DIM))
            coordinate_basis &= bool((mask >> INDEX[e]) & 1)
        record["contains_coordinate_basis"] = coordinate_basis
        record["basis_probe"] = basis_probe(points, seed=record["id"])

    result = {
        "dimension": DIM,
        "candidate_box": [[-1, 1] for _ in range(DIM)],
        "candidate_box_size": N_X,
        "blocks": list(blocks),
        "polytopes_scanned": total,
        "distinct_ray_constraints": len(masks.cache),
        "minimum_ewald_points": min(counts) if counts else None,
        "ewald_point_count_histogram": {str(k): counts[k] for k in sorted(counts)},
        "vertex_count_histogram": {str(k): vertex_counts[k] for k in sorted(vertex_counts)},
        "per_block": per_block,
        "smallest_ewald_sets": retained,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}; scanned {total} polytopes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--blocks", default="all")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("scan-summary.json"))
    args = parser.parse_args()
    blocks = parse_blocks(args.blocks, args.data_dir)
    scan(args.data_dir, blocks, args.top_k, args.output)


if __name__ == "__main__":
    main()
