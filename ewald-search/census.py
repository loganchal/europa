#!/usr/bin/env python3
"""Exact dimension-8 Ewald census over the Magma smooth-Fano database."""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DIM = 8
POINT_TUPLES = tuple(itertools.product((-1, 0, 1), repeat=DIM))
POINTS = np.asarray(POINT_TUPLES, dtype=np.int16)
NPOINTS = len(POINT_TUPLES)
NBYTES = (NPOINTS + 7) // 8
FULL_MASK = (1 << NPOINTS) - 1
POINT_INDEX = {p: i for i, p in enumerate(POINT_TUPLES)}
REP_SELECTOR = np.asarray([next((a > 0 for a in p if a), False) for p in POINT_TUPLES], dtype=bool)
WEIGHT = np.count_nonzero(POINTS, axis=1)
MOD2 = np.asarray([sum((int(a) & 1) << j for j, a in enumerate(p)) for p in POINT_TUPLES], dtype=np.uint16)


def decode_record(text: str, base: int) -> list[tuple[int, ...]]:
    n = int(text.strip())
    seq: list[int] = []
    while n:
        n, r = divmod(n, base)
        seq.append(r)
    if len(seq) < 2:
        raise ValueError("short encoded record")
    dim, shift = seq[0], seq[1]
    if dim != DIM:
        raise ValueError(f"expected dimension {DIM}, got {dim}")
    values = [a - shift for a in seq[2:]]
    if len(values) % dim:
        raise ValueError("coefficient count is not divisible by dimension")
    return [tuple(values[i:i + dim]) for i in range(0, len(values), dim)]


def canonical_sign(v: Sequence[int]) -> tuple[int, ...]:
    t = tuple(int(a) for a in v)
    for a in t:
        if a:
            return t if a > 0 else tuple(-x for x in t)
    return t


def determinant(columns: Sequence[Sequence[int]]) -> int:
    n = len(columns)
    if n == 0:
        return 1
    a = [[int(columns[j][i]) for j in range(n)] for i in range(n)]
    sign = 1
    previous = 1
    for k in range(n - 1):
        pivot = next((r for r in range(k, n) if a[r][k]), None)
        if pivot is None:
            return 0
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            sign = -sign
        p = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = a[i][j] * p - a[i][k] * a[k][j]
                if previous != 1:
                    if numerator % previous:
                        raise ArithmeticError("Bareiss division was not exact")
                    numerator //= previous
                a[i][j] = numerator
        previous = p
        for i in range(k + 1, n):
            a[i][k] = 0
    return sign * a[n - 1][n - 1]


def gf2_insert(basis: list[int], x: int) -> bool:
    y = int(x)
    for pivot in range(DIM - 1, -1, -1):
        if not ((y >> pivot) & 1):
            continue
        if basis[pivot]:
            y ^= basis[pivot]
        else:
            basis[pivot] = y
            return True
    return False


def gf2_rank(indices: Sequence[int]) -> int:
    basis = [0] * DIM
    rank = 0
    for idx in indices:
        if gf2_insert(basis, int(MOD2[idx])):
            rank += 1
            if rank == DIM:
                break
    return rank


def select_gf2_basis(order: Iterable[int]) -> list[int] | None:
    basis = [0] * DIM
    selected: list[int] = []
    for idx in order:
        if gf2_insert(basis, int(MOD2[idx])):
            selected.append(int(idx))
            if len(selected) == DIM:
                return selected
    return None


def rank_mod_p(vectors: Sequence[Sequence[int]], p: int) -> int:
    rows = [[int(x) % p for x in v] for v in vectors if any(v)]
    rank = 0
    col = 0
    while rank < len(rows) and col < DIM:
        pivot = next((r for r in range(rank, len(rows)) if rows[r][col]), None)
        if pivot is None:
            col += 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inv = pow(rows[rank][col], -1, p)
        rows[rank] = [(x * inv) % p for x in rows[rank]]
        for r in range(len(rows)):
            if r == rank or not rows[r][col]:
                continue
            factor = rows[r][col]
            rows[r] = [(rows[r][c] - factor * rows[rank][c]) % p for c in range(DIM)]
        rank += 1
        col += 1
        if rank == DIM:
            break
    return rank


def basis_bitmask(indices: Sequence[int]) -> int:
    ans = 0
    for idx in indices:
        ans |= 1 << int(idx)
    return ans


def unpack_representatives(mask: int) -> np.ndarray:
    raw = mask.to_bytes(NBYTES, "little")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:NPOINTS]
    return np.flatnonzero(bits & REP_SELECTOR)


def random_gf2_basis(reps: Sequence[int], rng: random.Random) -> list[int] | None:
    n = len(reps)
    selected: list[int] = []
    selected_set: set[int] = set()
    basis = [0] * DIM
    for _ in range(min(max(64, 4 * n), 1024)):
        idx = int(reps[rng.randrange(n)])
        if idx in selected_set:
            continue
        selected_set.add(idx)
        if gf2_insert(basis, int(MOD2[idx])):
            selected.append(idx)
            if len(selected) == DIM:
                return selected
    for idx0 in reps:
        idx = int(idx0)
        if idx in selected_set:
            continue
        if gf2_insert(basis, int(MOD2[idx])):
            selected.append(idx)
            if len(selected) == DIM:
                return selected
    return None


def cofactor_vector(columns: Sequence[Sequence[int]], column: int) -> list[int]:
    a = [[int(columns[j][i]) for j in range(DIM)] for i in range(DIM)]
    result: list[int] = []
    for row in range(DIM):
        minor_rows = [r for r in range(DIM) if r != row]
        minor_cols = [c for c in range(DIM) if c != column]
        minor_as_columns = [tuple(a[r][c] for r in minor_rows) for c in minor_cols]
        d = determinant(minor_as_columns)
        result.append(-d if (row + column) & 1 else d)
    return result


def one_swap_to_unimodular(selected: Sequence[int], reps: Sequence[int]) -> list[int] | None:
    columns = [POINT_TUPLES[int(i)] for i in selected]
    selected_set = set(int(i) for i in selected)
    for j in range(DIM):
        cof = cofactor_vector(columns, j)
        for idx0 in reps:
            idx = int(idx0)
            if idx in selected_set:
                continue
            v = POINT_TUPLES[idx]
            d = sum(cof[i] * v[i] for i in range(DIM))
            if abs(d) == 1:
                answer = list(map(int, selected))
                answer[j] = idx
                if abs(determinant([POINT_TUPLES[i] for i in answer])) != 1:
                    raise AssertionError("cofactor replacement check failed")
                return answer
    return None


def find_unimodular_basis(reps_array: Sequence[int], attempts: int = 96) -> list[int] | None:
    reps = [int(i) for i in reps_array]
    orders = [
        sorted(reps, key=lambda i: (int(WEIGHT[i]), i)),
        sorted(reps, key=lambda i: (-int(WEIGHT[i]), i)),
        list(reversed(reps)),
    ]
    near: list[int] | None = None
    near_det: int | None = None
    for order in orders:
        chosen = select_gf2_basis(order)
        if chosen is None:
            return None
        d = abs(determinant([POINT_TUPLES[i] for i in chosen]))
        if d == 1:
            return chosen
        if near_det is None or d < near_det:
            near, near_det = chosen, d
    if near is not None:
        swapped = one_swap_to_unimodular(near, reps)
        if swapped is not None:
            return swapped
    rng = random.Random(hash(tuple(reps)) ^ 0xE7A1D8)
    best: list[int] | None = near
    best_det = near_det if near_det is not None else 10**9
    for trial in range(attempts):
        chosen = random_gf2_basis(reps, rng)
        if chosen is None:
            return None
        d = abs(determinant([POINT_TUPLES[i] for i in chosen]))
        if d == 1:
            return chosen
        if d < best_det:
            best, best_det = chosen, d
            if trial % 8 == 7:
                swapped = one_swap_to_unimodular(best, reps)
                if swapped is not None:
                    return swapped
    if best is not None:
        return one_swap_to_unimodular(best, reps)
    return None


def lattice_index(vectors: Sequence[Sequence[int]]) -> int | None:
    from sympy import Matrix, ZZ
    from sympy.matrices.normalforms import smith_normal_form
    m = Matrix([[int(x) for x in v] for v in vectors])
    s = smith_normal_form(m, domain=ZZ)
    diag = [abs(int(s[i, i])) for i in range(min(s.rows, s.cols)) if s[i, i] != 0]
    if len(diag) < DIM:
        return None
    return math.prod(diag[:DIM])


def seed_hot_bases(limit: int = 32) -> list[int]:
    matrices: list[list[tuple[int, ...]]] = []
    identity = [tuple(1 if i == j else 0 for i in range(DIM)) for j in range(DIM)]
    matrices.append(identity)
    rng = random.Random(0xE0A1D)
    while len(matrices) < limit:
        lower = [[0] * DIM for _ in range(DIM)]
        for i in range(DIM):
            lower[i][i] = 1
            for j in range(i):
                lower[i][j] = rng.choice((-1, 0, 1))
        matrices.append([tuple(lower[i][j] for i in range(DIM)) for j in range(DIM)])
    masks: list[int] = []
    seen: set[int] = set()
    for cols in matrices:
        bm = basis_bitmask([POINT_INDEX[c] for c in cols])
        if bm not in seen:
            seen.add(bm)
            masks.append(bm)
    return masks


def record_payload(poly_id: int, rays: Sequence[Sequence[int]], ewald_mask: int, reason: str,
                   ranks: dict[str, int] | None = None, index: int | None = None) -> dict:
    reps = unpack_representatives(ewald_mask)
    epoints = [list(POINT_TUPLES[i]) for i in range(NPOINTS) if (ewald_mask >> i) & 1]
    payload = {
        "dimension": DIM,
        "database_id": poly_id,
        "database_convention": "rays/facet_normals u; P={x: u.x <= 1}",
        "reason": reason,
        "primitive_facet_normals": [list(map(int, v)) for v in rays],
        "ewald_points": epoints,
        "ewald_point_count": ewald_mask.bit_count(),
        "representative_count_mod_sign": len(reps),
        "bounding_box": [[-1, 1] for _ in range(DIM)],
    }
    if ranks is not None:
        payload["ranks_mod_p"] = ranks
    if index is not None:
        payload["generated_lattice_index"] = index
    return payload


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-polytopes", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=96)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    block_files = sorted(args.data.glob("block*"), key=lambda p: int(p.name[5:]))
    if not block_files:
        raise FileNotFoundError(f"no block files under {args.data}")
    ray_masks: dict[tuple[int, ...], int] = {}
    ewald_cache: dict[int, int] = {}
    cache_limit = 250_000
    hot = seed_hot_bases(32)
    minima: list[tuple[int, int, dict]] = []
    unresolved: list[dict] = []
    total = hot_hits = duplicate_hits = computed_bases = 0
    min_count = NPOINTS
    start = time.time()
    standard = {tuple(1 if i == j else 0 for i in range(DIM)) for j in range(DIM)}

    def ray_mask(ray: tuple[int, ...]) -> int:
        key = canonical_sign(ray)
        cached = ray_masks.get(key)
        if cached is not None:
            return cached
        valid = np.abs(POINTS @ np.asarray(key, dtype=np.int16)) <= 1
        result = int.from_bytes(np.packbits(valid, bitorder="little").tobytes(), "little") & FULL_MASK
        ray_masks[key] = result
        return result

    for block_path in block_files:
        block_number = int(block_path.name[5:])
        with block_path.open() as fh:
            base_line = fh.readline()
            if not base_line:
                continue
            base = int(base_line.strip())
            for local_index, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                poly_id = block_number * 7498 + local_index + 1
                rays = decode_record(line, base)
                if not standard.issubset(set(rays)):
                    raise AssertionError(f"ID {poly_id} lacks the normalized standard cone")
                mask = FULL_MASK
                for ray in rays:
                    mask &= ray_mask(ray)
                total += 1
                count = mask.bit_count()
                min_count = min(min_count, count)
                small = {"database_id": poly_id, "ewald_point_count": count, "ray_count": len(rays),
                         "primitive_facet_normals": [list(v) for v in rays]}
                item = (-count, -poly_id, small)
                if len(minima) < 200:
                    heapq.heappush(minima, item)
                elif item > minima[0]:
                    heapq.heapreplace(minima, item)

                cached_basis = ewald_cache.get(mask)
                if cached_basis is not None:
                    duplicate_hits += 1
                    if args.max_polytopes and total >= args.max_polytopes:
                        break
                    continue
                found_mask = 0
                found_pos = -1
                for pos, bm in enumerate(hot):
                    if mask & bm == bm:
                        found_mask, found_pos = bm, pos
                        break
                if found_mask:
                    hot_hits += 1
                    if found_pos > 0:
                        hot.insert(0, hot.pop(found_pos))
                else:
                    reps_array = unpack_representatives(mask)
                    rank2 = gf2_rank(reps_array)
                    if rank2 < DIM:
                        vectors = [POINT_TUPLES[int(i)] for i in reps_array]
                        ranks = {"2": rank2, **{str(p): rank_mod_p(vectors, p) for p in (3, 5, 7, 11)}}
                        payload = record_payload(poly_id, rays, mask, "rank of Ewald points modulo 2 is below dimension", ranks)
                        save_json(args.output / "COUNTEREXAMPLE.json", payload)
                        print(json.dumps({"counterexample": poly_id, "reason": payload["reason"]}), flush=True)
                        return 0
                    selected = find_unimodular_basis(reps_array, args.attempts)
                    if selected is not None:
                        found_mask = basis_bitmask(selected)
                        computed_bases += 1
                        hot.insert(0, found_mask)
                        dedup: list[int] = []
                        seen_hot: set[int] = set()
                        for b in hot:
                            if b not in seen_hot:
                                seen_hot.add(b)
                                dedup.append(b)
                            if len(dedup) == 48:
                                break
                        hot = dedup
                    else:
                        vectors = [POINT_TUPLES[int(i)] for i in reps_array]
                        ranks = {str(p): rank_mod_p(vectors, p) for p in (2, 3, 5, 7, 11, 13, 17, 19)}
                        bad_prime = next((p for p, r in ranks.items() if r < DIM), None)
                        if bad_prime is not None:
                            payload = record_payload(poly_id, rays, mask, f"rank of Ewald points modulo {bad_prime} is below dimension", ranks)
                            save_json(args.output / "COUNTEREXAMPLE.json", payload)
                            print(json.dumps({"counterexample": poly_id, "reason": payload["reason"]}), flush=True)
                            return 0
                        index = lattice_index(vectors)
                        if index is None or index > 1:
                            payload = record_payload(poly_id, rays, mask, "Ewald points generate a proper sublattice", ranks, index)
                            save_json(args.output / "COUNTEREXAMPLE.json", payload)
                            print(json.dumps({"counterexample": poly_id, "reason": payload["reason"], "index": index}), flush=True)
                            return 0
                        payload = record_payload(poly_id, rays, mask,
                            "heuristic search found no unimodular subset, but generated lattice has index one", ranks, index)
                        unresolved.append(payload)
                        save_json(args.output / "unresolved.json", unresolved)
                if len(ewald_cache) < cache_limit:
                    ewald_cache[mask] = found_mask
                if total % 25_000 == 0:
                    print(json.dumps({"processed": total, "id": poly_id, "elapsed_sec": round(time.time() - start, 1),
                        "min_ewald": min_count, "ray_masks": len(ray_masks), "ewald_cache": len(ewald_cache),
                        "hot_hits": hot_hits, "duplicates": duplicate_hits, "computed_bases": computed_bases,
                        "unresolved": len(unresolved)}), flush=True)
                if args.max_polytopes and total >= args.max_polytopes:
                    break
        summary = {"dimension": DIM, "processed": total, "last_block": block_number,
            "elapsed_sec": round(time.time() - start, 3), "minimum_ewald_count": min_count,
            "distinct_ray_constraints": len(ray_masks), "cached_ewald_sets": len(ewald_cache),
            "hot_basis_hits": hot_hits, "duplicate_ewald_hits": duplicate_hits,
            "new_bases_computed": computed_bases, "unresolved_count": len(unresolved),
            "smallest_200": [x[2] for x in sorted(minima, reverse=True)]}
        save_json(args.output / "summary.json", summary)
        if args.max_polytopes and total >= args.max_polytopes:
            break
    summary = json.loads((args.output / "summary.json").read_text())
    summary["complete"] = total == 749_892
    summary["expected_total"] = 749_892
    save_json(args.output / "summary.json", summary)
    print(json.dumps({"complete": summary["complete"], "processed": total,
        "minimum_ewald_count": min_count, "unresolved": len(unresolved),
        "elapsed_sec": round(time.time() - start, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
