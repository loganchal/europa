#!/usr/bin/env python3
"""Exact ordinary-Ewald census for one dimension-9 classification shard.

Input is a gzip file of records headed FACETS.  Each following row is
(1,a_1,...,a_9) and means 0 <= 1 + a.x, equivalently (-a).x <= 1.
All arithmetic used to accept a basis is integral and exact.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

DIM = 9
POINT_TUPLES = tuple(itertools.product((-1, 0, 1), repeat=DIM))
POINTS = np.asarray(POINT_TUPLES, dtype=np.int16)
NPOINTS = len(POINT_TUPLES)
NBYTES = (NPOINTS + 7) // 8
FULL_MASK = (1 << NPOINTS) - 1
POINT_INDEX = {p: i for i, p in enumerate(POINT_TUPLES)}
REP_SELECTOR = np.asarray(
    [next((a > 0 for a in p if a), False) for p in POINT_TUPLES],
    dtype=bool,
)
WEIGHT = np.count_nonzero(POINTS, axis=1)
MOD2 = np.asarray(
    [sum((int(a) & 1) << j for j, a in enumerate(p)) for p in POINT_TUPLES],
    dtype=np.uint16,
)


def determinant(columns: Sequence[Sequence[int]]) -> int:
    """Bareiss fraction-free exact determinant; arguments are columns."""
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
                z = a[i][j] * p - a[i][k] * a[k][j]
                if previous != 1:
                    if z % previous:
                        raise ArithmeticError("non-exact Bareiss division")
                    z //= previous
                a[i][j] = z
        previous = p
        for i in range(k + 1, n):
            a[i][k] = 0
    return sign * a[-1][-1]


def records(path: Path, expected_facets: int) -> Iterator[list[tuple[int, ...]]]:
    current: list[tuple[int, ...]] | None = None
    with gzip.open(path, "rt", encoding="ascii", newline="") as fh:
        for line_no, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            if text == "FACETS":
                if current is not None:
                    if len(current) != expected_facets:
                        raise ValueError(
                            f"record before line {line_no}: {len(current)} facets, "
                            f"expected {expected_facets}"
                        )
                    yield current
                current = []
                continue
            if current is None:
                raise ValueError(f"data before FACETS at line {line_no}")
            values = tuple(map(int, text.split()))
            if len(values) != DIM + 1 or values[0] != 1:
                raise ValueError(f"invalid facet row at line {line_no}: {values}")
            current.append(values[1:])
    if current is not None:
        if len(current) != expected_facets:
            raise ValueError(
                f"last record has {len(current)} facets, expected {expected_facets}"
            )
        yield current


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
            rows[r] = [
                (rows[r][c] - factor * rows[rank][c]) % p
                for c in range(DIM)
            ]
        rank += 1
        col += 1
        if rank == DIM:
            break
    return rank


def basis_bitmask(indices: Iterable[int]) -> int:
    answer = 0
    for idx in indices:
        answer |= 1 << int(idx)
    return answer


def unpack_representatives(mask: int) -> np.ndarray:
    bits = np.unpackbits(
        np.frombuffer(mask.to_bytes(NBYTES, "little"), dtype=np.uint8),
        bitorder="little",
    )[:NPOINTS]
    return np.flatnonzero(bits & REP_SELECTOR)


def cofactor_vector(columns: Sequence[Sequence[int]], column: int) -> list[int]:
    matrix = [[int(columns[j][i]) for j in range(DIM)] for i in range(DIM)]
    answer: list[int] = []
    for row in range(DIM):
        remaining_rows = [r for r in range(DIM) if r != row]
        remaining_columns = [c for c in range(DIM) if c != column]
        minor_columns = [
            tuple(matrix[r][c] for r in remaining_rows) for c in remaining_columns
        ]
        value = determinant(minor_columns)
        answer.append(-value if (row + column) & 1 else value)
    return answer


def one_swap_to_unimodular(
    selected: Sequence[int], representatives: Sequence[int]
) -> list[int] | None:
    columns = [POINT_TUPLES[int(i)] for i in selected]
    selected_set = set(map(int, selected))
    for column in range(DIM):
        cofactors = cofactor_vector(columns, column)
        for raw_idx in representatives:
            idx = int(raw_idx)
            if idx in selected_set:
                continue
            vector = POINT_TUPLES[idx]
            value = sum(cofactors[k] * vector[k] for k in range(DIM))
            if abs(value) == 1:
                answer = list(map(int, selected))
                answer[column] = idx
                if abs(determinant([POINT_TUPLES[i] for i in answer])) != 1:
                    raise AssertionError("cofactor replacement check failed")
                return answer
    return None


def find_unimodular_basis(
    representatives: Sequence[int], attempts: int
) -> list[int] | None:
    reps = list(map(int, representatives))
    orders = [
        sorted(reps, key=lambda i: (int(WEIGHT[i]), i)),
        sorted(reps, key=lambda i: (-int(WEIGHT[i]), i)),
        reps,
        list(reversed(reps)),
    ]
    best: list[int] | None = None
    best_det = 10**9
    for order in orders:
        selected = select_gf2_basis(order)
        if selected is None:
            return None
        value = abs(determinant([POINT_TUPLES[i] for i in selected]))
        if value == 1:
            return selected
        if value < best_det:
            best, best_det = selected, value
    if best is not None:
        swapped = one_swap_to_unimodular(best, reps)
        if swapped is not None:
            return swapped

    rng = random.Random(hash(tuple(reps)) ^ 0xE9A1D)
    for _ in range(attempts):
        order = reps.copy()
        rng.shuffle(order)
        selected = select_gf2_basis(order)
        if selected is None:
            return None
        value = abs(determinant([POINT_TUPLES[i] for i in selected]))
        if value == 1:
            return selected
        if value < best_det:
            best, best_det = selected, value
            swapped = one_swap_to_unimodular(best, reps)
            if swapped is not None:
                return swapped
    return one_swap_to_unimodular(best, reps) if best is not None else None


def lattice_index(vectors: Sequence[Sequence[int]]) -> int | None:
    from sympy import Matrix, ZZ
    from sympy.matrices.normalforms import smith_normal_form

    matrix = Matrix([[int(x) for x in vector] for vector in vectors])
    smith = smith_normal_form(matrix, domain=ZZ)
    diagonal = [
        abs(int(smith[i, i]))
        for i in range(min(smith.rows, smith.cols))
        if smith[i, i] != 0
    ]
    return None if len(diagonal) < DIM else math.prod(diagonal[:DIM])


def seed_hot_bases(limit: int = 48) -> list[int]:
    matrices: list[list[tuple[int, ...]]] = []
    identity = [
        tuple(1 if i == j else 0 for i in range(DIM)) for j in range(DIM)
    ]
    matrices.append(identity)
    rng = random.Random(0xE9A1D)
    while len(matrices) < limit:
        lower = [[0] * DIM for _ in range(DIM)]
        for i in range(DIM):
            lower[i][i] = 1
            for j in range(i):
                lower[i][j] = rng.choice((-1, 0, 1))
        matrices.append(
            [tuple(lower[i][j] for i in range(DIM)) for j in range(DIM)]
        )
    answer: list[int] = []
    seen: set[int] = set()
    for columns in matrices:
        bitmask = basis_bitmask(POINT_INDEX[column] for column in columns)
        if bitmask not in seen:
            seen.add(bitmask)
            answer.append(bitmask)
    return answer


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def candidate_payload(
    source_file: str,
    ordinal: int,
    coefficients: Sequence[Sequence[int]],
    mask: int,
    reason: str,
    ranks: dict[str, int] | None = None,
    index: int | None = None,
) -> dict:
    points = [
        list(POINT_TUPLES[i]) for i in range(NPOINTS) if (mask >> i) & 1
    ]
    value = {
        "dimension": DIM,
        "source_file": source_file,
        "ordinal_in_file": ordinal,
        "reason": reason,
        "primitive_facet_normals_leq_1": [
            [-int(x) for x in coefficient] for coefficient in coefficients
        ],
        "ewald_points": points,
        "ewald_point_count": len(points),
        "bounding_box": [[-1, 1] for _ in range(DIM)],
    }
    if ranks is not None:
        value["ranks_mod_p"] = ranks
    if index is not None:
        value["generated_lattice_index"] = index
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--facets", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-polytopes", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=192)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    constraint_masks: dict[tuple[int, ...], int] = {}
    ewald_cache: dict[int, int] = {}
    hot_bases = seed_hot_bases()
    minima: list[tuple[int, int, dict]] = []
    processed = hot_hits = duplicate_hits = computed_bases = 0
    minimum_count = NPOINTS
    started = time.time()
    certificate = hashlib.sha256()
    standard = {
        tuple(-1 if i == j else 0 for i in range(DIM)) for j in range(DIM)
    }

    def constraint_mask(coefficient: tuple[int, ...]) -> int:
        cached = constraint_masks.get(coefficient)
        if cached is not None:
            return cached
        valid = np.abs(POINTS @ np.asarray(coefficient, dtype=np.int16)) <= 1
        packed = np.packbits(valid, bitorder="little").tobytes()
        answer = int.from_bytes(packed, "little") & FULL_MASK
        constraint_masks[coefficient] = answer
        return answer

    for ordinal, coefficients in enumerate(
        records(args.input, args.facets), start=1
    ):
        if not standard.issubset(set(coefficients)):
            raise AssertionError(f"record {ordinal} lacks the normalized coordinate facets")
        mask = FULL_MASK
        for coefficient in coefficients:
            mask &= constraint_mask(coefficient)

        processed += 1
        point_count = mask.bit_count()
        minimum_count = min(minimum_count, point_count)
        small = {
            "source_file": args.input.name,
            "ordinal_in_file": ordinal,
            "ewald_point_count": point_count,
            "facet_count": len(coefficients),
            "primitive_facet_normals_leq_1": [
                [-int(x) for x in coefficient] for coefficient in coefficients
            ],
        }
        item = (-point_count, -ordinal, small)
        if len(minima) < 100:
            heapq.heappush(minima, item)
        elif item > minima[0]:
            heapq.heapreplace(minima, item)

        basis_mask = ewald_cache.get(mask, 0)
        if basis_mask:
            duplicate_hits += 1
        else:
            hot_position = next(
                (
                    i
                    for i, candidate_basis in enumerate(hot_bases)
                    if mask & candidate_basis == candidate_basis
                ),
                -1,
            )
            if hot_position >= 0:
                basis_mask = hot_bases[hot_position]
                hot_hits += 1
                if hot_position:
                    hot_bases.insert(0, hot_bases.pop(hot_position))
            else:
                representatives = unpack_representatives(mask)
                rank_two = gf2_rank(representatives)
                if rank_two < DIM:
                    vectors = [POINT_TUPLES[int(i)] for i in representatives]
                    ranks = {"2": rank_two}
                    ranks.update(
                        {str(p): rank_mod_p(vectors, p) for p in (3, 5, 7, 11)}
                    )
                    payload = candidate_payload(
                        args.input.name,
                        ordinal,
                        coefficients,
                        mask,
                        "rank modulo 2 below dimension",
                        ranks,
                    )
                    save_json(args.output / "COUNTEREXAMPLE.json", payload)
                    print(
                        json.dumps(
                            {
                                "counterexample": ordinal,
                                "ewald_point_count": point_count,
                                "reason": "rank_mod_2",
                            }
                        ),
                        flush=True,
                    )
                    return 0

                selected = find_unimodular_basis(representatives, args.attempts)
                if selected is not None:
                    basis_mask = basis_bitmask(selected)
                    computed_bases += 1
                    if mask & basis_mask != basis_mask:
                        raise AssertionError("reported basis is not contained in E(P)")
                    if abs(
                        determinant([POINT_TUPLES[i] for i in selected])
                    ) != 1:
                        raise AssertionError("reported basis is not unimodular")
                    hot_bases.insert(0, basis_mask)
                    hot_bases = list(dict.fromkeys(hot_bases))[:64]
                else:
                    vectors = [POINT_TUPLES[int(i)] for i in representatives]
                    ranks = {
                        str(p): rank_mod_p(vectors, p)
                        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
                    }
                    bad_prime = next(
                        (p for p, rank in ranks.items() if rank < DIM), None
                    )
                    if bad_prime is not None:
                        payload = candidate_payload(
                            args.input.name,
                            ordinal,
                            coefficients,
                            mask,
                            f"rank modulo {bad_prime} below dimension",
                            ranks,
                        )
                        save_json(args.output / "COUNTEREXAMPLE.json", payload)
                        print(
                            json.dumps(
                                {
                                    "counterexample": ordinal,
                                    "ewald_point_count": point_count,
                                    "reason": f"rank_mod_{bad_prime}",
                                }
                            ),
                            flush=True,
                        )
                        return 0
                    index = lattice_index(vectors)
                    if index is None or index > 1:
                        payload = candidate_payload(
                            args.input.name,
                            ordinal,
                            coefficients,
                            mask,
                            "Ewald points generate a proper sublattice",
                            ranks,
                            index,
                        )
                        save_json(args.output / "COUNTEREXAMPLE.json", payload)
                        print(
                            json.dumps(
                                {
                                    "counterexample": ordinal,
                                    "ewald_point_count": point_count,
                                    "reason": "lattice_index",
                                    "index": index,
                                }
                            ),
                            flush=True,
                        )
                        return 0
                    payload = candidate_payload(
                        args.input.name,
                        ordinal,
                        coefficients,
                        mask,
                        "unresolved: heuristic found no basis but lattice index is one",
                        ranks,
                        index,
                    )
                    save_json(args.output / "UNRESOLVED.json", payload)
                    print(
                        json.dumps(
                            {
                                "unresolved": ordinal,
                                "ewald_point_count": point_count,
                            }
                        ),
                        flush=True,
                    )
                    return 0

            if len(ewald_cache) < 50_000:
                ewald_cache[mask] = basis_mask

        certificate.update(ordinal.to_bytes(8, "little"))
        certificate.update(point_count.to_bytes(4, "little"))
        certificate.update(hashlib.sha256(mask.to_bytes(NBYTES, "little")).digest())
        certificate.update(basis_mask.to_bytes(NBYTES, "little"))

        if processed % 250_000 == 0:
            print(
                json.dumps(
                    {
                        "facets": args.facets,
                        "processed": processed,
                        "elapsed_sec": round(time.time() - started, 1),
                        "minimum_ewald_count": minimum_count,
                        "distinct_facet_normals": len(constraint_masks),
                        "hot_basis_hits": hot_hits,
                        "duplicate_ewald_hits": duplicate_hits,
                        "new_bases_computed": computed_bases,
                    }
                ),
                flush=True,
            )
        if args.max_polytopes and processed >= args.max_polytopes:
            break

    summary = {
        "dimension": DIM,
        "source_file": args.input.name,
        "facet_count": args.facets,
        "processed": processed,
        "complete": not bool(args.max_polytopes),
        "elapsed_sec": round(time.time() - started, 3),
        "minimum_ewald_count": minimum_count,
        "distinct_facet_normals": len(constraint_masks),
        "cached_ewald_sets": len(ewald_cache),
        "hot_basis_hits": hot_hits,
        "duplicate_ewald_hits": duplicate_hits,
        "new_bases_computed": computed_bases,
        "certificate_sha256": certificate.hexdigest(),
        "smallest_100": [item[2] for item in sorted(minima, reverse=True)],
    }
    save_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "source_file",
                    "processed",
                    "minimum_ewald_count",
                    "elapsed_sec",
                    "certificate_sha256",
                )
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
