#!/usr/bin/env python3
"""Search classified smooth Fano 9-polytopes for a non-neat witness.

For Q={x:u_i.x<=1}, normalize a displacement b modulo translations by
setting b_i=0 on one unimodular maximal cone C0.  Q_{+b} and Q_{-b}
have the same normal fan as Q exactly when, for every maximal cone C and
j outside C,

    |b_j-u_j U_C^{-1} b_C| <= 1-u_j U_C^{-1}1-1.

A normalized displacement is non-neat exactly when no x in Z^9 satisfies
|u_i.x-b_i|<=1 for every i.  The C0 inequalities reduce the possible x
to 3^9 exact candidates.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError, cKDTree
from sympy import Matrix, ZZ, zeros
from sympy.matrices.normalforms import smith_normal_decomp

DIM = 9
CUBE = np.asarray(list(itertools.product((-1, 0, 1), repeat=DIM)), dtype=np.int64)
EXPECTED = {
    10: 1, 11: 91, 12: 3331, 13: 63971, 14: 583544, 15: 2039665,
    16: 2822309, 17: 1829247, 18: 666151, 19: 173077, 20: 39218,
    21: 7515, 22: 1324, 23: 226, 24: 44, 25: 5, 26: 2,
}


def records(path: Path, facets: int) -> Iterator[list[tuple[int, ...]]]:
    current: list[tuple[int, ...]] | None = None
    with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            if text == "FACETS":
                if current is not None:
                    if len(current) != facets:
                        raise ValueError(f"record before {line_no}: expected {facets} rows")
                    yield current
                current = []
                continue
            if current is None:
                raise ValueError(f"data before FACETS at line {line_no}")
            row = tuple(map(int, text.split()))
            if len(row) != DIM + 1 or row[0] != 1:
                raise ValueError(f"bad row at line {line_no}: {row}")
            current.append(tuple(-x for x in row[1:]))  # outward normals u
    if current is not None:
        if len(current) != facets:
            raise ValueError(f"last record: expected {facets} rows")
        yield current


def det_bareiss(rows: Sequence[Sequence[int]]) -> int:
    a = [list(map(int, row)) for row in rows]
    n = len(a)
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
                value = a[i][j] * p - a[i][k] * a[k][j]
                if previous != 1:
                    if value % previous:
                        raise ArithmeticError("non-exact Bareiss division")
                    value //= previous
                a[i][j] = value
        previous = p
        for i in range(k + 1, n):
            a[i][k] = 0
    return sign * a[-1][-1]


def inverse_unimodular(rows: Sequence[Sequence[int]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.int64)
    inverse = np.rint(np.linalg.inv(matrix.astype(float))).astype(np.int64)
    if not np.array_equal(matrix @ inverse, np.eye(len(rows), dtype=np.int64)):
        exact = Matrix(rows).inv()
        inverse = np.asarray(exact.tolist(), dtype=np.int64)
        if not np.array_equal(matrix @ inverse, np.eye(len(rows), dtype=np.int64)):
            raise ValueError("matrix is not unimodular")
    return inverse


def maximal_cones(rays: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    hull = ConvexHull(np.asarray(rays, dtype=float), qhull_options="Qx")
    cones = sorted({tuple(sorted(map(int, simplex))) for simplex in hull.simplices})
    if any(len(cone) != DIM for cone in cones):
        raise ValueError("dual Fano polytope is not simplicial")
    ray_array = np.asarray(rays, dtype=np.int64)
    one = np.ones(DIM, dtype=np.int64)
    seen: set[tuple[int, ...]] = set()
    for cone in cones:
        rows = ray_array[list(cone)]
        if abs(det_bareiss(rows.tolist())) != 1:
            raise ValueError("non-unimodular cone returned by hull")
        inverse = inverse_unimodular(rows.tolist())
        vertex = inverse @ one
        values = ray_array @ vertex
        equality = tuple(np.flatnonzero(values == 1))
        if np.any(values > 1) or set(equality) != set(cone):
            raise ValueError("hull cone fails exact Fano check")
        key = tuple(map(int, vertex))
        if key in seen:
            raise ValueError("duplicate anticanonical vertex")
        seen.add(key)
    return cones


def canonical(c: Sequence[int]) -> tuple[int, ...]:
    value = list(map(int, c))
    for x in value:
        if x:
            if x < 0:
                value = [-y for y in value]
            break
    divisor = 0
    for x in value:
        divisor = math.gcd(divisor, abs(x))
    return tuple(x // divisor for x in value) if divisor > 1 else tuple(value)


def support_constraints(
    rays: Sequence[Sequence[int]], cones: Sequence[Sequence[int]], ref: Sequence[int]
) -> tuple[list[int], list[tuple[tuple[int, ...], int]]]:
    n_rays = len(rays)
    ref_set = set(ref)
    free = [i for i in range(n_rays) if i not in ref_set]
    free_position = {index: position for position, index in enumerate(free)}
    ray_array = np.asarray(rays, dtype=np.int64)
    one = np.ones(DIM, dtype=np.int64)
    tightest: dict[tuple[int, ...], int] = {}

    for cone0 in cones:
        cone = tuple(cone0)
        rows = ray_array[list(cone)]
        inverse = inverse_unimodular(rows.tolist())
        vertex = inverse @ one
        for j in range(n_rays):
            if j in cone:
                continue
            slack = 1 - int(ray_array[j] @ vertex)
            if slack < 1:
                raise ValueError("input fan is not strictly Fano")
            coefficient = np.zeros(len(free), dtype=np.int64)
            if j in free_position:
                coefficient[free_position[j]] += 1
            coordinates = ray_array[j] @ inverse
            for position, index in enumerate(cone):
                if index in free_position:
                    coefficient[free_position[index]] -= int(coordinates[position])
            if not np.any(coefficient):
                continue
            bound = slack - 1
            raw = tuple(map(int, coefficient))
            divisor = 0
            for value in raw:
                divisor = math.gcd(divisor, abs(value))
            if divisor > 1:
                raw = tuple(value // divisor for value in raw)
                bound //= divisor
            key = canonical(raw)
            old = tightest.get(key)
            if old is None or bound < old:
                tightest[key] = bound
    return free, sorted(tightest.items())


def integer_kernel(equations: Sequence[Sequence[int]], columns: int) -> np.ndarray:
    if not equations:
        return np.eye(columns, dtype=np.int64)
    matrix = Matrix(equations)
    diagonal, left, right = smith_normal_decomp(matrix, domain=ZZ)
    rank = sum(
        diagonal[i, i] != 0 for i in range(min(diagonal.rows, diagonal.cols))
    )
    basis = right[:, rank:]
    if matrix * basis != zeros(matrix.rows, basis.cols):
        raise AssertionError("Smith kernel computation failed")
    return np.asarray(basis.tolist(), dtype=np.int64)


def reduced_problem(
    constraints: Sequence[tuple[tuple[int, ...], int]], rho: int
) -> tuple[np.ndarray, list[tuple[tuple[int, ...], int]]]:
    equations = [coefficient for coefficient, bound in constraints if bound == 0]
    kernel = integer_kernel(equations, rho)
    reduced: dict[tuple[int, ...], int] = {}
    for coefficient, bound in constraints:
        if bound == 0:
            continue
        row = tuple(map(int, np.asarray(coefficient, dtype=np.int64) @ kernel))
        if not any(row):
            continue
        divisor = 0
        for value in row:
            divisor = math.gcd(divisor, abs(value))
        if divisor > 1:
            row = tuple(value // divisor for value in row)
            bound //= divisor
        key = canonical(row)
        old = reduced.get(key)
        if old is None or bound < old:
            reduced[key] = bound
    return kernel, sorted(reduced.items())


def conservative_bounds(
    constraints: Sequence[tuple[tuple[int, ...], int]], dimension: int
) -> list[tuple[int, int]] | None:
    if dimension == 0:
        return []
    if not constraints:
        return None
    a_rows: list[tuple[int, ...]] = []
    rhs: list[int] = []
    for coefficient, bound in constraints:
        a_rows.extend((coefficient, tuple(-x for x in coefficient)))
        rhs.extend((bound, bound))
    a = np.asarray(a_rows, dtype=float)
    b = np.asarray(rhs, dtype=float)
    result: list[tuple[int, int]] = []
    for coordinate in range(dimension):
        objective = np.zeros(dimension)
        objective[coordinate] = 1
        low = linprog(objective, A_ub=a, b_ub=b, bounds=[(None, None)] * dimension, method="highs")
        high = linprog(-objective, A_ub=a, b_ub=b, bounds=[(None, None)] * dimension, method="highs")
        if not low.success or not high.success:
            return None
        result.append((math.floor(float(low.fun)) - 2, math.ceil(float(-high.fun)) + 2))
    return result


def feasible_displacements(
    constraints: Sequence[tuple[tuple[int, ...], int]], rho: int, max_box: int
) -> tuple[list[tuple[int, ...]] | None, dict]:
    kernel, reduced = reduced_problem(constraints, rho)
    dimension = kernel.shape[1]
    if dimension == 0:
        return [tuple(0 for _ in range(rho))], {"kernel_dimension": 0, "box": 1}
    bounds = conservative_bounds(reduced, dimension)
    if bounds is None:
        return None, {"kernel_dimension": dimension, "reason": "unbounded"}
    box = math.prod(high - low + 1 for low, high in bounds)
    if box > max_box:
        return None, {"kernel_dimension": dimension, "box": box, "bounds": bounds}
    answer: list[tuple[int, ...]] = []
    for parameter in itertools.product(*(range(low, high + 1) for low, high in bounds)):
        if not all(
            abs(sum(coefficient[i] * parameter[i] for i in range(dimension))) <= bound
            for coefficient, bound in reduced
        ):
            continue
        vector = kernel @ np.asarray(parameter, dtype=np.int64)
        z = tuple(map(int, vector))
        if not all(
            abs(sum(coefficient[i] * z[i] for i in range(rho))) <= bound
            for coefficient, bound in constraints
        ):
            raise AssertionError("reduced displacement failed original constraints")
        answer.append(z)
    return answer, {
        "kernel_dimension": dimension,
        "box": box,
        "bounds": bounds,
        "feasible_count": len(answer),
    }


def signatures(
    rays: Sequence[Sequence[int]], ref: Sequence[int], free: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    ray_array = np.asarray(rays, dtype=np.int64)
    inverse = inverse_unimodular(ray_array[list(ref)].tolist())
    points = CUBE @ inverse.T
    signature = points @ ray_array[list(free)].T
    return np.unique(signature, axis=0), points


def exact_nonneat(
    rays: Sequence[Sequence[int]], cones: Sequence[Sequence[int]], max_box: int
) -> tuple[dict | None, dict]:
    standard = [tuple(1 if i == j else 0 for i in range(DIM)) for j in range(DIM)]
    index = {tuple(ray): i for i, ray in enumerate(rays)}
    proposed = tuple(index[ray] for ray in standard) if all(ray in index for ray in standard) else ()
    cone_set = set(map(tuple, cones))
    ref = proposed if proposed in cone_set else tuple(cones[0])
    free, constraints = support_constraints(rays, cones, ref)
    feasible, enumeration = feasible_displacements(constraints, len(free), max_box)
    metadata = {
        "reference_cone": list(ref),
        "free_facets": free,
        "constraint_count": len(constraints),
        **enumeration,
    }
    if feasible is None:
        return None, {**metadata, "hard": True}
    signature, points = signatures(rays, ref, free)
    tree = cKDTree(signature)
    for z in feasible:
        distance = float(tree.query(np.asarray(z, dtype=float), k=1, p=np.inf)[0])
        if distance <= 1.0 + 1e-9:
            continue
        b = [0] * len(rays)
        for facet, value in zip(free, z):
            b[facet] = int(value)
        # Independent exact exhaustion of all 3^9 candidates.
        ray_array = np.asarray(rays, dtype=np.int64)
        residual = points @ ray_array.T - np.asarray(b, dtype=np.int64)
        if np.any(np.all(np.abs(residual) <= 1, axis=1)):
            raise AssertionError("KD-tree reported a false non-neat witness")
        # Build the segment bundle P in dimension ten.
        p_normals = [[1] + [0] * DIM, [-1] + [0] * DIM]
        p_normals.extend([[-b_i] + list(map(int, ray)) for b_i, ray in zip(b, rays)])
        e_q = points[np.all(np.abs(points @ ray_array.T) <= 1, axis=1)]
        e_p = [[0] + list(map(int, point)) for point in e_q]
        return {
            "dimension_Q": DIM,
            "dimension_counterexample_P": DIM + 1,
            "primitive_facet_normals_Q": [list(map(int, ray)) for ray in rays],
            "normalized_displacement_b": b,
            "primitive_facet_normals_P": p_normals,
            "ewald_points_P": e_p,
            "ewald_point_count_P": len(e_p),
            "ewald_rank_P": DIM,
            "bounding_certificate": "reference-cone coordinates of every candidate x lie in {-1,0,1}^9; bundle coordinate t lies in {-1,0,1}",
            "normal_fan_constraints": [
                {"coefficient": list(coefficient), "bound": bound}
                for coefficient, bound in constraints
            ],
            "reference_cone": list(ref),
            "free_facets": free,
        }, metadata
    return None, metadata


def selected(ordinal: int, facets: int, sample_count: int, stride: int) -> bool:
    if stride > 0:
        return (ordinal - 1) % stride == (facets * 7919) % stride
    total = EXPECTED[facets]
    if sample_count <= 0 or sample_count >= total:
        return True
    step = total / sample_count
    return int((ordinal - 1) / step) != int(ordinal / step)


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--facets", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--max-box", type=int, default=2_000_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    scanned = sampled = hard = nontrivial = 0
    maximum_feasible = 0
    kernel_histogram: dict[str, int] = {}
    certificate = hashlib.sha256()
    for ordinal, rays in enumerate(records(args.input, args.facets), 1):
        scanned += 1
        if not selected(ordinal, args.facets, args.sample_count, args.stride):
            continue
        sampled += 1
        try:
            cones = maximal_cones(rays)
            witness, metadata = exact_nonneat(rays, cones, args.max_box)
        except (ValueError, QhullError) as error:
            save(args.output / "ERROR.json", {
                "source": args.input.name, "ordinal": ordinal, "error": repr(error),
                "primitive_facet_normals_Q": rays,
            })
            raise
        kernel_dimension = int(metadata.get("kernel_dimension", -1))
        kernel_histogram[str(kernel_dimension)] = kernel_histogram.get(str(kernel_dimension), 0) + 1
        feasible_count = int(metadata.get("feasible_count", 0))
        maximum_feasible = max(maximum_feasible, feasible_count)
        if feasible_count > 1:
            nontrivial += 1
        if metadata.get("hard"):
            hard += 1
            save(args.output / f"HARD-{ordinal}.json", {
                "source": args.input.name,
                "ordinal": ordinal,
                "metadata": metadata,
                "primitive_facet_normals_Q": rays,
            })
        if witness is not None:
            witness.update({"source": args.input.name, "ordinal": ordinal, "metadata": metadata})
            save(args.output / "COUNTEREXAMPLE.json", witness)
            print(json.dumps({
                "found": True, "facets": args.facets, "ordinal": ordinal,
                "ewald_point_count_P": witness["ewald_point_count_P"],
            }), flush=True)
            return 0
        if sampled % 250 == 0:
            print(json.dumps({
                "facets": args.facets, "sampled": sampled, "ordinal": ordinal,
                "nontrivial": nontrivial, "hard": hard,
                "max_feasible": maximum_feasible,
                "elapsed_sec": round(time.time() - started, 1),
            }), flush=True)
        certificate.update(ordinal.to_bytes(8, "little"))
        certificate.update(str(metadata).encode())

    summary = {
        "dimension": DIM,
        "source": args.input.name,
        "facets": args.facets,
        "records_scanned": scanned,
        "records_sampled": sampled,
        "sample_count_requested": args.sample_count,
        "stride": args.stride,
        "hard_count": hard,
        "nontrivial_displacement_count": nontrivial,
        "maximum_feasible_displacements": maximum_feasible,
        "kernel_dimension_histogram": kernel_histogram,
        "counterexample_found": False,
        "elapsed_sec": round(time.time() - started, 3),
        "certificate_sha256": certificate.hexdigest(),
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
