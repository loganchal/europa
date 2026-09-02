#!/usr/bin/env python3
"""Check prefiltered dimension-9 neatness candidates against the full type cone.

The prefilter emits a displacement with no ternary middle point.  This program
reconstructs every vertex from the primitive facet system, validates lattice
integrality and smoothness, derives the exact integer inequalities for support
vectors whose positive and negative displacements have the original normal
fan, and then tests every such support vector in its finite coordinate-vertex
box.  Any surviving no-lift vector is a dimension-10 ordinary-Ewald
counterexample via the bundle over [-1,1].
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import HalfspaceIntersection

DIM = 9
TERNARY = np.asarray(list(itertools.product((-1, 0, 1), repeat=DIM)), dtype=np.int16)
NEG_E = [tuple(-1 if i == j else 0 for j in range(DIM)) for i in range(DIM)]


def vertices_from_facets(a: np.ndarray) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    # polyDB convention: 1 + a_i.x >= 0.  Qhull expects Hx+b <= 0.
    halfspaces = np.c_[-a.astype(float), -np.ones(len(a))]
    hull = HalfspaceIntersection(halfspaces, np.zeros(a.shape[1]))
    vertices: dict[tuple[int, ...], tuple[int, ...]] = {}
    for approximate in hull.intersections:
        rounded = np.rint(approximate).astype(np.int64)
        if float(np.max(np.abs(approximate - rounded))) > 1e-6:
            raise AssertionError(f"nonintegral reconstructed vertex: {approximate}")
        slacks = 1 + a @ rounded
        if not np.all(slacks >= 0):
            raise AssertionError("rounded vertex is infeasible")
        active = tuple(int(i) for i in np.flatnonzero(slacks == 0))
        vertices[tuple(int(x) for x in rounded)] = active
    if not vertices:
        raise AssertionError("no vertices reconstructed")
    return list(vertices.items())


def type_cone_data(record: dict) -> tuple[np.ndarray, list[int], list[int], list[tuple[tuple[int, ...], int]], int]:
    a = np.asarray([row[1:] for row in record["facets"]], dtype=np.int64)
    m, n = a.shape
    if n != DIM:
        raise AssertionError(f"wrong dimension for {record['id']}")
    normal_to_index = {tuple(int(x) for x in row): i for i, row in enumerate(a)}
    coordinate = [normal_to_index[u] for u in NEG_E]
    extra = [i for i in range(m) if i not in coordinate]
    q = np.ones(n, dtype=np.int64)
    bounds = [int(a[j] @ q) for j in extra]
    if any(b < 0 for b in bounds):
        raise AssertionError(f"negative coordinate-vertex bound at {record['id']}")

    vertices = vertices_from_facets(a)
    extra_position = {facet: k for k, facet in enumerate(extra)}
    identity = np.eye(n, dtype=np.int64)
    inequalities: dict[tuple[int, ...], int] = {}

    for vertex_tuple, active_tuple in vertices:
        active = list(active_tuple)
        if len(active) != n:
            raise AssertionError(f"polytope not simple at {record['id']} {vertex_tuple}")
        t_active = (-a[active, :]).astype(np.int64)
        inverse = np.rint(np.linalg.inv(t_active)).astype(np.int64)
        if not np.array_equal(t_active @ inverse, identity):
            raise AssertionError(f"nonunimodular vertex at {record['id']} {vertex_tuple}")

        selector = np.zeros((n, len(extra)), dtype=np.int64)
        for row, facet in enumerate(active):
            if facet in extra_position:
                selector[row, extra_position[facet]] = 1
        displacement_matrix = inverse @ selector
        vertex = np.asarray(vertex_tuple, dtype=np.int64)

        for facet in range(m):
            if facet in active:
                continue
            t = -a[facet]
            integral_slack = 1 - int(t @ vertex)
            if integral_slack < 1:
                raise AssertionError(f"bad strict slack at {record['id']}")
            coefficient = np.zeros(len(extra), dtype=np.int64)
            if facet in extra_position:
                coefficient[extra_position[facet]] = 1
            coefficient -= t @ displacement_matrix
            key = tuple(int(x) for x in coefficient)
            first = next((x for x in key if x), 0)
            if first < 0:
                key = tuple(-x for x in key)
            rhs = integral_slack - 1
            inequalities[key] = min(rhs, inequalities.get(key, 10**18))

    return a, extra, bounds, sorted(inequalities.items()), len(vertices)


def type_feasible(values: tuple[int, ...], inequalities: Iterable[tuple[tuple[int, ...], int]]) -> bool:
    for coefficient, rhs in inequalities:
        total = sum(c * x for c, x in zip(coefficient, values))
        if abs(total) > rhs:
            return False
    return True


def lift_count(t_extra_y: np.ndarray, values: tuple[int, ...]) -> int:
    support = np.asarray(values, dtype=np.int64)
    return int(np.count_nonzero(np.all(np.abs(t_extra_y + support) <= 1, axis=1)))


def analyze(record: dict, max_box: int) -> dict:
    a, extra, bounds, inequalities, vertex_count = type_cone_data(record)
    box_volume = math.prod(2 * b + 1 for b in bounds)
    t_extra_y = TERNARY @ (-a[extra, :]).T if extra else np.zeros((len(TERNARY), 0), dtype=np.int64)

    full_shift = [int(x) for x in record["normalized_facet_shifts"]]
    reported = tuple(full_shift[j] for j in extra)
    reported_lifts = lift_count(t_extra_y, reported)
    if reported_lifts != 0:
        raise AssertionError(f"prefilter no-lift witness is not no-lift at {record['id']}")
    reported_type_feasible = type_feasible(reported, inequalities)

    result = {
        "id": record["id"],
        "n_facets": len(a),
        "vertices": vertex_count,
        "bounds": bounds,
        "box_volume": box_volume,
        "type_inequalities": len(inequalities),
        "reported_no_lift_twist": list(reported),
        "reported_twist_type_feasible": reported_type_feasible,
        "exact_box_checked": box_volume <= max_box,
        "normal_fan_preserving_twists": 0,
        "minimum_lifts": None,
        "best_twist": None,
        "uncovered_type_twists": [],
        "facets": record["facets"],
    }
    if reported_type_feasible:
        result["uncovered_type_twists"].append(list(reported))
        result["minimum_lifts"] = 0
        result["best_twist"] = list(reported)
        return result

    if box_volume > max_box:
        return result

    minimum = len(TERNARY) + 1
    best: tuple[int, ...] | None = None
    feasible_count = 0
    uncovered: list[list[int]] = []
    domains = [range(-b, b + 1) for b in bounds]
    for values in itertools.product(*domains):
        if not type_feasible(values, inequalities):
            continue
        feasible_count += 1
        count = lift_count(t_extra_y, values)
        if count < minimum:
            minimum = count
            best = values
        if count == 0:
            uncovered.append(list(values))

    result["normal_fan_preserving_twists"] = feasible_count
    result["minimum_lifts"] = minimum
    result["best_twist"] = list(best) if best is not None else None
    result["uncovered_type_twists"] = uncovered
    return result


def worker(payload: tuple[dict, int]) -> dict:
    return analyze(payload[0], payload[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(4, mp.cpu_count())))
    parser.add_argument("--max-box", type=int, default=1_000_000)
    parser.add_argument("--retain", type=int, default=200)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    candidates = source["uncovered"]
    payloads = ((record, args.max_box) for record in candidates)
    results: list[dict] = []
    if args.workers == 1:
        iterator = map(worker, payloads)
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        iterator = pool.map(worker, payloads, chunksize=20)

    histogram: Counter[int] = Counter()
    counterexamples: list[dict] = []
    skipped: list[dict] = []
    checked = 0
    for checked, result in enumerate(iterator, start=1):
        if result["normal_fan_preserving_twists"]:
            histogram[result["normal_fan_preserving_twists"]] += 1
        if result["uncovered_type_twists"]:
            counterexamples.append(result)
        if not result["exact_box_checked"]:
            skipped.append(result)
        results.append(result)
        if checked % 1000 == 0:
            print(
                f"checked={checked}/{len(candidates)} hits={len(counterexamples)} "
                f"large_boxes={len(skipped)}",
                flush=True,
            )
    if args.workers != 1:
        pool.shutdown()

    near = [r for r in results if r["minimum_lifts"] is not None]
    near.sort(key=lambda r: (r["minimum_lifts"], -r["box_volume"], r["id"]))
    output = {
        "dimension": DIM,
        "source_shard": source.get("shard"),
        "prefilter_candidates": len(candidates),
        "checked": checked if candidates else 0,
        "counterexample_fiber_count": len(counterexamples),
        "counterexample_fibers": counterexamples,
        "large_box_unresolved_count": len(skipped),
        "large_box_unresolved": skipped,
        "normal_fan_twist_count_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "lowest_lift_candidates": near[: args.retain],
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"complete checked={output['checked']} hits={len(counterexamples)} "
        f"large_boxes={len(skipped)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
