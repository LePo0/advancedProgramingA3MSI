#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brute-force weighted search on a graph of nodes with 50-dimensional feature vectors.

Usage
-----
python brute_force_search.py <points.csv> <queries.csv> [output.csv]

- <points.csv>  : CSV contenant au minimum les colonnes 'node_id' et 'feature_1'..'feature_50' (autres colonnes ignorées).
- <queries.csv> : CSV contenant au minimum les colonnes 'point_A', 'Y_vector', 'D'.
                  'Y_vector' est une chaîne de 50 poids séparés par ';'.
- [output.csv]  : (optionnel) fichier de sortie, par défaut 'responses.csv'.

Méthode
-------
Pour chaque requête q=(point_A, Y, D), on génère un vecteur A (50 dim) déterministe à partir de 'point_A'
(supposé ne pas appartenir au graphe), puis on calcule la distance euclidienne pondérée:

    dist(A, P) = sqrt( sum_i ( Y_i * (A_i - P_i)^2 ) )

On retourne tous les noeuds P tels que dist(A,P) <= D. Implémentation 100% brute-force (pas d'optimisations).

Remarque
--------
Afin de rendre la génération de A **reproductible**, on utilise un seed dérivé de 'point_A'.
Par défaut, A est généré en [0,100] (réels) pour chaque dimension.
"""

from __future__ import annotations
import sys
import math
import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd

NUM_FEATURES = 50

def parse_weights(y_str: str) -> np.ndarray:
    """Parse a 'Y_vector' string like '0.1;0.2;...' into a numpy array of shape (50,)."""
    parts = [p.strip() for p in str(y_str).split(';') if p.strip() != '']
    weights = np.array([float(x) for x in parts], dtype=float)
    if weights.shape[0] != NUM_FEATURES:
        raise ValueError(f"Y_vector length {weights.shape[0]} != {NUM_FEATURES}")
    return weights

def generate_A(point_A: str) -> np.ndarray:
    """Generate a deterministic 50-dim vector A in [0,100] seeded by point_A.

    This mirrors the idea that A (e.g., an ad or user profile) is not part of the graph but exists in the same feature space.
    """
    # Derive a 64-bit integer seed from the query id
    h = hashlib.sha256(point_A.encode('utf-8')).hexdigest()
    seed = int(h[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)

    # Option 1 (simple & uniform):
    A = rng.uniform(0.0, 100.0, size=NUM_FEATURES).astype(float)

    # If you'd like a "sparse peaks" version similar to the draft, uncomment below:
    # A = np.zeros(NUM_FEATURES, dtype=float)
    # k_hi = rng.integers(low=1, high=4)
    # k_md = rng.integers(low=4, high=10)
    # hi_idx = rng.choice(NUM_FEATURES, size=k_hi, replace=False)
    # md_pool = [i for i in range(NUM_FEATURES) if i not in hi_idx]
    # md_idx = rng.choice(md_pool, size=k_md, replace=False)
    # A[hi_idx] = rng.uniform(80, 100, size=k_hi)
    # A[md_idx] = rng.uniform(40, 70, size=k_md)

    return A

def extract_point_vector(row: pd.Series, feature_cols: List[str]) -> np.ndarray:
    return row[feature_cols].to_numpy(dtype=float)

def weighted_euclidean(a: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    # Ensure all arrays are 1D of length NUM_FEATURES
    diff = a - p
    return float(np.sqrt(np.sum(w * diff * diff)))

def brute_force_search(points_df: pd.DataFrame, queries_df: pd.DataFrame) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
    # Identify the 50 feature columns (tolerant to extra columns like 'cluster_id')
    feature_cols = [f'feature_{i+1}' for i in range(NUM_FEATURES)]
    for col in feature_cols:
        if col not in points_df.columns:
            raise KeyError(f"Missing column '{col}' in points file")

    results_per_query: List[Tuple[str, float, List[Tuple[str, float]]]] = []

    # Pre-materialize points to numpy for speed (still brute-force)
    node_ids = points_df['node_id'].astype(str).to_list()
    points_mat = points_df[feature_cols].to_numpy(dtype=float)

    for _, q in queries_df.iterrows():
        q_id = str(q['point_A'])
        D = float(q['D'])
        Y = parse_weights(q['Y_vector'])
        A = generate_A(q_id)

        # Compute distances to all nodes
        # Broadcasting: (N,50) - (50,) -> (N,50)
        diff = points_mat - A
        dists = np.sqrt(np.sum(Y * diff * diff, axis=1))

        # Collect matches within radius D
        matches: List[Tuple[str, float]] = [
            (node_ids[i], float(dists[i])) for i in range(len(node_ids)) if dists[i] <= D
        ]

        # Sort matches by distance asc, then node_id for determinism
        matches.sort(key=lambda x: (x[1], x[0]))

        results_per_query.append((q_id, D, matches))

    return results_per_query

def write_response_csv(results: List[Tuple[str, float, List[Tuple[str, float]]]], output_path: str) -> None:
    rows = []
    for q_id, D, matches in results:
        node_list = ';'.join(n for n, _ in matches)
        node_with_dist_list = ';'.join(f"{n}:{d:.6f}" for n, d in matches)
        rows.append({
            'query_id': q_id,
            'D': D,
            'num_matches': len(matches),
            'nodes': node_list,
            'nodes_with_distance': node_with_dist_list,
        })
    out_df = pd.DataFrame(rows, columns=['query_id', 'D', 'num_matches', 'nodes', 'nodes_with_distance'])
    out_df.to_csv(output_path, index=False)

def main(argv: List[str]) -> None:
    if len(argv) < 3 or len(argv) > 4:
        print("Usage : python brute_force_search.py <points.csv> <queries.csv> [output.csv]")
        sys.exit(1)

    points_file = argv[1]
    queries_file = argv[2]
    output_file = argv[3] if len(argv) == 4 else 'responses.csv'

    # Read inputs
    points_df = pd.read_csv(points_file)
    queries_df = pd.read_csv(queries_file)

    # Sanity checks
    if 'node_id' not in points_df.columns:
        raise KeyError("Points file must contain 'node_id' column")
    for col in ('point_A', 'Y_vector', 'D'):
        if col not in queries_df.columns:
            raise KeyError(f"Queries file must contain '{col}' column")

    # Compute brute-force results
    results = brute_force_search(points_df, queries_df)

    # Write output
    write_response_csv(results, output_file)
    print(f"✅ Fichier de réponse généré : {output_file}")

if __name__ == '__main__':
    main(sys.argv)
