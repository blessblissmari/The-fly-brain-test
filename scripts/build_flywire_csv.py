#!/usr/bin/env python3
"""Convert the FlyWire 783 connectivity feather (Zenodo 10676866)
into the ``neurons.csv`` + ``connections.csv`` pair the Rust loader
(``flybrain.graph.load_zenodo``) consumes.

Inputs (downloaded once):

* ``data/flybrain/raw/proofread_connections_783.feather`` — 16.8 M
  rows ``(pre_root_id, post_root_id, neuropil, syn_count, *_avg)``.
* ``data/flybrain/raw/proofread_root_ids_783.npy`` — 139 255 root IDs
  that survived FlyWire proofreading.

Outputs:

* ``data/flybrain/raw/neurons.csv`` — columns ``id, cell_type,
  region``. ``cell_type`` is the dominant neurotransmitter
  (``ACH``/``GABA``/``GLUT``) inferred from per-connection
  ``*_avg`` columns (Eckstein et al., 2024 prediction). ``region``
  is the neuropil where the neuron sends most of its output.
* ``data/flybrain/raw/connections.csv`` — columns ``pre_root_id,
  post_root_id, syn_count, is_excitatory``. Multiple synapse-level
  rows for the same neuron pair are aggregated by summing
  ``syn_count``. ``is_excitatory`` is ``ach_avg > gaba_avg``.

Run::

    python scripts/build_flywire_csv.py

This script is idempotent and exits early if the output CSVs already
exist. Override with ``--force``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.feather as ft

DEFAULT_RAW = Path("data/flybrain/raw")
NEUROTRANSMITTERS = ("gaba", "ach", "glut", "oct", "ser", "da")
EXCITATORY = {"ach", "glut", "oct", "da"}


def _argmax_per_neuron(
    pairs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[int, str]:
    """Return ``neuron_id -> key-with-max-cumulative-weight``.

    Inputs are equal-length arrays ``(neuron, key, weight)``. The key
    chosen for each neuron is the one whose ``sum(weight)`` over all
    rows ``(neuron, key, w)`` is largest — i.e. the dominant
    neuropil / neurotransmitter by total synapse count, not the
    single highest-weight row. Ties broken by lexicographic key.
    """
    import pandas as pd

    df = pd.DataFrame({"neuron": pairs[0], "key": pairs[1], "weight": pairs[2]})
    grouped = (
        df.groupby(["neuron", "key"], sort=False, as_index=False)["weight"]
        .sum()
        .sort_values(["neuron", "weight", "key"], ascending=[True, False, True])
    )
    first = grouped.drop_duplicates(subset=["neuron"], keep="first")
    return dict(zip(first["neuron"].tolist(), first["key"].tolist(), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw = args.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    feather = raw / "proofread_connections_783.feather"
    root_ids_npy = raw / "proofread_root_ids_783.npy"
    neurons_csv = raw / "neurons.csv"
    conn_csv = raw / "connections.csv"

    if not args.force and neurons_csv.exists() and conn_csv.exists():
        print(f"[skip] {neurons_csv} and {conn_csv} already exist (use --force).")
        return

    if not feather.exists() or not root_ids_npy.exists():
        raise SystemExit(
            f"Missing FlyWire 783 inputs in {raw}. Download with:\n"
            "  curl -L -o data/flybrain/raw/proofread_connections_783.feather \\\n"
            "    https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content\n"
            "  curl -L -o data/flybrain/raw/proofread_root_ids_783.npy \\\n"
            "    https://zenodo.org/api/records/10676866/files/proofread_root_ids_783.npy/content"
        )

    print(f"[load] {feather}")
    table = ft.read_table(
        feather,
        columns=[
            "pre_pt_root_id",
            "post_pt_root_id",
            "neuropil",
            "syn_count",
            "gaba_avg",
            "ach_avg",
            "glut_avg",
            "oct_avg",
            "ser_avg",
            "da_avg",
        ],
    )
    print(f"[load] rows={table.num_rows:,}")

    proofread_ids = np.load(root_ids_npy)
    proofread_set = set(int(x) for x in proofread_ids)
    print(f"[load] proofread root_ids: {len(proofread_set):,}")

    pre = np.asarray(table["pre_pt_root_id"], dtype=np.int64)
    post = np.asarray(table["post_pt_root_id"], dtype=np.int64)
    neuropil = np.asarray(table["neuropil"].cast("string").to_pylist(), dtype=object)
    syn = np.asarray(table["syn_count"], dtype=np.int64)
    nt_stack = np.stack(
        [np.asarray(table[f"{nt}_avg"], dtype=np.float32) for nt in NEUROTRANSMITTERS],
        axis=1,
    )

    # ---- region per neuron (dominant outgoing neuropil) ----------------------
    print("[region] aggregating dominant outgoing neuropil per neuron...")
    region_lut = _argmax_per_neuron((pre, neuropil, syn))
    # Fill missing pre-only neurons (post-only) with their dominant incoming neuropil.
    print("[region] aggregating dominant incoming neuropil for post-only neurons...")
    incoming_lut = _argmax_per_neuron((post, neuropil, syn))
    for nid, region in incoming_lut.items():
        region_lut.setdefault(nid, region)

    # ---- dominant cell_type per neuron (argmax neurotransmitter) -------------
    print("[type] aggregating dominant neurotransmitter per neuron...")
    nt_argmax_idx = nt_stack.argmax(axis=1).astype(np.int8)
    nt_strings = np.array([nt.upper() for nt in NEUROTRANSMITTERS])
    nt_per_row = nt_strings[nt_argmax_idx]
    type_lut = _argmax_per_neuron((pre, nt_per_row, syn))
    type_lut_post = _argmax_per_neuron((post, nt_per_row, syn))
    for nid, t in type_lut_post.items():
        type_lut.setdefault(nid, t)

    # ---- write neurons.csv ---------------------------------------------------
    print(f"[write] {neurons_csv}")
    sorted_ids = sorted(proofread_set)
    with neurons_csv.open("w", encoding="utf-8") as fh:
        fh.write("id,cell_type,region\n")
        for nid in sorted_ids:
            region = region_lut.get(nid, "")
            ctype = type_lut.get(nid, "")
            fh.write(f"{nid},{ctype},{region}\n")
    print(f"[write] {neurons_csv}: {len(sorted_ids):,} rows")

    # ---- aggregate connections by (pre, post) --------------------------------
    print("[conn] aggregating syn_count + is_excitatory per (pre, post) ...")
    is_excit_lookup = np.array(
        [1 if NEUROTRANSMITTERS[i] in EXCITATORY else 0 for i in range(len(NEUROTRANSMITTERS))],
        dtype=np.int8,
    )
    excit_per_row = is_excit_lookup[nt_argmax_idx]

    import pandas as pd

    df = pd.DataFrame(
        {
            "pre": pre,
            "post": post,
            "syn": syn,
            "excit": excit_per_row.astype(np.int8),
        }
    )
    # Drop rows whose endpoints aren't in the proofread set.
    print("[conn] filtering to proofread endpoints...")
    proofread_arr = np.fromiter(proofread_set, dtype=np.int64)
    df = df[df["pre"].isin(proofread_arr) & df["post"].isin(proofread_arr)]
    print(f"[conn] retained rows: {len(df):,}")

    # is_excitatory voted per (pre,post) by majority weighted by syn
    df["w_excit"] = df["syn"] * df["excit"]
    print("[conn] groupby pre,post sum...")
    grouped = df.groupby(["pre", "post"], sort=False, as_index=False).agg(
        syn_count=("syn", "sum"),
        w_excit=("w_excit", "sum"),
    )
    # Rust CSV deserializer expects "true"/"false" literals, not 0/1.
    grouped["is_excitatory"] = (grouped["w_excit"] * 2 >= grouped["syn_count"]).map(
        {True: "true", False: "false"}
    )
    print(f"[conn] unique pairs: {len(grouped):,}")
    print(f"[write] {conn_csv}")
    grouped[["pre", "post", "syn_count", "is_excitatory"]].rename(
        columns={"pre": "pre_root_id", "post": "post_root_id"}
    ).to_csv(conn_csv, index=False)
    print(f"[write] {conn_csv}: {len(grouped):,} rows")
    print("[done]")


if __name__ == "__main__":
    main()
