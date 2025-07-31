#!/usr/bin/env python3
"""
convert_qmcl_to_anids.py - TFRecord to ANI-HDF5 converter

This script converts a subshard from the QMCL dataset format (stored in TFRecord files)
into a TorchANI-compatible HDF5 format following the ANIDataset structure. It is intended
for use in workflows involving machine learning potentials, such as ANI, and supports 
per-atom and per-conformer properties as well as molecular metadata.

Key Features:
- Reads TFRecord files containing atomic properties, energies, forces, and metadata.
- Performs unit conversions (e.g., coordinates and forces from Bohr to Ångstrom).
- Filters and groups data by number of atoms.
- Outputs the converted data into an `.h5` file for use with TorchANI models.
- Supports parallel ingestion of multiple features and automatic batching via CHUNKSIZE.
- It keeps the hash of each record so we later check externally that both datasets are consistent.

Since it is intented to be used in a HPC environment, it provides a CLI interface so it can be called as:

python convert_qmcl_to_anids.py \
    --TFDRPATH /path/to/qmcl/subshards \
    --SHARD 00003-of-00011 \ 
    --SUBSHARD 002 \
    --ANIPATH /path/to/anids/output \
    --CHUNKSIZE 1000


The output is an .h5 file named aniqmcl-<SHARD>-<SUBSHARD>.h5 created in the ANIPATH directory, storing grouped and
converted molecular data compatible with TorchANI.

Notes:
------
SHARD and SUBSHARD follow the original QMCL naming convention and reflect the structure
of the TFRecord dataset (split into shards and subshards for preprocessing and distribution).

"""

import os
import sys
import time
import argparse
from pathlib import Path
from functools import partial

import numpy as np
import tensorflow as tf
from torchani.datasets import ANIDataset

sys.stdout.reconfigure(line_buffering=True)
#--------------------------- DICTS 

# each entry:  feature_name : (subdir-prefix,  tf.io.*Feature  instance, feature_name_in_anids)
QMCL_TO_ANI = {
    # --------  per-atom arrays ------------------------------------------
    "atomic_numbers":          ("dft",     tf.io.VarLenFeature(tf.int64),   "species"),
    "positions":               ("dft",     tf.io.VarLenFeature(tf.float32), "coordinates"),
    "pbe0_forces":             ("dft",     tf.io.VarLenFeature(tf.float32), "forces"),
    "pbe0_hirshfeld_charges":  ("dft",     tf.io.VarLenFeature(tf.float32), "hirshfeld_atomic_charges"),
    "pbe0_mulliken_charges":   ("dft",     tf.io.VarLenFeature(tf.float32), "mulliken_atomic_charges"),
    "pbe0_hirshfeld_volumes":  ("dft",     tf.io.VarLenFeature(tf.float32), "hirshfeld_atomic_volumes"),
    "d4_atomic_charges":       ("dft",  tf.io.VarLenFeature(tf.float32), "d4_atomic_charges"),
    "d4_polarizabilities":     ("dft",  tf.io.VarLenFeature(tf.float32), "d4_atomic_polarizabilities"),
    "mbd_polarizabilities":    ("dft", tf.io.VarLenFeature(tf.float32), "mbd_atomic_polarizabilities"),

    # --------  per-conformer scalars / vectors ---------------------------
    "pbe0_energy":             ("dft",     tf.io.FixedLenFeature([],  tf.float32), "energies"),
    "d4_energy":               ("dft",  tf.io.FixedLenFeature([],  tf.float32), "d4_energies"),
    "mbd_energy":              ("dft", tf.io.FixedLenFeature([],  tf.float32), "mbd_energies"),

    # 3-component vector → store as length-3 dense feature
    "pbe0_dipole":             ("dft",     tf.io.FixedLenFeature([3], tf.float32), "dipoles"),

    # 3*N forces for D4/MBD — still atom-length, so VarLen
    "d4_forces":               ("dft",  tf.io.VarLenFeature(tf.float32), "d4_forces"),
    "mbd_forces":              ("dft", tf.io.VarLenFeature(tf.float32), "mbd_forces"),

    # --------  boolean flag ---------------------------------------------
    "is_outlier":              ("dft", tf.io.FixedLenFeature([], tf.int64),     "is_outlier"),

}

# Metadata handling

METADATA_SPEC = {
    "key_hash"            : tf.io.FixedLenFeature([], tf.string),
    "smiles"              : tf.io.FixedLenFeature([], tf.string),
    "smiles_hash"         : tf.io.FixedLenFeature([], tf.string),
    "charge"              : tf.io.FixedLenFeature([], tf.int64),
    "multiplicity"        : tf.io.FixedLenFeature([], tf.int64),

    # optional / nice-to-haves
    "conformation_seq"       : tf.io.FixedLenFeature([], tf.int64),
    "conformation_parent_seq": tf.io.FixedLenFeature([], tf.int64),
    "molecular_weight"       : tf.io.FixedLenFeature([], tf.float32),
#   "chemical_formula"       : tf.io.FixedLenFeature([], tf.string), #It seems not to be present in the metadata....
    "num_atoms"              : tf.io.FixedLenFeature([], tf.int64),
    "num_heavy_atoms"        : tf.io.FixedLenFeature([], tf.int64),
    
}

#--------------------------- HELPERS

_KEY = ("key_hash", tf.io.FixedLenFeature([], tf.string))   # reused literal

def _parse(record, feature_name, feature_proto):
    spec = { _KEY[0]: _KEY[1], feature_name: feature_proto }
    return tf.io.parse_single_example(record, spec)

def parse_meta(record):
    return tf.io.parse_single_example(record, METADATA_SPEC)


def build_datasets(mapping: dict,
                   TFDRPATH: str,
                   SHARD: str,
                   SUBSHARD: str,
                   version: str = "1.0.0"):
    """Return a dict {feature_name: tf.data.Dataset} built in lock-step."""

    datasets = {}
    for feat, (folder, proto, _) in mapping.items():
        # ---------- construct the file name ----------
        path = (
            f"{TFDRPATH}/{SHARD}/{folder}_{feat}_{version}/"
            f"qcml-full.tfrecord-{SHARD}-{SUBSHARD}.tfrecord"
        )
        # ---------- build ds ----------
        ds = (
            tf.data.TFRecordDataset(path)
            .map(partial(_parse, feature_name=feat, feature_proto=proto))
        )
        datasets[feat] = ds

    return datasets

def new_bucket():
    core_keys   = { out: [] for *_, out in QMCL_TO_ANI.values() }
    extras_keys = {k: [] for k in METADATA_SPEC}          # same as .keys()
    extras_keys.update({"ANI2x_elements_only": []})
    return {**core_keys, **extras_keys}

#--------------------------- LOG
LOG_PATH = Path(f"progress_{os.getenv('SLURM_JOB_ID', 'local')}.log")

def log_progress(msg: str, path: Path = LOG_PATH):
    """Append *msg* to *path* and force it to disk immediately."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", buffering=1) as fh:      # line-buffered
        fh.write(f"{ts}  {msg}\n")
        fh.flush()
        os.fsync(fh.fileno())                     # guarantee it’s on disk

#--------------------------- ARG PARSER

def get_builder_args() -> argparse.Namespace:
    """Return CLI options for the dataset-builder workflow."""
    parser = argparse.ArgumentParser(
        description="Builder-config command-line options"
    )
    parser.add_argument(
        "-t", "--TFDRPATH",
        type=Path,
        default=Path("/home/jsemelak/2-ANI-QMCL/1-DS/1-SplitdDS/subshards/"),
        help="Root directory that contains the TFDR shards.",
    )
    parser.add_argument(
        "-sh", "--SHARD",
        default="00000-of-00011",
        help="Shard identifier (e.g. 00000-of-00011).",
    )
    parser.add_argument(
        "-su", "--SUBSHARD",
        default="000",
        help="Subshard identifier (e.g. 000).",
    )
    parser.add_argument(
        "-a", "--ANIPATH",
        default=Path("/home/jsemelak/2-ANI-QMCL/1-DS/2-ANIDS/subshards/"),
        help="Filename of the ANI dataset inside each shard.",
    )
    parser.add_argument(
        "-c", "--CHUNKSIZE",
        type=int,
        default=1000,
        metavar="N",
        help="Number of records to process per batch.",
    )
    return parser.parse_args()

# --------------------------------------------------------------------------- #
# Example usage                                                               #
# --------------------------------------------------------------------------- #
args = get_builder_args()

# builder-config file lists 
TFDRPATH = args.TFDRPATH
SHARD = args.SHARD
SUBSHARD = args.SUBSHARD
ANIPATH = args.ANIPATH
CHUNKSIZE = args.CHUNKSIZE

# Internal var
ani_ds_name = f"aniqmcl-{SHARD}-{SUBSHARD}.h5"

# Dicts set up
ds_dict = build_datasets(QMCL_TO_ANI, TFDRPATH=TFDRPATH, SHARD=SHARD, SUBSHARD=SUBSHARD)

meta_files = f"{TFDRPATH}/{SHARD}/dft_metadata_1.0.0/qcml-full.tfrecord-{SHARD}-{SUBSHARD}.tfrecord"
ds_meta = (
    tf.data.TFRecordDataset(meta_files, compression_type=None)  # set gz if gzipped
      .map(parse_meta, num_parallel_calls=tf.data.AUTOTUNE)
)

all_streams = {**ds_dict, "meta": ds_meta}   # every *value* is a dataset
merged      = tf.data.Dataset.zip(all_streams)

# Ingestion loop
BOHR_TO_ANGSTROM = 0.52918
ANI_ELEMS    = {1, 6, 7, 8, 9, 16, 17}

ani_ds = ANIDataset(locations=ani_ds_name, grouping="by_num_atoms")
buckets = {}

count = 0
# ------------------------------------------------------------------
print(f"Creating {ani_ds_name}...")
# ------------------------------------------------------------------
for packed in merged.as_numpy_iterator():   # eager → numpy scalars
    # ---------------- sanity check & filters ---------------------------
    meta=packed["meta"]
    kh = meta["key_hash"]
    assert all(packed[k]["key_hash"] == kh for k in ds_dict)

    # ---------- fetch atomic numbers & natoms ---------------------------
    ex_Z = packed["atomic_numbers"]
    Z    = ex_Z["atomic_numbers"]
    n = len(Z)

    bucket = buckets.setdefault(n, new_bucket())
    # ---------- convert & append every feature automatically ------------
    for feat, (folder, proto, out_key) in QMCL_TO_ANI.items():
        raw = packed[feat][feat]
        if isinstance(raw, tf.sparse.SparseTensor):
            raw = tf.sparse.to_dense(raw)
            raw = raw.numpy()

        # reshape / unit-convert by name pattern
        if out_key == "species":               val = raw
        elif out_key == "coordinates":         val = raw.reshape(-1,3)*BOHR_TO_ANGSTROM
        elif out_key.endswith("forces"):       val = raw.reshape(-1,3)/BOHR_TO_ANGSTROM
        elif out_key.endswith("energies"):     val = np.asarray([raw])
        elif out_key == "dipoles":             val = raw*BOHR_TO_ANGSTROM     # already (3,)
        else:                                  val = raw  # everything else

        bucket[out_key].append(val)

    # ---------- metadata extras -----------------------------------------
    for out_key, proto in METADATA_SPEC.items():
        raw = meta[out_key]
        if isinstance(raw, tf.sparse.SparseTensor):
            raw = tf.sparse.to_dense(raw)
            raw = raw.numpy()
        if proto == tf.io.FixedLenFeature([], tf.int64):
            val = int(raw)
        elif proto == tf.io.FixedLenFeature([], tf.float32):
            val = raw
        elif proto == tf.io.FixedLenFeature([], tf.string):
            val = raw.decode()
        else:
             print(f"Warning!, unrecognized data type for feat: {out_key}")

        bucket[out_key].append(val)

    bucket["ANI2x_elements_only"].append(int(set(Z).issubset(ANI_ELEMS)))
    # ---------- flush this natoms bucket if full ------------------------
    if len(bucket["species"]) == CHUNKSIZE:
       print(f"UPDATING BUCKET {n}")
       core = {}
       for feat, (folder, proto, out_key) in QMCL_TO_ANI.items():
           if proto == tf.io.VarLenFeature(tf.int64) or proto == tf.io.FixedLenFeature([], tf.int64):
               core.update({out_key: np.asarray(bucket[out_key], np.int32)})
           elif proto == tf.io.VarLenFeature(tf.float32) or \
               proto == tf.io.FixedLenFeature([],  tf.float32) or \
               proto == tf.io.FixedLenFeature([3], tf.float32):
               core.update({out_key: np.asarray(bucket[out_key], np.float64)})
           else:
               print(f"Warning!, unrecognized data type for feat: {feat}")

       extra = {}
       for out_key, proto in METADATA_SPEC.items():
           if proto == tf.io.FixedLenFeature([], tf.int64):
               extra.update({out_key: np.asarray(bucket[out_key], np.int32)})
           elif proto == tf.io.FixedLenFeature([], tf.float32):
               extra.update({out_key: np.asarray(bucket[out_key], np.float64)})
           elif proto == tf.io.FixedLenFeature([], tf.string):
               extra.update({out_key: np.array(bucket[out_key])})
           
           else:
               print(f"Warning!, unrecognized data type for feat: {out_key}")

       extra.update({"ANI2x_elements_only": np.asarray(bucket["ANI2x_elements_only"], np.int32)})

       core.update(extra)

       ani_ds.auto_append_conformers(core)
       count += CHUNKSIZE
       log_progress(f"Chunk of {CHUNKSIZE} pushed to group of natoms = {n} | total={count:,}")

       for k in bucket: bucket[k].clear()

for n, bucket in buckets.items():
    log_progress(f"Flushing bucket of natoms = {n}")
    if bucket["species"]:
        core = {}
        for feat, (folder, proto, out_key) in QMCL_TO_ANI.items():
            if proto == tf.io.VarLenFeature(tf.int64) or proto == tf.io.FixedLenFeature([], tf.int64):
                core.update({out_key: np.asarray(bucket[out_key], np.int32)})
            elif proto == tf.io.VarLenFeature(tf.float32) or \
                 proto == tf.io.FixedLenFeature([],  tf.float32) or \
                 proto == tf.io.FixedLenFeature([3], tf.float32):
                core.update({out_key: np.asarray(bucket[out_key], np.float64)})
            else:
                print(f"Warning!, unrecognized data type for feat: {feat}")

        extra = {}
        for out_key, proto in METADATA_SPEC.items():
            if proto == tf.io.FixedLenFeature([], tf.int64):
                extra.update({out_key: np.asarray(bucket[out_key], np.int32)})
            elif proto == tf.io.FixedLenFeature([], tf.float32):
                extra.update({out_key: np.asarray(bucket[out_key], np.float64)})
            elif proto == tf.io.FixedLenFeature([], tf.string):
                extra.update({out_key: np.array(bucket[out_key])})
            else:
                print(f"Warning!, unrecognized data type for feat: {out_key}")
        extra.update({"ANI2x_elements_only": np.asarray(bucket["ANI2x_elements_only"], np.int32)})
        core.update(extra)
        ani_ds.auto_append_conformers(core)
        count += len(bucket["species"])

log_progress(f"Done – total {count:,} conformers added")

