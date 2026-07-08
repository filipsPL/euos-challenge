#!/usr/bin/env python3
"""
benchmark_structural_alerts.py

-> to be run in batch for all targets

Benchmark published structural-alert filters as baseline classifiers for the
EUOS25 optical interference endpoints (fluorescence / transmittance), to
satisfy the "comparison against a published method" requirement.

Filters used
------------
1. PAINS (Baell & Holloway, J. Med. Chem. 2010)
   RDKit built-in FilterCatalog (PAINS_A + PAINS_B + PAINS_C combined via PAINS).
   Directly relevant: PAINS was explicitly designed to flag, among other things,
   compounds prone to optical/fluorescence assay interference.

2. REOS-proxy (Walters & Murcko, REOS, 2002)
   RDKit has no built-in REOS catalog, so we use RDKit's ZINC "unwanted
   functionality" filter set as a published, drug-likeness/reactive-group proxy
   in the same spirit as REOS (property + reactive-group based pan-assay
   filtering). This is flagged explicitly in the output -- it is an
   approximation, not a literal re-implementation of the original REOS SMARTS,
   which were never made fully public.

3. Negative control: BRENK (Brenk et al., ChemMedChem 2008)
   Flags reactive, toxic, and metabolically unstable substructures (e.g.
   Michael acceptors, aldehydes, alkyl halides). Mechanistically unrelated to
   optical/fluorescence interference -- included as a sanity-check baseline.
   If BRENK "predicts" our optical endpoints as well as PAINS/REOS do, that
   would indicate the apparent PAINS/REOS signal is just a generic
   "weird/reactive molecule" artifact rather than a real optical-interference
   signal, so this needs to be reported alongside the real baselines.

4. PAINS_chromophore_subset / PAINS_remainder
   PAINS bundles ~480 SMARTS rules built mainly for generic reactive/promiscuous
   binder triage; most have no connection to light absorption or emission. A
   subset, however, corresponds to chemotypes with independent literature
   grounding as colorants/fluorophores: azo compounds, quinones, dyes,
   hydrazones, aminonaphthalenes, acridines, tetrazines, and styrene-type
   extended-conjugation systems (see CHROMOPHORE_RELEVANT_PAINS_FAMILIES
   below for the exact rule-name prefixes and per-family rationale). We split
   the full PAINS rule set into this "chromophore-relevant" subset and the
   remaining ~445 "other" PAINS rules, and score each subset separately. This
   lets us test whether PAINS' (lack of) signal is uniform across the whole
   rule set, or concentrated in/absent from the chemotypes that are actually
   plausible for optical interference.

Usage
-----
    python benchmark_structural_alerts.py \\
        --data path/to/activity_file.csv.gz \\
        --smiles-col SMILES \\
        --label-col activity \\
        --target-name fluorescence_340_450 \\
        --out results/structural_alerts_fluorescence_340_450.csv

The input file must contain a SMILES column and a binary 0/1 label column
(same files used upstream of join_activity_and_descriptors.py, i.e. before
SMILES is dropped). Run once per target/subtask and then concatenate the
per-target CSVs (see combine step at the bottom of this docstring / use
`--combine-out` to do it automatically across multiple --data args).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

RDLogger.DisableLog("rdApp.*")

# PAINS rule-name prefixes selected for plausible chromophore/fluorophore
# relevance, each with independent literature grounding as a colorant or
# fluorophore class (as opposed to a reactive/promiscuous-binder motif).
# Matching is by exact prefix against the RDKit PAINS entry description,
# e.g. "quinone_A(370)" matches prefix "quinone_".
CHROMOPHORE_RELEVANT_PAINS_FAMILIES = {
    "dyes":           "Curated dye-like extended-conjugation scaffolds.",
    "azo_":           "Azo (-N=N-) chromophores; the largest class of synthetic colorants.",
    "quinone_":       "Quinones; classic chromophoric/redox-colored species, common in "
                       "absorbance-assay interference.",
    "naphth_amino_":  "Aminonaphthalenes; azo-dye precursors and common fluorophore cores.",
    "amino_acridine_": "Acridines; textbook fluorophore scaffold (e.g. acridine orange).",
    "het_6_tetrazine": "Tetrazines; intensely colored conjugated heterocycles.",
    "styrene":        "Stilbene-like extended-conjugation systems; common fluorophore motif "
                       "(includes styrene_A/B/C, styrene_anil_A, styrene_imidazole_A).",
    "hzone_":         "Hydrazones (conjugated C=N-N=C); historically used as colorimetric/"
                       "chromogenic reagents.",
}

# Explicitly excluded despite superficial keyword overlap, since these PAINS
# families describe reactivity/promiscuity motifs rather than chromophores:
# rhod_* (rhodanine: reactive covalent/promiscuous-binder warhead), cyano_*,
# sulfonamide_*, mannich_*, anil_di_alk_* (generic reactive anilines).

FILTER_DEFINITIONS = {
    "PAINS": {
        "catalogs": [FilterCatalogParams.FilterCatalogs.PAINS],
        "description": "Baell & Holloway 2010 -- pan-assay interference compounds, "
                        "includes known optical/fluorescence-interference chemotypes.",
    },
    "REOS_proxy_ZINC": {
        "catalogs": [FilterCatalogParams.FilterCatalogs.ZINC],
        "description": "RDKit ZINC unwanted-functionality filters, used as a published-method "
                        "proxy for REOS (Walters & Murcko 2002); original REOS SMARTS are not "
                        "fully public.",
    },
    "BRENK_negative_control": {
        "catalogs": [FilterCatalogParams.FilterCatalogs.BRENK],
        "description": "Brenk et al. 2008 -- reactive/toxic/unstable group alerts, mechanistically "
                        "unrelated to optical interference. Negative control.",
    },
}


def build_catalog(catalog_enums):
    params = FilterCatalogParams()
    for c in catalog_enums:
        params.AddCatalog(c)
    return FilterCatalog(params)


def build_full_pains_catalog():
    return build_catalog([FilterCatalogParams.FilterCatalogs.PAINS])


def is_chromophore_relevant(entry_name):
    return any(entry_name.startswith(prefix) for prefix in CHROMOPHORE_RELEVANT_PAINS_FAMILIES)


def partition_pains_entries(pains_catalog):
    """Return (chromophore_names, remainder_names) -- sets of PAINS entry
    description strings, partitioned by CHROMOPHORE_RELEVANT_PAINS_FAMILIES."""
    chromophore_names, remainder_names = set(), set()
    for i in range(pains_catalog.GetNumEntries()):
        name = pains_catalog.GetEntry(i).GetDescription()
        if is_chromophore_relevant(name):
            chromophore_names.add(name)
        else:
            remainder_names.add(name)
    return chromophore_names, remainder_names


def score_filter(catalog, mols):
    """Binary hit/no-hit per molecule against a full FilterCatalog. NaN if
    molecule failed to parse."""
    flags = []
    for mol in mols:
        if mol is None:
            flags.append(np.nan)
        else:
            flags.append(1 if catalog.HasMatch(mol) else 0)
    return np.array(flags, dtype=float)


def score_pains_subset(pains_catalog, mols, allowed_names):
    """Binary hit/no-hit per molecule, counting a hit only if the matching
    PAINS entry's name is in `allowed_names`. NaN if molecule failed to parse."""
    flags = []
    for mol in mols:
        if mol is None:
            flags.append(np.nan)
            continue
        hit = False
        for filter_match in pains_catalog.GetFilterMatches(mol):
            if filter_match.filterMatch.GetName() in allowed_names:
                hit = True
                break
        flags.append(1.0 if hit else 0.0)
    return np.array(flags, dtype=float)


def evaluate(y_true, y_flag):
    """Compute metrics for a binary flag used directly as a prediction.

    Rows where the molecule failed to parse (flag is NaN) are excluded.
    """
    mask = ~np.isnan(y_flag)
    y_true_m = y_true[mask]
    y_flag_m = y_flag[mask]

    n = len(y_true_m)
    n_pos = int(y_true_m.sum())
    n_flagged = int(y_flag_m.sum())

    out = {
        "n_molecules": n,
        "n_parse_failures": int((~mask).sum()),
        "n_positive_label": n_pos,
        "positive_rate": n_pos / n if n else np.nan,
        "n_flagged": n_flagged,
        "flagged_rate": n_flagged / n if n else np.nan,
    }

    # AUROC/AUPRC need both classes present in y_true, and some variance in y_flag
    if n_pos == 0 or n_pos == n:
        out["AUROC"] = np.nan
        out["AUPRC"] = np.nan
    else:
        try:
            out["AUROC"] = roc_auc_score(y_true_m, y_flag_m)
        except ValueError:
            out["AUROC"] = np.nan
        try:
            out["AUPRC"] = average_precision_score(y_true_m, y_flag_m)
        except ValueError:
            out["AUPRC"] = np.nan

    out["balanced_accuracy"] = balanced_accuracy_score(y_true_m, y_flag_m) if n else np.nan

    # Precision/recall of "flag => positive label" (treat filter hit as predicted-positive)
    tp = int(((y_flag_m == 1) & (y_true_m == 1)).sum())
    fp = int(((y_flag_m == 1) & (y_true_m == 0)).sum())
    fn = int(((y_flag_m == 0) & (y_true_m == 1)).sum())
    out["precision"] = tp / (tp + fp) if (tp + fp) else np.nan
    out["recall"] = tp / (tp + fn) if (tp + fn) else np.nan
    # at some point we may fix this (later) (todo: py library to use)

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV(.gz) with SMILES + binary label column")
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--label-col", default="activity")
    ap.add_argument("--target-name", required=True,
                     help="e.g. fluorescence_340_450, fluorescence_480plus, transmittance_340, transmittance_450plus")
    ap.add_argument("--out", required=True, help="Output CSV path for this target's results")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    compression = "gzip" if str(data_path).endswith(".gz") else None
    df = pd.read_csv(data_path, compression=compression)

    for col in (args.smiles_col, args.label_col):
        if col not in df.columns:
            sys.exit(f"ERROR: column '{col}' not found in {data_path}. Available: {list(df.columns)[:15]}...")

    smiles = df[args.smiles_col].astype(str).tolist()
    y_true = df[args.label_col].to_numpy(dtype=float)

    print(f"[{args.target_name}] Loaded {len(df)} molecules from {data_path}")
    print(f"[{args.target_name}] Parsing SMILES...")
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    n_failed = sum(m is None for m in mols)
    if n_failed:
        print(f"[{args.target_name}] WARNING: {n_failed}/{len(mols)} SMILES failed to parse and will be excluded")

    rows = []

    def add_row(filter_name, y_flag, description):
        metrics = evaluate(y_true, y_flag)
        metrics["target"] = args.target_name
        metrics["filter"] = filter_name
        metrics["filter_description"] = description
        rows.append(metrics)
        print(f"    AUROC={metrics['AUROC']:.4f}  AUPRC={metrics['AUPRC']:.4f}  "
              f"flagged_rate={metrics['flagged_rate']:.4f}  precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}" if not np.isnan(metrics["AUROC"])
              else f"    AUROC=NaN (degenerate labels) flagged_rate={metrics['flagged_rate']:.4f}")

    for filter_name, spec in FILTER_DEFINITIONS.items():
        print(f"[{args.target_name}] Scoring filter: {filter_name}")
        catalog = build_catalog(spec["catalogs"])
        y_flag = score_filter(catalog, mols)
        add_row(filter_name, y_flag, spec["description"])

    # PAINS chromophore-relevant subset vs. remainder
    print(f"[{args.target_name}] Scoring filter: PAINS_chromophore_subset")
    pains_catalog = build_full_pains_catalog()
    chromophore_names, remainder_names = partition_pains_entries(pains_catalog)
    print(f"    ({len(chromophore_names)} chromophore-relevant rules, "
          f"{len(remainder_names)} remainder rules, "
          f"{pains_catalog.GetNumEntries()} total)")

    y_flag_chromo = score_pains_subset(pains_catalog, mols, chromophore_names)
    add_row(
        "PAINS_chromophore_subset",
        y_flag_chromo,
        f"{len(chromophore_names)} PAINS rules from chromophore/fluorophore-relevant families "
        "(dyes, azo, quinone, naphthylamine, acridine, tetrazine, styrene, hydrazone). "
        "See CHROMOPHORE_RELEVANT_PAINS_FAMILIES in script header for family-level rationale.",
    )

    print(f"[{args.target_name}] Scoring filter: PAINS_remainder")
    y_flag_remainder = score_pains_subset(pains_catalog, mols, remainder_names)
    add_row(
        "PAINS_remainder",
        y_flag_remainder,
        f"Remaining {len(remainder_names)} PAINS rules (reactive/promiscuous-binder motifs "
        "with no specific chromophore relevance), reported for comparison against "
        "PAINS_chromophore_subset.",
    )

    out_df = pd.DataFrame(rows)
    col_order = ["target", "filter", "AUROC", "AUPRC", "balanced_accuracy", "precision", "recall",
                 "n_molecules", "n_parse_failures", "n_positive_label", "positive_rate",
                 "n_flagged", "flagged_rate", "filter_description"]
    out_df = out_df[col_order]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n[{args.target_name}] Results written to {out_path}")
    print(out_df.drop(columns=["filter_description"]).to_string(index=False))


if __name__ == "__main__":
    main()
