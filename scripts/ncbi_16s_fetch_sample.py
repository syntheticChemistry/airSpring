#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Fetch sample 16S data from NCBI for baseCamp 06 coupling validation.

Downloads metadata and representative sequences from a soil tillage
metagenome BioProject. This is the first stage of the NestGate → airSpring
data pipeline.

Target: PRJNA481146 — "16S amplicon sequences of soil microbiome after
different tillage managements" (Zuber & Villamil group)
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIOPROJECT = "PRJNA481146"
DATA_DIR = Path("data/ncbi_16s")

passed = 0
failed = 0
total = 0


def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}{f' — {detail}' if detail else ''}")
    return condition


def rate_limit():
    time.sleep(0.4)


def fetch_biosample_metadata(biosample_accessions):
    """Fetch BioSample metadata using accessions from SRA runs."""
    print(f"\n=== Step 1b: BioSample Metadata via SRA accessions ===")

    if not biosample_accessions:
        check("Have BioSample accessions from SRA", False)
        return []

    accs = list(set(biosample_accessions))[:20]
    print(f"  Fetching metadata for {len(accs)} BioSamples: {accs[:5]}...")

    rate_limit()
    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params={
        "db": "biosample",
        "term": " OR ".join(f"{a}[Accession]" for a in accs[:10]),
        "retmode": "json",
        "retmax": 50,
    }, timeout=30)
    data = resp.json().get("esearchresult", {})
    ids = data.get("idlist", [])
    count = int(data.get("count", 0))

    check("BioSample search OK", resp.status_code == 200)
    check(f"Found BioSamples", count > 0, f"count={count}")
    print(f"  Found {count} BioSamples, fetched IDs: {len(ids)}")

    if not ids:
        return []

    rate_limit()
    resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params={
        "db": "biosample",
        "id": ",".join(ids[:20]),
        "retmode": "xml",
    }, timeout=30)
    check("BioSample XML fetch OK", resp.status_code == 200)

    samples = []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        for bs in root.findall(".//BioSample"):
            sample_id = bs.get("accession", "")
            attrs = {}
            for attr in bs.findall(".//Attribute"):
                name = attr.get("attribute_name", attr.get("harmonized_name", ""))
                val = attr.text or ""
                if name:
                    attrs[name] = val

            tillage = attrs.get("tillage", attrs.get("management", attrs.get("treatment", "")))
            depth = attrs.get("depth", attrs.get("sample_depth", ""))
            geo = attrs.get("geographic location", attrs.get("geo_loc_name", ""))
            soil_type = attrs.get("soil_type", attrs.get("soil type", ""))

            samples.append({
                "biosample_accession": sample_id,
                "tillage": tillage,
                "depth": depth,
                "geo_location": geo,
                "soil_type": soil_type,
                "all_attrs": attrs,
            })
    except Exception as e:
        print(f"  Warning: XML parse issue: {e}")

    check("Parsed BioSample metadata", len(samples) > 0, f"parsed {len(samples)}")

    tillage_labels = [s["tillage"] for s in samples if s["tillage"]]
    if tillage_labels:
        unique = set(tillage_labels)
        print(f"  Tillage treatments found: {unique}")
        check("Has tillage treatment labels", len(unique) >= 1)
    else:
        all_keys = set()
        for s in samples[:5]:
            all_keys.update(s["all_attrs"].keys())
        print(f"  Available attributes: {sorted(all_keys)}")
        check("Has sample attributes", len(all_keys) > 3)

    return samples


def fetch_sra_runs():
    """Fetch SRA run accessions for the BioProject."""
    print(f"\n=== Step 2: SRA Runs for {BIOPROJECT} ===")

    rate_limit()
    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params={
        "db": "sra",
        "term": f"{BIOPROJECT}[BioProject]",
        "retmode": "json",
        "retmax": 200,
    }, timeout=30)
    data = resp.json().get("esearchresult", {})
    ids = data.get("idlist", [])
    count = int(data.get("count", 0))

    check(f"SRA search OK", resp.status_code == 200)
    check(f"Found SRA runs", count > 0, f"count={count}")
    print(f"  Total SRA runs: {count}")

    if not ids:
        return []

    rate_limit()
    resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params={
        "db": "sra",
        "id": ",".join(ids[:5]),
        "retmode": "xml",
    }, timeout=30)
    check("SRA XML fetch OK", resp.status_code == 200)

    runs = []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        for pkg in root.findall(".//EXPERIMENT_PACKAGE"):
            exp = pkg.find(".//EXPERIMENT")
            run_el = pkg.find(".//RUN")
            sample = pkg.find(".//SAMPLE")

            run_acc = run_el.get("accession", "") if run_el is not None else ""
            platform_el = pkg.find(".//INSTRUMENT_MODEL")
            platform = platform_el.text if platform_el is not None else ""
            lib_strat_el = pkg.find(".//LIBRARY_STRATEGY")
            lib_strat = lib_strat_el.text if lib_strat_el is not None else ""

            biosample_el = sample.find(".//EXTERNAL_ID[@namespace='BioSample']") if sample is not None else None
            biosample = biosample_el.text if biosample_el is not None else ""

            runs.append({
                "run_accession": run_acc,
                "biosample": biosample,
                "platform": platform,
                "library_strategy": lib_strat,
            })
            print(f"    {run_acc} | {biosample} | {platform} | {lib_strat}")
    except Exception as e:
        print(f"  Warning: SRA XML parse issue: {e}")

    check("Parsed SRA run metadata", len(runs) > 0, f"parsed {len(runs)}")
    return runs


def fetch_representative_16s():
    """Fetch a representative 16S sequence from the same soil type."""
    print(f"\n=== Step 3: Representative 16S Reference Sequence ===")

    rate_limit()
    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params={
        "db": "nucleotide",
        "term": "16S ribosomal RNA[Title] AND agricultural soil[All Fields] AND 1200:1600[Sequence Length]",
        "retmode": "json",
        "retmax": 3,
    }, timeout=30)
    data = resp.json().get("esearchresult", {})
    ids = data.get("idlist", [])

    check("16S reference search OK", resp.status_code == 200)
    check("Found reference sequences", len(ids) > 0)

    if ids:
        rate_limit()
        fasta_resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params={
            "db": "nucleotide",
            "id": ids[0],
            "rettype": "fasta",
            "retmode": "text",
        }, timeout=30)
        check("FASTA fetch OK", fasta_resp.status_code == 200)

        fasta = fasta_resp.text.strip()
        header = fasta.split("\n")[0]
        seq_lines = [l for l in fasta.split("\n") if not l.startswith(">")]
        seq_len = sum(len(l) for l in seq_lines)

        check("Valid FASTA header", fasta.startswith(">"))
        check("16S length 1200-1600bp", 1000 < seq_len < 2000, f"len={seq_len}")

        fasta_path = DATA_DIR / "reference_16s.fasta"
        fasta_path.write_text(fasta)
        print(f"  Saved: {fasta_path}")
        print(f"  Header: {header[:100]}")
        print(f"  Length: {seq_len} bp")

        return fasta


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"NCBI 16S Data Fetch for baseCamp 06: {BIOPROJECT}")
    print("=" * 70)

    runs = fetch_sra_runs()
    biosample_accs = [r["biosample"] for r in runs if r.get("biosample")]
    samples = fetch_biosample_metadata(biosample_accs)
    fasta = fetch_representative_16s()

    metadata = {
        "_provenance": {
            "query_date": "2026-02-28",
            "bioproject": BIOPROJECT,
            "purpose": "baseCamp 06 extension: no-till vs tillage 16S + soil moisture coupling",
            "pipeline": "NCBI ESearch → EFetch → wetSpring 16S → airSpring Anderson coupling",
        },
        "biosamples": samples[:20],
        "sra_runs": runs,
        "bioproject_total_runs": len(runs),
    }

    meta_path = DATA_DIR / f"{BIOPROJECT}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Saved metadata: {meta_path}")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {passed}/{total} PASS")
    if failed == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{failed} CHECKS FAILED")
        sys.exit(1)
    print(f"{'=' * 70}")

    print(f"\nNext steps:")
    print(f"  1. Download FASTQ files via SRA toolkit: fastq-dump --split-files <RUN_ACC>")
    print(f"  2. Process through wetSpring 16S pipeline → OTU table + Shannon H'")
    print(f"  3. Reconstruct site weather via Open-Meteo → FAO-56 ET₀ → water balance θ(t)")
    print(f"  4. Run airSpring Anderson coupling: θ(t) → d_eff(t) → QS regime prediction")
    print(f"  5. Compare predicted QS regime against observed community diversity")


if __name__ == "__main__":
    main()
