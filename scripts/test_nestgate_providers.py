#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Test NestGate data providers for airSpring data needs.

Tests:
1. Open-Meteo ERA5 fetch (single station, recent data) — no API key
2. NCBI ESearch for soil 16S rRNA metagenomes — no API key for basic access
3. NCBI EFetch for a known soil microbiome study accession

This validates the data pipeline before full NestGate Rust provider integration.
"""

import json
import sys
import time
from pathlib import Path

import requests

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

passed = 0
failed = 0
total = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}{f' — {detail}' if detail else ''}")


def test_open_meteo():
    """Fetch a single day of East Lansing weather from Open-Meteo ERA5."""
    print("\n=== Open-Meteo ERA5 Provider Test ===")

    params = {
        "latitude": 42.727,
        "longitude": -84.474,
        "start_date": "2023-07-15",
        "end_date": "2023-07-15",
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "relative_humidity_2m_max", "relative_humidity_2m_min",
            "precipitation_sum", "wind_speed_10m_max",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration",
        ]),
        "timezone": "America/Detroit",
    }

    try:
        resp = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
        check("HTTP 200", resp.status_code == 200, f"got {resp.status_code}")

        data = resp.json()
        daily = data.get("daily", {})

        check("has tmax", "temperature_2m_max" in daily and len(daily["temperature_2m_max"]) == 1)
        check("has tmin", "temperature_2m_min" in daily and len(daily["temperature_2m_min"]) == 1)
        check("has precip", "precipitation_sum" in daily)
        check("has radiation", "shortwave_radiation_sum" in daily)
        check("has ET0_fao", "et0_fao_evapotranspiration" in daily)

        tmax = daily["temperature_2m_max"][0]
        tmin = daily["temperature_2m_min"][0]
        et0 = daily["et0_fao_evapotranspiration"][0]

        check("tmax plausible (15-45°C July MI)", tmax is not None and 15 < tmax < 45, f"tmax={tmax}")
        check("tmin plausible (5-30°C July MI)", tmin is not None and 5 < tmin < 30, f"tmin={tmin}")
        check("ET0 plausible (1-12 mm/day July)", et0 is not None and 1 < et0 < 12, f"ET0={et0}")

        print(f"  Data: tmax={tmax}°C, tmin={tmin}°C, ET0={et0} mm/day")

    except requests.RequestException as e:
        check("network reachable", False, str(e))


def test_ncbi_soil_16s_search():
    """Search NCBI for soil 16S rRNA metagenome studies."""
    print("\n=== NCBI ESearch: Soil 16S rRNA ===")

    params = {
        "db": "sra",
        "term": "soil 16S agricultural",
        "retmode": "json",
        "retmax": 5,
    }

    try:
        time.sleep(0.4)
        resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
        check("HTTP 200", resp.status_code == 200, f"got {resp.status_code}")

        data = resp.json()
        result = data.get("esearchresult", {})
        count = int(result.get("count", 0))
        ids = result.get("idlist", [])

        check("results found", count > 0, f"count={count}")
        check("IDs returned", len(ids) > 0, f"ids={ids}")
        check("count > 100 (soil 16S is well-studied)", count > 100, f"count={count}")

        print(f"  Found {count} SRA entries for soil 16S, first IDs: {ids[:3]}")
        return ids[:3]

    except requests.RequestException as e:
        check("network reachable", False, str(e))
        return []


def test_ncbi_taxonomy_fetch():
    """Fetch taxonomy for a known soil bacterium (Bacillus subtilis, TaxID 1423)."""
    print("\n=== NCBI EFetch: Soil Bacterium Taxonomy ===")

    params = {
        "db": "taxonomy",
        "id": "1423",
        "retmode": "xml",
    }

    try:
        time.sleep(0.4)
        resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=30)
        check("HTTP 200", resp.status_code == 200, f"got {resp.status_code}")

        content = resp.text
        check("contains Bacillus", "Bacillus" in content)
        check("contains subtilis", "subtilis" in content)
        check("contains Firmicutes", "Firmicutes" in content or "Bacillota" in content)

        print(f"  Taxonomy XML length: {len(content)} bytes")

    except requests.RequestException as e:
        check("network reachable", False, str(e))


def test_ncbi_nucleotide_search():
    """Search NCBI nucleotide for soil 16S sequences."""
    print("\n=== NCBI ESearch: 16S rRNA Sequences ===")

    params = {
        "db": "nucleotide",
        "term": "16S ribosomal RNA[Title] AND soil[All Fields] AND 1200:1600[Sequence Length]",
        "retmode": "json",
        "retmax": 3,
    }

    try:
        time.sleep(0.4)
        resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
        check("HTTP 200", resp.status_code == 200, f"got {resp.status_code}")

        data = resp.json()
        result = data.get("esearchresult", {})
        count = int(result.get("count", 0))
        ids = result.get("idlist", [])

        check("results found", count > 0, f"count={count}")
        check("IDs returned", len(ids) > 0, f"ids={ids}")

        if ids:
            time.sleep(0.4)
            fetch_params = {
                "db": "nucleotide",
                "id": ids[0],
                "rettype": "fasta",
                "retmode": "text",
            }
            fasta_resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=fetch_params, timeout=30)
            check("FASTA fetch HTTP 200", fasta_resp.status_code == 200)
            check("FASTA starts with >", fasta_resp.text.startswith(">"))
            seq_lines = [l for l in fasta_resp.text.strip().split("\n") if not l.startswith(">")]
            seq_len = sum(len(l) for l in seq_lines)
            check("sequence length 1200-1600bp", 1000 < seq_len < 2000, f"len={seq_len}")
            print(f"  Fetched 16S sequence: {seq_len} bp")

        print(f"  Found {count} 16S sequences from soil studies")

    except requests.RequestException as e:
        check("network reachable", False, str(e))


def main():
    print("=" * 60)
    print("NestGate Provider Validation for airSpring Data Needs")
    print("=" * 60)

    test_open_meteo()
    test_ncbi_soil_16s_search()
    test_ncbi_taxonomy_fetch()
    test_ncbi_nucleotide_search()

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {passed}/{total} PASS")
    if failed == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{failed} CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
