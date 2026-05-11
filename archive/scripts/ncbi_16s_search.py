#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Search NCBI for agricultural soil 16S metagenome studies.

Finds BioProject accessions for no-till vs conventional tillage
soil microbiome studies — the data needed for baseCamp 06 extension.
"""

import json
import sys
import time
import requests
import xml.etree.ElementTree as ET

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def esearch(db, term, retmax=20):
    time.sleep(0.4)
    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params={
        "db": db, "term": term, "retmode": "json", "retmax": retmax,
    }, timeout=30)
    resp.raise_for_status()
    result = resp.json().get("esearchresult", {})
    return result.get("idlist", []), int(result.get("count", 0))


def esummary(db, ids):
    if not ids:
        return {}
    time.sleep(0.4)
    resp = requests.get(f"{NCBI_BASE}/esummary.fcgi", params={
        "db": db, "id": ",".join(ids[:10]), "retmode": "json",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json().get("result", {})


def efetch_xml(db, ids):
    if not ids:
        return ""
    time.sleep(0.4)
    resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params={
        "db": db, "id": ",".join(ids[:10]), "retmode": "xml",
    }, timeout=30)
    resp.raise_for_status()
    return resp.text


def search_bioproject_tillage():
    """Search for no-till / tillage 16S soil BioProjects."""
    print("=" * 70)
    print("NCBI BioProject Search: Soil Tillage 16S Metagenome Studies")
    print("=" * 70)

    queries = [
        ("tillage soil 16S microbiome", "bioproject"),
        ("no-till soil bacterial community 16S", "bioproject"),
        ("conservation tillage soil metagenome", "bioproject"),
    ]

    all_ids = set()
    for term, db in queries:
        ids, count = esearch(db, term, retmax=10)
        print(f"\n  Query: '{term}' → {count} results, {len(ids)} IDs")
        all_ids.update(ids)

    print(f"\n  Unique BioProject IDs: {len(all_ids)}")

    if all_ids:
        summaries = esummary("bioproject", list(all_ids))
        print(f"\n{'─' * 70}")
        print("BioProject Summaries:")
        print(f"{'─' * 70}")

        projects = []
        for uid in sorted(all_ids):
            info = summaries.get(uid, {})
            if isinstance(info, dict):
                title = info.get("Project_Title", info.get("project_title", "N/A"))
                acc = info.get("Project_Acc", info.get("project_acc", "N/A"))
                desc = info.get("Project_Description", info.get("project_description", ""))
                org = info.get("Project_Data_Type_Name", info.get("Organism_Name", ""))

                if desc and len(desc) > 120:
                    desc = desc[:120] + "..."

                print(f"\n  [{uid}] {acc}")
                print(f"    Title: {title}")
                if desc:
                    print(f"    Desc:  {desc}")
                if org:
                    print(f"    Org:   {org}")

                projects.append({
                    "uid": uid,
                    "accession": acc,
                    "title": title,
                    "description": desc,
                })

        return projects
    return []


def search_sra_for_project(bioproject_acc):
    """Search SRA for runs belonging to a BioProject."""
    ids, count = esearch("sra", f"{bioproject_acc}[BioProject]", retmax=5)
    return ids, count


def main():
    projects = search_bioproject_tillage()

    if projects:
        print(f"\n{'=' * 70}")
        print("SRA Run Counts per BioProject:")
        print(f"{'=' * 70}")

        for proj in projects[:10]:
            acc = proj["accession"]
            if acc and acc != "N/A":
                ids, count = search_sra_for_project(acc)
                print(f"  {acc}: {count} SRA runs (sample IDs: {ids[:3]})")
                proj["sra_count"] = count
                proj["sample_sra_ids"] = ids[:3]

    print(f"\n{'=' * 70}")
    print("Search complete. Use these BioProject accessions to download 16S data.")
    print(f"{'=' * 70}")

    out_path = "data/ncbi_16s_projects.json"
    import os
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "query_date": "2026-02-28",
            "purpose": "baseCamp 06 extension: no-till vs tillage soil 16S coupling",
            "projects": projects,
        }, f, indent=2)
    print(f"\nSaved project metadata to {out_path}")


if __name__ == "__main__":
    main()
