#!/usr/bin/env python3
"""
fetch_human_proteome.py
Download all reviewed human proteins from UniProt REST API.
Outputs:
  data/human_proteome/raw/uniprot_human_reviewed.tsv
  data/human_proteome/raw/protein_ids.txt
  data/human_proteome/raw/sequences.fasta
"""

import os, sys, time, csv
from pathlib import Path
import requests

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
QUERY = "(organism_id:9606) AND (reviewed:true)"
FIELDS = "accession,gene_primary,sequence,cc_subcellular_location"
PAGE_SIZE = 500

OUT_DIR = Path("data/human_proteome/raw")
TSV_FILE = OUT_DIR / "uniprot_human_reviewed.tsv"
IDS_FILE = OUT_DIR / "protein_ids.txt"
FASTA_FILE = OUT_DIR / "sequences.fasta"

# Existing dataset DE results
DATASET_DIRS = ["data/ucec/raw/de_results.csv",
                "data/coad/raw/de_results.csv",
                "data/luad/raw/de_results.csv"]


def fetch_all_proteins():
    """Paginate through UniProt REST API and collect all entries."""
    rows = []
    headers = None
    cursor = None
    page = 0

    while True:
        page += 1
        params = {
            "query": QUERY,
            "fields": FIELDS,
            "format": "tsv",
            "size": PAGE_SIZE,
        }
        if cursor:
            params["cursor"] = cursor

        print(f"  Fetching page {page} (collected {len(rows)} so far)...")
        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()

        lines = resp.text.strip().split("\n")
        if headers is None:
            headers = lines[0].split("\t")
            data_lines = lines[1:]
        else:
            data_lines = lines[1:] if lines[0].startswith("Entry") else lines

        for line in data_lines:
            if line.strip():
                rows.append(line.split("\t"))

        # Check for next page via Link header
        link = resp.headers.get("Link", "")
        if 'rel="next"' in link:
            # Extract cursor from link URL
            import re
            m = re.search(r'cursor=([^&>]+)', link)
            if m:
                cursor = m.group(1)
            else:
                break
        else:
            break

        time.sleep(0.5)  # polite rate limiting

    return headers, rows


def load_existing_ids():
    """Load UniProt IDs from existing dataset DE results."""
    existing = set()
    for path in DATASET_DIRS:
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing.add(row["UniProt_ID"])
    return existing


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Fetching reviewed human proteome from UniProt ===")
    headers, rows = fetch_all_proteins()
    print(f"  Downloaded {len(rows)} proteins")

    if not rows:
        print("ERROR: No proteins retrieved. Check network/API.", file=sys.stderr)
        sys.exit(1)

    # Identify column indices
    col_idx = {h: i for i, h in enumerate(headers)}
    acc_i = col_idx.get("Entry", col_idx.get("accession", 0))
    gene_i = col_idx.get("Gene Names (primary)", col_idx.get("gene_primary", 1))
    seq_i = col_idx.get("Sequence", col_idx.get("sequence", 2))
    loc_i = col_idx.get("Subcellular location [CC]",
                        col_idx.get("cc_subcellular_location", 3))

    # Write TSV
    print(f"  Saving TSV -> {TSV_FILE}")
    with open(TSV_FILE, "w", newline="") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")

    # Write protein IDs
    ids = []
    for row in rows:
        acc = row[acc_i] if acc_i < len(row) else ""
        if acc:
            ids.append(acc)

    print(f"  Saving IDs -> {IDS_FILE}")
    with open(IDS_FILE, "w") as f:
        for pid in ids:
            f.write(pid + "\n")

    # Write FASTA
    print(f"  Saving FASTA -> {FASTA_FILE}")
    with open(FASTA_FILE, "w") as f:
        for row in rows:
            acc = row[acc_i] if acc_i < len(row) else "unknown"
            gene = row[gene_i] if gene_i < len(row) else ""
            seq = row[seq_i] if seq_i < len(row) else ""
            if not seq:
                continue
            header = f">{acc}|{gene}" if gene else f">{acc}"
            f.write(header + "\n")
            # Wrap sequence at 60 chars
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

    # Summary
    existing_ids = load_existing_ids()
    all_ids = set(ids)
    overlap = all_ids & existing_ids
    novel = all_ids - existing_ids

    print(f"\n=== Summary ===")
    print(f"  Total reviewed human proteins: {len(ids)}")
    print(f"  Already in ucec/coad/luad:     {len(overlap)}")
    print(f"  Truly novel (not in training): {len(novel)}")
    print(f"  Output files:")
    print(f"    {TSV_FILE}")
    print(f"    {IDS_FILE}")
    print(f"    {FASTA_FILE}")


if __name__ == "__main__":
    main()
