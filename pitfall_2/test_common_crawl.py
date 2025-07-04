#!/usr/bin/env python3
"""
primevul_cc_presence_from_csv.py
Check whether each repository listed in project_urls.csv appears
in any of the 41 Common Crawl snapshots used for GPT-3 training.

Usage:
    python primevul_cc_presence_from_csv.py \
        --in  project_urls.csv \
        --out primevul_cc_presence.csv \
        --workers 32
"""

import argparse, csv, json, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, quote_plus

import requests

CDX_BASE   = "https://index.commoncrawl.org"
COLLINFO   = f"{CDX_BASE}/collinfo.json"
TIMEOUT    = 10          # seconds per request
PAUSE      = 0.1         # friendly delay between calls

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def get_snapshots():
    """Return the 41 CC-MAIN IDs spanning 2016-01 → 2019-10 (inclusive)."""
    resp = requests.get(COLLINFO, timeout=TIMEOUT)
    resp.raise_for_status()                            # fail fast on 503/404
    wanted = []
    for coll in resp.json():
        cid  = coll["id"]                              # e.g. CC-MAIN-2016-07
        year = int(cid.split("-")[2])
        if 2016 <= year <= 2019:
            wanted.append(cid)
    return sorted(wanted)                              # chronological order

def load_repos(path):
    """Read unique, non-empty URLs from project_urls.csv."""
    with open(path, newline="") as fh:
        return {line.strip().rstrip("/") for line in fh if line.strip()}

def repo_in_snapshot(repo_url, snapshot):
    """True ↔ at least one capture under repo_url exists in snapshot."""
    parsed  = urlparse(repo_url)
    prefix  = f"{parsed.netloc}{parsed.path.rstrip('/')}"
    query   = quote_plus(f"{prefix}/*")
    api     = (f"{CDX_BASE}/{snapshot}-index?url={query}"
               f"&matchType=prefix&limit=1&output=json")

    try:
        r = requests.get(api, timeout=TIMEOUT)
        if r.status_code == 200 and r.text.strip():
            return True
    except requests.RequestException:
        pass                                            # treat errors as absences
    time.sleep(PAUSE)
    return False

# ----------------------------------------------------------------------
def main(csv_in, csv_out, max_workers):
    snapshots = get_snapshots()
    repos     = sorted(load_repos(csv_in))
    total     = len(repos) * len(snapshots)
    print(f"{len(repos)} repos × {len(snapshots)} snapshots → {total:,} API calls",
          file=sys.stderr)

    with open(csv_out, "w", newline="") as fh, \
         ThreadPoolExecutor(max_workers=max_workers) as exe:

        writer = csv.writer(fh)
        writer.writerow(["repo_url", "snapshot_id", "in_commoncrawl"])

        futures = {exe.submit(repo_in_snapshot, repo, snap): (repo, snap)
                   for repo in repos for snap in snapshots}

        for fut in as_completed(futures):
            repo, snap = futures[fut]
            writer.writerow([repo, snap, fut.result()])

# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in",  default="project_urls.csv",
                   dest="csv_in", help="CSV with one repo URL per line")
    p.add_argument("--out", default="primevul_cc_presence.csv",
                   dest="csv_out", help="output CSV path")
    p.add_argument("--workers", type=int, default=32,
                   help="parallel HTTP requests (default 32)")
    args = p.parse_args()
    main(args.csv_in, args.csv_out, args.workers)