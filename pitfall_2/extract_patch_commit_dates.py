#!/usr/bin/env python3
"""
Extract commit dates for every (project_url, commit_id) in the PrimeVul dataset.

Input : ./datasets/primevul/{primevul_train.jsonl,primevul_test.jsonl,primevul_valid.jsonl}
Output: ./datasets/primevul/commit_dates.jsonl       (one JSON-per-line)

Each line:
    {"project": "<original project_url>",
     "commit_id": "<sha>",
     "commit_date": "2024-11-30T14:52:18+00:00"}
"""

from __future__ import annotations
import json, subprocess, shutil, tempfile, urllib.parse as up
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("./datasets/primevul")
SPLITS   = ["primevul_train.jsonl", "primevul_test.jsonl", "primevul_valid.jsonl"]
OUT_FILE = DATA_DIR / "commit_dates.jsonl"

# ───────────────────── helper functions ──────────────────────

def load_commits() -> dict[str, set[str]]:
    """Return {project_url: {sha, …}, …}."""
    projects: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        with open(DATA_DIR / split) as fh:
            for line in fh:
                rec = json.loads(line)
                projects[rec["project_url"].strip()].add(rec["commit_id"].strip())
    return projects

def normalise(url: str) -> str | None:
    """
    Convert a project URL from the dataset into something `git clone` understands.
    """
    if "gitweb" in url and "?p=" in url:
        parsed = up.urlparse(url)
        repo   = up.parse_qs(parsed.query).get("p", [None])[0]
        if repo:
            return f"{parsed.scheme}://{parsed.netloc}/git/{repo}.git"

    if url.startswith(("https://github.com/", "https://gitlab.com/")) and not url.endswith(".git"):
        return url + ".git"

    return url

def commit_date(repo: Path, sha: str) -> str:
    """Return ISO-8601 date for a commit inside `repo`."""
    return subprocess.check_output(
        ["git", "-C", repo, "show", "-s", "--format=%cI", sha],
        text=True,
    ).strip()

def load_existing_commits(filepath: Path) -> set[tuple[str, str]]:
    """Load existing (project, commit_id) pairs from output file."""
    existing = set()
    if filepath.exists():
        with open(filepath) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    existing.add((rec["project"], rec["commit_id"]))
                except json.JSONDecodeError:
                    continue
    return existing

# ───────────────────── main ──────────────────────

def main() -> None:
    projects = load_commits()
    processed = load_existing_commits(OUT_FILE)
    skipped_projects, skipped_commits, already_done = 0, 0, 0

    with open(OUT_FILE, "a") as out, tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_path = tmpdir / "repo"

        for proj_url, shas in projects.items():
            if any((proj_url, sha) in processed for sha in shas):
                print(f"Skipping already partially processed project: {proj_url}")
                skipped_projects += 1
                continue

            clone_url = normalise(proj_url)
            print(f"Cloning {clone_url} …")

            try:
                subprocess.run(["git", "clone", "--quiet", clone_url, repo_path], check=True)
            except subprocess.CalledProcessError:
                print(f"  ⚠️  failed – skipping whole project\n")
                skipped_projects += 1
                continue

            for sha in shas:
                try:
                    date = commit_date(repo_path, sha)
                    out.write(json.dumps({"project": proj_url,
                                          "commit_id": sha,
                                          "commit_date": date}) + "\n")
                    out.flush()
                    processed.add((proj_url, sha))
                except subprocess.CalledProcessError:
                    skipped_commits += 1
            shutil.rmtree(repo_path)

    print(f"\nDone. Results saved to {OUT_FILE}")
    if skipped_projects or skipped_commits:
        print(f"Skipped {skipped_projects} projects and {skipped_commits} commits.")

if __name__ == "__main__":
    main()