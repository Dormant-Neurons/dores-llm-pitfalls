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

# ───────────────────────── helper functions ──────────────────────────

def load_commits() -> dict[str, set[str]]:
    """Return {project_url: {sha, …}, …}."""
    projects: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        with open(DATA_DIR / split) as fh:
            for line in fh:
                rec = json.loads(line)
                projects[rec["project_url"]].add(rec["commit_id"])
    return projects


def normalise(url: str) -> str | None:
    """
    Convert a project URL from the dataset into something `git clone` understands.

    Currently:
      • GitHub/GitLab/etc: append '.git' if missing
      • Savannah GitWeb:   https://git.savannah.gnu.org/gitweb/?p=gnutls
                           → https://git.savannah.gnu.org/git/gnutls.git
    Return None if we cannot guess.
    """
    if "gitweb" in url and "?p=" in url:
        parsed = up.urlparse(url)
        repo   = up.parse_qs(parsed.query).get("p", [None])[0]
        if repo:
            return f"{parsed.scheme}://{parsed.netloc}/git/{repo}.git"

    if url.startswith(("https://github.com/", "https://gitlab.com/")) and not url.endswith(".git"):
        return url + ".git"

    # already looks cloneable – hope for the best
    return url


def commit_date(repo: Path, sha: str) -> str:
    """Return ISO-8601 date for a commit inside `repo`."""
    return subprocess.check_output(
        ["git", "-C", repo, "show", "-s", "--format=%cI", sha],
        text=True,
    ).strip()


# ────────────────────────────── main ─────────────────────────────────

def main() -> None:
    projects = load_commits()
    skipped_projects, skipped_commits = 0, 0

    with open(OUT_FILE, "w") as out, tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_path = tmpdir / "repo"   # will be wiped after each project

        for proj_url, shas in projects.items():
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
                except subprocess.CalledProcessError:
                    skipped_commits += 1   # SHA missing
            # clean up for the next repo
            shutil.rmtree(repo_path)

    print(f"\nDone. Results saved to {OUT_FILE}")
    if skipped_projects or skipped_commits:
        print(f"Skipped {skipped_projects} projects and {skipped_commits} commits.")


if __name__ == "__main__":
    main()