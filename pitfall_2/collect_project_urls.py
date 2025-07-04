#!/usr/bin/env python3
"""
Extract all unique project URLs from the PrimeVul dataset.

The script expects the three JSONL files
    - primevul_train.jsonl
    - primevul_valid.jsonl
    - primevul_test.jsonl

to live inside a directory (default: datasets/primevul).

Usage
-----
# 1 Print URLs to stdout
python extract_primevul_urls.py

# 2 Save them to a text file
python extract_primevul_urls.py --output urls.txt

# 3 If your data is elsewhere
python extract_primevul_urls.py --data-dir /path/to/primevul
"""
import argparse
import json
from pathlib import Path
from typing import List, Set


def collect_urls(data_dir: Path) -> List[str]:
    """Return a sorted list of unique project URLs found in the dataset."""
    splits = [
        "primevul_train.jsonl",
        "primevul_valid.jsonl",
        "primevul_test.jsonl",
    ]

    urls: Set[str] = set()

    for split in splits:
        file_path = data_dir / split
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found. Check --data-dir argument.")
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    # skip blank lines
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {file_path}: {e}") from e
                url = obj.get("project_url")
                if url:
                    urls.add(url)

    return sorted(urls)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all unique project URLs from the PrimeVul dataset."
    )
    parser.add_argument(
        "--data-dir",
        default="datasets/primevul",
        help="Directory where the JSONL files reside (default: datasets/primevul)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output file path. Use '-' (default) to write to stdout.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    urls = collect_urls(data_dir)

    if args.output in {"-", ""}:
        for url in urls:
            print(url)
    else:
        output_path = Path(args.output)
        output_path.write_text("\n".join(urls), encoding="utf-8")
        print(f"Wrote {len(urls)} unique URLs to {output_path}")  # noqa: T201


if __name__ == "__main__":
    main()