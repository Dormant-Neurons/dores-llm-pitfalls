#!/usr/bin/env python3
"""
Find the first‑token index of the line
    pairs = palloc(key_count * sizeof(Pairs));
inside a C source file using the CodeT5‑small tokenizer.

- Works even when the file is >512 tokens (no model inference is done).
- Tolerates extra / fewer spaces and the pcount‑versus‑key_count variant.

Requires:
    pip install transformers sentencepiece
"""

import re
from pathlib import Path
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# CONFIG ­– change these two lines if you want a different file / target
# ----------------------------------------------------------------------
SOURCE_FILE = Path("pitfall_5/bigvul_example.c")   # or Path("your_file.c")
TARGET_REGEXES = [
    # user’s original variant
    r"pairs\s*=\s*palloc\s*\(\s*pcount\s*\*\s*sizeof\s*\(\s*Pairs\s*\)\s*\)\s*;",
    # the variant actually present in PostgreSQL
    r"pairs\s*=\s*palloc\s*\(\s*key_count\s*\*\s*sizeof\s*\(\s*Pairs\s*\)\s*\)\s*;",
]

# ----------------------------------------------------------------------
# 1. Read file and locate the target line
# ----------------------------------------------------------------------
code = SOURCE_FILE.read_text(encoding="utf‑8", errors="replace")

match = None
for pattern in TARGET_REGEXES:
    match = re.search(pattern, code)
    if match:
        break

if not match:
    raise SystemExit("❌  Target line not found in the source file.")

# ----------------------------------------------------------------------
# 2. Tokenise everything *before* the target line
# ----------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
tokens_before = tokenizer.tokenize(code[: match.start()])
first_index   = len(tokens_before)

# Optional: tokenise the line itself so you can see what matched
tokens_line   = tokenizer.tokenize(code[match.start() : match.end()])

# ----------------------------------------------------------------------
# 3. Report
# ----------------------------------------------------------------------
print(f"First token index for the target line: {first_index}")
print("Tokens of the target line:")
print(tokens_line)