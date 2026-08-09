"""Extract approved headwords and listed forms from the Issue 9 PDF."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader

ENTRY = re.compile(
    r"^([A-Z][A-Z0-9./+-]*(?: [A-Z0-9./+-]+)?)\s+\(([^)]+)\)(?:,)?(?:\s+.*)?$"
)
FORM = re.compile(r"^[A-Z][A-Z0-9-]*,$")
DICTIONARY_FIRST_PAGE = 148  # zero-based; PDF page 149 is Part 2, A1


def extract(pdf: Path) -> set[str]:
    reader = PdfReader(pdf)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError(f"Cannot decrypt {pdf}")

    words: set[str] = set()
    for page in reader.pages[DICTIONARY_FIRST_PAGE:]:
        lines = [" ".join(line.split()) for line in (page.extract_text() or "").splitlines()]
        for index, line in enumerate(lines):
            match = ENTRY.match(line)
            if not match:
                continue
            words.add(match.group(1).lower())
            if not line.endswith(","):
                continue
            for following in lines[index + 1 : index + 5]:
                if FORM.fullmatch(following):
                    words.add(following[:-1].lower())
                else:
                    break
    return words


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_ste_words.py INPUT.pdf OUTPUT.txt")
    words = extract(Path(sys.argv[1]))
    if len(words) < 500:
        raise SystemExit(f"only extracted {len(words)} words; inspect the PDF/parser")
    Path(sys.argv[2]).write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
    print(f"wrote {len(words)} words")


if __name__ == "__main__":
    main()
