#!/usr/bin/env python3
"""Render a research paper Markdown notes file into a small, dependency-free PDF.

The generated PDF records the SHA-256 of the source notes in its metadata.  This
lets ``check`` detect whether a manually generated PDF is stale without a
sidecar file or any external Markdown/PDF tool.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import textwrap
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LIST_RE = re.compile(r"^(\s*)([-+*]|\d+[.)])\s+(.*)$")
LINK_RE = re.compile(r"!?(\[([^]]+)\])\(([^)]+)\)")
INLINE_MARKUP_RE = re.compile(r"(`+|\*\*|__|~~|\*|_)")
HASH_RE = re.compile(rb"/NotesSHA256 \(([0-9a-f]{64})\)")

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
MARGIN_X = 54
TOP_Y = 738
BOTTOM_Y = 54
LINE_HEIGHT = 14


class NotesPdfError(Exception):
    """A user-facing input or rendering error."""


def clean_inline(text: str) -> str:
    """Make common Markdown inline syntax readable in a plain PDF font."""
    text = LINK_RE.sub(lambda match: match.group(2), text)
    text = INLINE_MARKUP_RE.sub("", text)
    return text


def parse_notes(markdown: str) -> tuple[str, list[tuple[str, str, int]]]:
    """Parse the Markdown subset used by paper notes.

    The exporter intentionally keeps this parser small and predictable. It
    supports headings, paragraphs, lists, block math, fenced code, quotes, and
    horizontal rules; unsupported Markdown is retained as readable text.
    """
    blocks: list[tuple[str, str, int]] = []
    paragraph: list[str] = []
    in_code = False
    code_lines: list[str] = []
    title = "Research paper notes"

    def flush_paragraph() -> None:
        if paragraph:
            value = " ".join(line.strip() for line in paragraph).strip()
            if value:
                blocks.append(("body", value, 0))
            paragraph.clear()

    def flush_code() -> None:
        if code_lines:
            blocks.extend(("code", line, 0) for line in code_lines)
            code_lines.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            flush_paragraph()
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            value = clean_inline(heading.group(2).strip())
            if level == 1 and title == "Research paper notes":
                title = value or title
            blocks.append((f"h{level}", value, 0))
            continue

        if re.match(r"^\s*(---+|___+|\*\*\*+)\s*$", line):
            flush_paragraph()
            blocks.append(("rule", "", 0))
            continue

        list_item = LIST_RE.match(line)
        if list_item:
            flush_paragraph()
            indent = len(list_item.group(1).replace("\t", "  ")) // 2
            blocks.append(("bullet", f"{list_item.group(2)} {clean_inline(list_item.group(3))}", indent))
            continue

        if line.lstrip().startswith(">"):
            flush_paragraph()
            blocks.append(("quote", clean_inline(line.lstrip()[1:].strip()), 0))
            continue

        if line.lstrip().startswith("|"):
            # Keep tables legible without pretending to perform full table layout.
            flush_paragraph()
            if not re.match(r"^\s*\|?\s*:?-{3,}", line):
                blocks.append(("table", clean_inline(line.strip()), 0))
            continue

        paragraph.append(line)

    if in_code:
        flush_code()
    flush_paragraph()
    return title, blocks


def ascii_text(value: str) -> str:
    """Convert Unicode to text supported by PDF's built-in Helvetica font."""
    replacements = {
        "—": "-", "–": "-", "‑": "-", "−": "-", "…": "...",
        "→": "->", "←": "<-", "↔": "<->", "≤": "<=", "≥": ">=",
        "≠": "!=", "×": "x", "÷": "/", "•": "*", "·": ".",
        "’": "'", "‘": "'", "“": '"', "”": '"', " ": " ",
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    normalized = value.encode("ascii", "replace").decode("ascii")
    return normalized.replace("\r", "")


def wrap_block(kind: str, value: str, indent: int) -> list[tuple[str, str, int]]:
    """Wrap one parsed block into renderable lines."""
    if kind == "rule":
        return [(kind, "", indent)]
    if kind == "bullet":
        prefix = "  " * indent
        marker, _, content = value.partition(" ")
        first_prefix = f"{prefix}{marker} "
        continuation = " " * len(first_prefix)
        width = max(20, 94 - len(first_prefix))
        chunks = textwrap.wrap(content, width=width, break_long_words=True, break_on_hyphens=False) or [""]
        return [(kind, first_prefix + chunks[0], indent)] + [(kind, continuation + chunk, indent) for chunk in chunks[1:]]

    prefix = "  " * indent if kind in {"quote", "table"} else ""
    if kind == "code":
        prefix = "    "
    width = max(20, 98 - len(prefix))
    chunks = textwrap.wrap(value, width=width, break_long_words=True, break_on_hyphens=False) or [""]
    return [(kind, prefix + chunk, indent) for chunk in chunks]


def render_lines(blocks: list[tuple[str, str, int]]) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for kind, value, indent in blocks:
        lines.extend((line_kind, ascii_text(line)) for line_kind, line, _ in wrap_block(kind, value, indent))
        if kind in {"h1", "h2", "h3", "rule"}:
            lines.append(("spacer", ""))
    return lines


def pdf_escape(value: str) -> str:
    return ascii_text(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(title: str, lines: list[tuple[str, str]], notes_hash: str) -> bytes:
    """Build a valid PDF using only built-in Type 1 fonts."""
    pages: list[bytes] = []
    y = TOP_Y
    page_lines: list[tuple[str, str]] = []

    def finish_page(page_number: int) -> None:
        nonlocal page_lines
        commands = ["q"]
        page_y = TOP_Y
        for kind, value in page_lines:
            line_height = 18 if kind == "h1" else 16 if kind in {"h2", "h3"} else LINE_HEIGHT
            if kind == "spacer":
                page_y -= 4
                continue
            size = {"h1": 18, "h2": 15, "h3": 13, "h4": 11, "h5": 10, "h6": 10,
                    "code": 9, "quote": 10, "table": 9, "bullet": 10}.get(kind, 10)
            font = "F2" if kind.startswith("h") else "F1"
            x = MARGIN_X + (12 if kind == "quote" else 0)
            commands.append(f"BT /{font} {size} Tf {x} {page_y:.1f} Td ({pdf_escape(value)}) Tj ET")
            page_y -= line_height
        commands.append("Q")
        # Add a small footer separately so it remains at the bottom of every page.
        commands.extend([
            "BT /F1 8 Tf 54 32 Td (Generated from notes.md - page " + str(page_number) + ") Tj ET",
        ])
        page_lines = []
        pages.append("\n".join(commands).encode("latin-1"))

    page_number = 1
    for kind, value in lines:
        line_height = 18 if kind == "h1" else 16 if kind in {"h2", "h3"} else LINE_HEIGHT
        if y < BOTTOM_Y + line_height:
            finish_page(page_number)
            page_number += 1
            y = TOP_Y
        page_lines.append((kind, value))
        if kind == "spacer":
            y -= 4
        else:
            y -= line_height
    if page_lines or not pages:
        finish_page(page_number)

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    page_object_numbers = []
    next_object = 5 + (len(pages) * 2)
    for index in range(len(pages)):
        page_object_numbers.append(5 + index * 2)
    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    for index, page_content in enumerate(pages):
        page_number = 5 + index * 2
        content_number = page_number + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents {content_number} 0 R >>".encode("ascii")
        )
        objects.append(f"<< /Length {len(page_content)} >>\nstream\n".encode("ascii") + page_content + b"\nendstream")

    # The hash in /Info is the synchronization contract used by ``check``.
    info_number = next_object
    objects.append(
        f"<< /Title ({pdf_escape(title)}) /Subject (Research paper notes) "
        f"/Producer (ml_dl_rl_notes notes_to_pdf.py) /NotesSHA256 ({notes_hash}) >>".encode("latin-1")
    )

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_number} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_output(notes_path: Path) -> Path:
    return notes_path.with_suffix(".pdf")


def write_pdf(notes_path: Path, output_path: Path) -> str:
    if notes_path.resolve() == output_path.resolve():
        raise NotesPdfError("the PDF output must not overwrite the Markdown notes file")
    try:
        markdown = notes_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise NotesPdfError(f"notes file not found: {notes_path}") from exc
    except UnicodeDecodeError as exc:
        raise NotesPdfError(f"notes file is not valid UTF-8: {notes_path}") from exc

    digest = source_hash(notes_path)
    title, blocks = parse_notes(markdown)
    pdf = build_pdf(title, render_lines(blocks), digest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(pdf)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def check_pdf(notes_path: Path, output_path: Path) -> int:
    if not output_path.exists():
        print(f"STALE: generated PDF does not exist: {output_path}", file=sys.stderr)
        return 1
    digest = source_hash(notes_path)
    match = HASH_RE.search(output_path.read_bytes())
    if not match:
        print(f"STALE: PDF has no notes hash (regenerate it): {output_path}", file=sys.stderr)
        return 1
    recorded = match.group(1).decode("ascii")
    if recorded != digest:
        print(f"STALE: notes changed since {output_path} was generated", file=sys.stderr)
        print(f"  expected {digest}", file=sys.stderr)
        print(f"  recorded {recorded}", file=sys.stderr)
        return 1
    print(f"FRESH: {output_path} matches {notes_path} ({digest})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and manually sync PDFs from paper notes Markdown.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("generate", "create or replace the PDF"),
        ("sync", "replace the PDF with the current notes"),
        ("check", "check whether the PDF matches the current notes"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("notes", type=Path, help="path to a Markdown notes file")
        subparser.add_argument("--output", type=Path, help="PDF path (default: notes.md beside the source as notes.pdf)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    notes_path = args.notes
    output_path = args.output or default_output(notes_path)
    if not notes_path.is_file():
        print(f"error: notes file not found: {notes_path}", file=sys.stderr)
        return 2
    try:
        if args.command == "check":
            return check_pdf(notes_path, output_path)
        digest = write_pdf(notes_path, output_path)
    except (OSError, NotesPdfError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    verb = "Generated" if args.command == "generate" else "Synced"
    print(f"{verb}: {output_path} from {notes_path} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
