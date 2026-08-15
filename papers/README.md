# Paper workspaces

Create one directory per paper using a stable lowercase slug:

```text
papers/<paper-slug>/
├── notes.md
├── notes.pdf          # Optional generated PDF export of notes.md
├── concepts.md
├── resources.md
├── questions.md
├── pyproject.toml       # Paper-level Python dependencies and metadata
├── uv.lock              # Reproducible dependency resolution
├── .venv/               # Local uv-managed environment; do not commit
├── scripts/             # Reproducible setup, run, and test entry points
└── prototypes/
```

`notes.md` is the authoritative paper record and contains the interaction log. Companion files should link back to the paper notes. The `research_assistant` agent creates or updates these files when a paper is studied.

For a readable export, use the repository-level `scripts/notes_to_pdf.py` command documented in the root [README](../README.md). PDF generation and synchronization are manual maintenance actions and do not modify `notes.md`.
