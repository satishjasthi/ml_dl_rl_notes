# Paper workspaces

Create one directory per paper using a stable lowercase slug:

```text
papers/<paper-slug>/
├── notes.md
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
