# Paper workspaces

Create one directory per paper using a stable lowercase slug:

```text
papers/<paper-slug>/
├── notes.md
├── concepts.md
├── resources.md
├── questions.md
└── prototypes/
```

`notes.md` is the authoritative paper record and contains the interaction log. Companion files should link back to the paper notes. The `research_assistant` agent creates or updates these files when a paper is studied.
