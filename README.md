# ml_dl_rl_notes

A private, structured knowledge base for machine-learning, deep-learning, and reinforcement-learning papers.

## Research assistant

This repository includes a local Kiro agent that helps study papers through interactive explanation, source-aware Q&A, concept tracking, related-resource discovery, cross-paper comparisons, and small opt-in prototypes.

Start it from the repository root:

```bash
kiro-cli --agent research_assistant
```

At startup, provide:

1. A paper URL, DOI, arXiv URL, or readable local source.
2. A repository-relative notes path, normally `papers/<paper-slug>/notes.md`.
3. Any study goals. Goals are independent and can be selected together:
   - Understand the intuition
   - Reproduce the method
   - Implement a prototype
   - Compare with another paper
   - Prepare for an interview or presentation
4. The first priority and, optionally, your background level.

## Repository layout

Each paper uses the same workspace structure:

```text
papers/<paper-slug>/
├── notes.md          # Authoritative paper notes and Q&A interaction log
├── concepts.md       # Paper-specific concept checklist
├── resources.md      # Primary, related, prerequisite, and follow-up links
├── questions.md      # Unresolved questions
├── pyproject.toml    # Paper-level Python dependencies and metadata
├── uv.lock           # Reproducible dependency resolution
├── .venv/            # Local uv-managed environment; do not commit
├── scripts/          # Reproducible setup, run, and test entry points
└── prototypes/       # Small implementations requested by the user
```

Cross-paper indexes live under `knowledge/`:

```text
knowledge/
├── papers.md
├── concepts.md
├── methods.md
├── claims.md
├── comparisons.md
├── questions.md
└── resources.md
```

Paper notes are the source of truth. Knowledge files provide linked navigation and synthesis across papers; every entry should link back to its source paper.

## Useful commands

```text
/find <term>
/concept <name>
/compare <paper-or-method> <paper-or-method>
/synthesize <topic>
/related <paper>
/trace <claim>
/knowledge-status
/backlog-concepts <concepts>
```

Normal natural-language questions are supported too. The assistant preserves existing notes, appends one concise Q&A entry for each interaction, and reports any write failure instead of claiming that notes were saved.

## Capability quick guide

Each item below is a short usage pointer; the detailed behavior is defined in [research assistant steering rules](.kiro/steering/research_assistant.md).

- **Paper onboarding:** Start with a paper URL or local source, notes path, and any combination of study goals; see [session start](.kiro/steering/research_assistant.md#session-start).
- **Interactive tutoring:** Ask for intuition, section walkthroughs, equation explanations, quizzes, assumptions, or limitations; see [teaching behavior](.kiro/steering/research_assistant.md#teaching-behavior).
- **Persistent Q&A notes:** Every interaction is appended as one concise Q&A entry; see [one Q&A entry per interaction](.kiro/steering/research_assistant.md#one-qa-entry-per-interaction).
- **Paper workspaces:** Keep the same `notes.md`, `concepts.md`, `resources.md`, `questions.md`, and `prototypes/` layout for every paper; see [standard paper workspace](.kiro/steering/research_assistant.md#standard-paper-workspace).
- **Concept tracking:** Use `/backlog-concepts <concepts>` to record unresolved concepts without duplicates; see [concepts and resources](.kiro/steering/research_assistant.md#concepts-and-resources).
- **Resources and related papers:** Use `/related <paper>` and ask for verified prerequisite, follow-up, code, and dataset links; see [concepts and resources](.kiro/steering/research_assistant.md#concepts-and-resources).
- **Cross-paper search:** Use `/find <term>` or `/concept <name>` to search paper notes and central indexes; see [central knowledge base](.kiro/steering/research_assistant.md#central-knowledge-base).
- **Comparisons and synthesis:** Use `/compare <paper-or-method> <paper-or-method>` or `/synthesize <topic>` to connect ideas across papers; see [central knowledge base](.kiro/steering/research_assistant.md#central-knowledge-base).
- **Claim tracing:** Use `/trace <claim>` to locate source notes and supporting evidence; see [source discipline](.kiro/steering/research_assistant.md#source-discipline).
- **Reproducible prototypes:** Explicitly request a small implementation or reproduction; the assistant creates or reuses a paper-level `uv` environment, lockfile, scripts, and a smoke test where practical, and adds a container only when native/system or hardware parity requires it; see [prototypes and reproducibility](.kiro/steering/research_assistant.md#prototypes-and-reproducibility).
- **Knowledge-base status:** Use `/knowledge-status` to review papers, concepts, questions, resources, and prototypes in progress.

## Local configuration

- `.kiro/agents/research_assistant.json` — agent definition and tools.
- `.kiro/steering/research_assistant.md` — paper, notes, concept, resource, and prototype rules.
- `.kiro/hooks/research_assistant.json` — lifecycle reminders.
- `papers/` — consistent per-paper workspaces.
- `knowledge/` — central cross-paper indexes.
Notes and experiments in machine learning, deep learning, and reinforcement learning
