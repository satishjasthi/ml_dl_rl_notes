# Central knowledge base

The `knowledge/` directory contains cross-paper indexes. The detailed notes under `papers/<paper-slug>/` remain the source of truth; every index entry should link back to one or more paper files.

Recommended indexes:

- `papers.md` — paper catalog and topics.
- `concepts.md` — concepts and the papers where they appear.
- `methods.md` — methods, origins, extensions, and uses.
- `claims.md` — claims, evidence, limitations, and supporting or conflicting papers.
- `comparisons.md` — reusable comparison tables.
- `questions.md` — unresolved cross-paper questions.
- `resources.md` — verified paper, code, dataset, and learning-resource links.

The `research_assistant` agent maintains these indexes when adding papers or answering cross-paper questions. It should preserve existing entries, avoid duplicates, and cite source paper paths.
