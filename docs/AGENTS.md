# Contributing to RAG Facts Check Documentation — AI Assistant Instructions

> **Read this file before making any changes to `docs/`.** It tells you how
> the documentation is structured, what conventions to follow, and what
> workflows to use when adding, editing, or restructuring documents.

---

## 1. What This Is

The `docs/` directory is an **[Open Knowledge Format (OKF) v0.1](https://github.com/open-knowledge-format/spec)**
knowledge bundle. It is a directory of markdown files with YAML frontmatter,
organized into subdirectories, with `index.md` files for progressive
disclosure and a `log.md` for change history.

---

## 2. Directory Layout

```
docs/
├── AGENTS.md              # this file
├── index.md               # root directory listing
├── log.md                 # chronological change log
├── overview/              # project intro, approaches survey
├── architecture/          # module layout, data flow, output format
├── guides/                # setup, config, web service, testing
├── architecture-tension.md  # standalone design discussion
└── debug-prompts/         # CLI debug command reference
```

---

## 3. Reserved filenames

- `index.md` — directory listing. No frontmatter. Bulleted list of concepts
  with links and one-line descriptions.
- `log.md` — chronological change log. No frontmatter. Date headings in
  `YYYY-MM-DD` format, newest first.
- `AGENTS.md` — this file. Not a concept document.

All other `.md` files are **concept documents** and MUST have YAML frontmatter.

---

## 4. Concept Document Format

Every concept document MUST have:

```yaml
---
type: <Type name>                  # REQUIRED — see §5 for valid types
title: <Display name>              # RECOMMENDED
description: '<One-line summary>'   # RECOMMENDED — wrap in single quotes if it contains colons
tags: [tag1, tag2]                 # RECOMMENDED
timestamp: '2025-01-01T00:00:00Z'  # RECOMMENDED — update when you edit
---

# Document body in standard markdown
```

### Body conventions

- Use **structural markdown** — headings, lists, tables, fenced code blocks —
  over freeform prose.
- Use **Mermaid** for diagrams (not ASCII art). Wrap in ```` ```mermaid ` ``` fences.
- Use **absolute bundle-relative links** (starting with `/`) for cross-references
  to other docs. Example: `[Pipeline](/architecture/data-flow.md)` not
  `[Pipeline](../architecture/data-flow.md)`.
- Keep `> **Update when:** ...` near the top of documents that have a natural
  trigger for review (e.g. API changes, config updates).

---

## 5. Valid type values

| Type | Purpose |
|------|---|
| `Architecture` | System architecture, module layout, design discussions |
| `DataFlow` | Pipeline phases, field-level data tracing |
| `DataModel` | Internal data structures, report schemas |
| `Guide` | Setup, usage, integration guides |
| `Configuration` | Env vars, config reference, tuning parameters |
| `EntryPoint` | CLI, worker, API entry points |
| `Framework` | Testing, evaluation frameworks |
| `Tool` | Debug tools, CLI commands |
| `Reference` | Quick reference, surveys, comparisons |

If none fit, create a new type that is descriptive and self-explanatory.

---

## 6. Workflows

### Adding a new document

1. Decide which subdirectory it belongs to. If none fit, ask the user before
   creating a new subdirectory.
2. Create the `.md` file with proper YAML frontmatter (`type` is required).
3. Write the body following conventions in §4.
4. Add an entry to the parent subdirectory's `index.md`.
5. Add an entry to `docs/log.md` under today's date.
6. Add cross-links from related documents.

### Editing an existing document

1. Read the document fully before making changes.
2. Make the content changes.
3. Update the `timestamp` in frontmatter.
4. Update `description` or `tags` if they are no longer accurate.
5. Add an entry to `docs/log.md` under today's date.
6. Check that cross-links from other documents still resolve.

### Restructuring

1. Move or rename the file.
2. Update ALL cross-links throughout the bundle (search for the old filename).
3. Update `index.md` files in affected directories.
4. Add an entry to `docs/log.md`.

### Updating `index.md` files

- No frontmatter. Use headings and bulleted lists.
- Each entry: `[Title](file.md) - One-line description from frontmatter.`
- Keep entries sorted alphabetically within sections.

### Updating `log.md`

Append-only. Add entries under today's date (`## YYYY-MM-DD`), newest first:

```markdown
## 2025-07-15
* **Creation**: Added [New Doc](/path/to/doc.md) describing X.
* **Update**: Updated [Existing Doc](/path/to/doc.md) — changed Y.
* **Restructure**: Moved [Doc](/path/to/doc.md) from old/ to new/.
```

---

## 7. Cross-Linking Rules

- **Always use absolute bundle-relative links** starting with `/`.
  Example: `/architecture/data-flow.md` not `../architecture/data-flow.md`.
- **Use descriptive link text** — `[Data Flow](/architecture/data-flow.md)`
  not `[this](/architecture/data-flow.md)`.
- **Link to related documents** from within the body when a concept is
  discussed that another document covers in detail.

---

## 8. What NOT to do

- **Do NOT skip the `type` frontmatter field** — it is required for OKF conformance.
- **Do NOT forget to update `log.md`** — the change log tracks bundle evolution.
- **Do NOT forget to update the relevant `index.md`** — every new or moved
  document must be listed.
- **Do NOT use ASCII art for diagrams** — use Mermaid instead.
- **Do NOT use relative links between docs** — always use `/`-prefixed paths.

---

## 9. Relationship to Other Project Files

- **`AGENTS.md`** (repo root) — project-level AI instructions. Covers git
  conventions, working patterns, pyproject.toml conventions. Points to `docs/`.
- **`README.md`** (repo root) — brief project overview. Links to
  `docs/index.md` for the full documentation index. Should stay short.
