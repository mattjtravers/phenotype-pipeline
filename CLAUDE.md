# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML pipeline to predict human phenotype from biological/genetic data.

## Environment

- Python 3.12 (managed via the devcontainer)
- Package manager: **uv** (installed at `/home/vscode/.uv/bin/uv` after devcontainer setup)
- Linter/formatter: **Ruff** (auto-runs on save in VS Code; format on save, organize imports, fix all)

## Common Commands

```bash
# Install dependencies
uv sync

# Run the pipeline / a script
uv run python <script.py>

# Run tests
uv run pytest

# Run a single test
uv run pytest path/to/test_file.py::test_name

# Lint and format
uv run ruff check .
uv run ruff format .
```

## Dev Setup Notes

- The devcontainer (Python 3.12) is the canonical development environment.
- `uv` is installed by the `postCreateCommand` in [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) — use it for all dependency management, not pip.
- Ruff is the sole formatter and linter; do not introduce Black, isort, or flake8.
- The Pydantic AI skills plugin is installed in Claude Code — use `/ai:building-pydantic-ai-agents` when adding ML agent components.

## LID Mode: Full

## Linked-Intent Development (MANDATORY)

**Consult the `linked-intent-dev` skill for ALL code changes.** All changes flow through the arrow of intent in one direction:

```
HLD → LLDs → EARS → Tests → Code
```

- **New features and refactors**: full six-phase workflow (HLD check → LLD check/draft → EARS → intent-narrowing edge audit → tests-first → code).
- **Bug fixes**: walk the arrow like any other change — find where behavior diverged from intent and cascade from there. No short-circuit.
- **If unsure**: use the full workflow.

Stop after each phase for user review. Mutation, not accumulation — docs reflect current intent, not history.

### Navigation

| What you need | Where to look |
|---|---|
| High-level design | `docs/high-level-design.md` |
| Low-level designs | `docs/llds/` |
| EARS specs | `docs/specs/` |

### Terminology

- **HLD**: High-Level Design — single project-level doc at `docs/high-level-design.md`.
- **LLD**: Low-Level Design — detailed component design doc in `docs/llds/`. One per intent component.
- **EARS**: Easy Approach to Requirements Syntax — structured one-line requirements with globally unique IDs in `docs/specs/`. Markers: `[x]` implemented, `[ ]` active gap, `[D]` deferred.
- **Arrow**: the unidirectional chain from vision to code (HLD → LLDs → EARS → Tests → Code). Strictly a DAG of intent.
- **Arrow segment**: the territory owned by one LLD — the LLD itself plus the specs, tests, and code that cite its EARS IDs. Within-segment cascade is free; across-segment cascade pauses.
- **Cascade**: propagating a change downstream through the arrow so adjacent levels stay coherent.

### Code annotations

Annotate code and tests with `@spec` comments citing EARS IDs:

```
# @spec PHENO-001, PHENO-002
```

Place the annotation at the *entry point of the behavior's implementation graph* — the topmost function or module owning the specified behavior, not every helper. When a behavior spans multiple subsystems, annotate at the entry point in each subsystem. Tests follow the same rule: annotate the test that directly exercises the spec, not every inner assertion.
