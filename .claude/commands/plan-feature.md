# /plan-feature — Plan a New Feature

**Usage:** `/plan-feature <feature description>`

## Steps

1. **Understand the request** — identify which subsystems are affected (rag, adaptive, personalization, embedding, preprocessing, dashboard).

2. **Research current state** — for each affected subsystem, read:
   - The relevant `.claude/rules/<subsystem>.md`
   - The key implementation files (2–3 files max per subsystem)

3. **Identify integration points** — where does the new feature plug in? What existing classes/functions does it extend or call?

4. **Draft the plan** with these sections:
   - **Goal**: one-sentence description
   - **Affected files**: list of files to create or modify
   - **Implementation steps**: numbered, each ≤ 2 sentences
   - **Tests needed**: unit and/or integration
   - **Risks / open questions**: anything uncertain

5. **Save the plan** to `.claude/plans/<feature-slug>-<YYYY-MM-DD>.md`

6. **Show the plan** to the user and ask for approval before implementing.

## Plan File Format

```markdown
# Plan: <Feature Name>

**Date:** YYYY-MM-DD
**Status:** draft | approved | in-progress | done

## Goal
One sentence.

## Affected Files
- `path/to/file.py` — what changes
- `path/to/new_file.py` — new file, what it does

## Implementation Steps
1. ...
2. ...

## Tests
- Unit: ...
- Integration: ...

## Risks / Open Questions
- ...
```
