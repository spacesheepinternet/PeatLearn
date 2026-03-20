# /handoff — Write a Handoff Document

Generate a `HANDOFF.md` at the project root capturing the current state of work.

## Steps

1. Run `git status` to see uncommitted changes.
2. Run `git log --oneline -10` to see recent commits.
3. Run `git diff --stat HEAD` to summarize what has changed.
4. Check `.claude/plans/` for any active plan files.
5. Note any servers currently running (ports 8000, 8001, 8501).

## Write `HANDOFF.md` with these sections:

```markdown
# Handoff — <date>

## Current Branch & Git State
- Branch: <branch>
- Uncommitted changes: <list or "none">
- Last 5 commits: <list>

## What Was Being Worked On
<1–3 paragraph summary of the task in progress>

## Key Decisions Made
- <decision 1 and why>
- <decision 2 and why>

## Active Plan
<link to .claude/plans/<file> or "none">

## What Needs to Happen Next
1. <next action>
2. <next action>

## Open Questions / Blockers
- <question or blocker>

## Files Modified This Session
<list from git diff --stat>
```

After writing `HANDOFF.md`, confirm to the user where it was saved.
