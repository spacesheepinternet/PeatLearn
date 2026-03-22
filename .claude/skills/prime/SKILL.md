---
name: prime
description: Load full project overview — entry points, key subsystems, and current state
---
# /prime — Full Project Overview

Read the following files to build a complete picture of the current project state:

1. `CLAUDE.md` — global rules and conventions
2. `app/dashboard.py` — main entry point (first 60 lines)
3. `app/api.py` — RAG backend (first 60 lines)
4. `app/advanced_api.py` — ML backend (first 60 lines)
5. `peatlearn/rag/rag_system.py` — core RAG (first 60 lines)
6. `peatlearn/adaptive/quiz_generator.py` — quiz generation (first 40 lines)
7. `peatlearn/personalization/engine.py` — personalization (first 40 lines)
8. `config/settings.py` — full file

After reading, summarize:
- What each service does and its current state
- Any obvious issues or TODOs visible in the code
- The key integration points between subsystems

Do not read deep implementation details unless a specific subsystem is the focus.
