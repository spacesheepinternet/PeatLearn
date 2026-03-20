# /prime-ml — Advanced ML Subsystem Overview

Read these files to understand the current state of the ML/personalization subsystem:

1. `.claude/rules/personalization.md` — subsystem rules
2. `peatlearn/personalization/engine.py` — full file
3. `peatlearn/personalization/neural.py` — full file
4. `peatlearn/personalization/rl_agent.py` — full file
5. `peatlearn/personalization/knowledge_graph.py` — first 60 lines
6. `peatlearn/recommendation/mf_trainer.py` — full file
7. `app/advanced_api.py` — full file

After reading, summarize:
- Which ML components are fully implemented vs. stubs
- How `ADVANCED_ML_AVAILABLE` guards optional heavy dependencies
- The MF recommender's current state (trained / untrained, dimensions)
- All available API endpoints on port 8001
- Any issues or gaps visible in the current implementation
