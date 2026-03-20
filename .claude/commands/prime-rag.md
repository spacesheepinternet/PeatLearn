# /prime-rag — RAG Subsystem Overview

Read these files to understand the current state of the RAG subsystem:

1. `.claude/rules/rag.md` — subsystem rules
2. `peatlearn/rag/vector_search.py` — full file
3. `peatlearn/rag/rag_system.py` — full file
4. `app/api.py` — full file

After reading, summarize:
- How `PineconeVectorSearch` works (embedding, search, offline fallback)
- How `PineconeRAG` retrieves, reranks, and generates answers
- All available API endpoints and their signatures
- Any issues or gaps visible in the current implementation
