---
paths:
  - app/dashboard.py
  - scripts/streamlit_dashboard.py
---

# Dashboard Rules

## Entry Points

| File | Notes |
|------|-------|
| `app/dashboard.py` | Primary dashboard — all new work goes here |
| `peatlearn_master.py` | Thin backward-compat launcher that imports `app/dashboard.py` |
| `scripts/streamlit_dashboard.py` | Alternate UI; may lag behind `app/dashboard.py` |

## Dev Mode

- Flag: `--dev` CLI argument OR `PEATLEARN_DEV_MODE=true` in `.env`
- Effect: enables **watchdog** file watcher → Streamlit auto-refreshes on code changes
- Check: `if settings.ENVIRONMENT == "development" or "--dev" in sys.argv`

```bash
streamlit run app/dashboard.py -- --dev
```

## Streamlit Conventions

- Use `st.session_state` for all stateful data (quiz state, user profile, search history).
- Cache expensive calls with `@st.cache_data` (data loads) or `@st.cache_resource` (model loads).
- Never call `asyncio.run()` inside a Streamlit callback — use `asyncio.new_event_loop()` or a sync wrapper.
- Keep sidebar for navigation/settings; main area for content.

## Plotly

- Use `plotly.express` for standard charts, `plotly.graph_objects` only when `px` is insufficient.
- Always set `use_container_width=True` on `st.plotly_chart` calls.
- Dark theme: `template="plotly_dark"` to match Streamlit's dark mode.

## API Communication

The dashboard talks to both backends:
- RAG queries → `http://localhost:8000/api/ask` and `/api/search`
- ML/personalization → `http://localhost:8001/api/*`
- Use `httpx` (async) or `requests` (sync) — never call backend functions directly from the dashboard.

## Do Not
- Do not import `app.api` or `app.advanced_api` directly in the dashboard — always go through HTTP.
- Do not block the Streamlit main thread with long-running operations — use `st.spinner` + background threads.
- Do not store sensitive data (API keys) in `st.session_state`.
