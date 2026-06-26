import { useState, useEffect, useMemo } from "react";
import { fetchConversations, ADMIN_TOKEN_KEY } from "./api.js";

const TIERS = ["ALL", "HIGH", "MEDIUM", "LOW", "ABSTAIN"];

function fmtTime(ts) {
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function ConvCard({ c }) {
  const tier = (c.confidence || "").toUpperCase();
  return (
    <div className="admin-card">
      <div className="admin-card-top">
        {tier ? <span className={"badge badge-" + tier.toLowerCase()}>{tier}</span> : null}
        <span className="admin-time">{fmtTime(c.ts)}</span>
        {c.latency_s != null && <span className="admin-lat">{c.latency_s}s</span>}
      </div>
      <p className="admin-q">{c.question}</p>
      <details className="admin-answer">
        <summary>Answer</summary>
        <div className="admin-answer-body">{c.answer || "(empty)"}</div>
      </details>
      <div className="admin-meta">
        <span title="raw IP">🌐 {c.ip_raw || "—"}</span>
        <span title="hashed IP">#{c.ip_hash || "—"}</span>
        <span>{c.n_sources ?? 0} src</span>
      </div>
      {c.sources && c.sources.length > 0 && (
        <div className="admin-src">
          {c.sources.map((s, i) => (
            <span key={i} className="admin-src-pill">
              {String(s).split(/[\\/]/).pop()}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Admin({ onExit }) {
  const [data, setData] = useState({ stats: null, conversations: [] });
  const [status, setStatus] = useState("loading"); // loading | ok | error | needtoken
  const [error, setError] = useState("");
  const [limit, setLimit] = useState(100);
  const [filter, setFilter] = useState("");
  const [tier, setTier] = useState("ALL");
  const [tokenInput, setTokenInput] = useState("");

  async function load() {
    if (!localStorage.getItem(ADMIN_TOKEN_KEY)) {
      setStatus("needtoken");
      return;
    }
    setStatus("loading");
    try {
      const d = await fetchConversations(limit);
      setData(d);
      setStatus("ok");
    } catch (e) {
      if (e.status === 403) {
        setStatus("needtoken");
        setError(e.message);
      } else {
        setStatus("error");
        setError(e.message || "Failed to load.");
      }
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [limit]);

  function saveToken(e) {
    e.preventDefault();
    const t = tokenInput.trim();
    if (!t) return;
    localStorage.setItem(ADMIN_TOKEN_KEY, t);
    setTokenInput("");
    setError("");
    load();
  }

  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase();
    return (data.conversations || []).filter((c) => {
      if (tier !== "ALL" && (c.confidence || "").toUpperCase() !== tier) return false;
      if (!f) return true;
      return (
        (c.question || "").toLowerCase().includes(f) ||
        (c.answer || "").toLowerCase().includes(f) ||
        (c.ip_raw || "").toLowerCase().includes(f)
      );
    });
  }, [data, filter, tier]);

  return (
    <div className="admin">
      <header className="admin-head">
        <button className="admin-btn" onClick={onExit} aria-label="Back to chat">
          ← Chat
        </button>
        <h1>Conversations</h1>
        <button className="admin-btn" onClick={load} aria-label="Refresh">
          ↻
        </button>
      </header>

      {status === "needtoken" ? (
        <form className="admin-token" onSubmit={saveToken}>
          <p>{error || "Enter your admin token to view logged conversations."}</p>
          <input
            type="password"
            value={tokenInput}
            onChange={(e) => setTokenInput(e.target.value)}
            placeholder="Admin token"
            autoFocus
          />
          <button type="submit" className="admin-btn primary">
            Unlock
          </button>
        </form>
      ) : (
        <>
          {data.stats && (
            <div className="admin-stats">
              <strong>{data.stats.total}</strong> total
              {Object.entries(data.stats.by_confidence || {}).map(([k, v]) => (
                <span key={k} className="admin-stat-pill">
                  {k}: {v}
                </span>
              ))}
            </div>
          )}

          <div className="admin-controls">
            <input
              className="admin-filter"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Search question / answer / IP…"
            />
            <div className="admin-control-row">
              <select value={tier} onChange={(e) => setTier(e.target.value)}>
                {TIERS.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
              <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
                {[50, 100, 250, 500, 1000].map((n) => (
                  <option key={n} value={n}>
                    last {n}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {status === "loading" && <p className="admin-note">Loading…</p>}
          {status === "error" && <p className="admin-note error">⚠️ {error}</p>}
          {status === "ok" && shown.length === 0 && (
            <p className="admin-note">No conversations match.</p>
          )}

          <div className="admin-list">
            {shown.map((c) => (
              <ConvCard key={c.id} c={c} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
