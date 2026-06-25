import { useState, useRef, useEffect } from "react";
import { ask, fetchDocument } from "./api.js";

const SUGGESTIONS = [
  "What did Ray Peat think about polyunsaturated fats?",
  "How does thyroid affect metabolism?",
  "Is coffee good or bad according to Peat?",
  "What is the role of CO₂ in the body?",
];

function ConfidenceBadge({ tier }) {
  if (!tier) return null;
  const cls = "badge badge-" + tier.toLowerCase();
  return <span className={cls}>{tier}</span>;
}

// Turn a corpus path like
// "01_Audio_Transcripts\\Other_Interviews\\kmud-141219-you-are-what-you-eat.mp3-transcript_processed.txt"
// into something readable.
function cleanName(path) {
  if (!path) return "source";
  const base = path.split(/[\\/]/).pop();
  return base
    .replace(/-transcript_processed\.txt$/i, "")
    .replace(/_processed\.txt$/i, "")
    .replace(/\.(txt|mp3|pdf)$/i, "")
    .replace(/\.mp3.*$/i, "")
    .trim();
}

function SourceItem({ s, n }) {
  const score = (s.rerank_score ?? s.score ?? 0).toFixed(2);
  const hasText = (s.context && s.context.trim()) || (s.ray_peat_response && s.ray_peat_response.trim());
  const [doc, setDoc] = useState({ status: "idle", text: "" });

  async function loadDoc() {
    if (doc.status === "loading") return;
    setDoc({ status: "loading", text: "" });
    try {
      const d = await fetchDocument(s.source_file);
      setDoc({ status: "loaded", text: d.content });
    } catch (e) {
      setDoc({ status: "error", text: e.message || "Couldn't load document." });
    }
  }

  return (
    <details className="source-item">
      <summary>
        <span className="src-num">S{n}</span>
        <span className="src-file">{cleanName(s.source_file)}</span>
        <span className="src-score">{score}</span>
      </summary>
      <div className="src-body">
        {s.context && s.context.trim() && (
          <p className="src-context">{s.context.trim()}</p>
        )}
        {s.ray_peat_response && s.ray_peat_response.trim() && (
          <blockquote className="src-quote">{s.ray_peat_response.trim()}</blockquote>
        )}
        {!hasText && <p className="src-empty">No excerpt available for this source.</p>}

        <div className="src-doc">
          {doc.status === "idle" && (
            <button className="src-doc-btn" onClick={loadDoc}>
              📄 Read full document
            </button>
          )}
          {doc.status === "loading" && <p className="src-empty">Loading full document…</p>}
          {doc.status === "error" && <p className="src-empty">{doc.text}</p>}
          {doc.status === "loaded" && <pre className="src-doc-text">{doc.text}</pre>}
        </div>
      </div>
    </details>
  );
}

function Sources({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <details className="sources">
      <summary>{sources.length} source{sources.length > 1 ? "s" : ""}</summary>
      <div className="source-list">
        {sources.map((s, i) => (
          <SourceItem key={i} s={s} n={i + 1} />
        ))}
      </div>
    </details>
  );
}

// Clickable follow-up questions. Shown only under the latest answer so old
// suggestions don't pile up. Clicking submits the question as the next query.
function FollowUps({ items, onPick, disabled }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="followups">
      <span className="followups-label">💡 Follow up with</span>
      <div className="followup-list">
        {items.map((q, i) => (
          <button
            key={i}
            className="chip chip-followup"
            onClick={() => onPick(q)}
            disabled={disabled}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function Message({ msg, isLast, onFollowup, busy }) {
  const isUser = msg.role === "user";
  return (
    <div className={`msg ${isUser ? "msg-user" : "msg-assistant"}`}>
      <div className="msg-bubble">
        {!isUser && <ConfidenceBadge tier={msg.confidence} />}
        <div className="msg-content">{msg.content}</div>
        {!isUser && <Sources sources={msg.sources} />}
        {!isUser && isLast && (
          <FollowUps items={msg.followups} onPick={onFollowup} disabled={busy} />
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [remaining, setRemaining] = useState(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text) {
    const query = (text ?? input).trim();
    if (!query || loading) return;

    setError(null);
    setInput("");
    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    const next = [...messages, { role: "user", content: query }];
    setMessages(next);
    setLoading(true);

    try {
      const data = await ask(query, history);
      if (data.remaining !== null && data.remaining !== undefined) {
        setRemaining(data.remaining);
      }
      setMessages([
        ...next,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
          confidence: data.confidence,
          followups: data.followups,
        },
      ]);
    } catch (e) {
      if (e.status === 429) setRemaining(0);
      setError(e.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🧬 PeatLearn</h1>
        <p>Grounded Q&amp;A over Dr. Ray Peat's bioenergetic work</p>
      </header>

      <main className="chat">
        {messages.length === 0 && !loading && (
          <div className="empty">
            <p>Ask anything about metabolism, hormones, nutrition, or health.</p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="chip" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <Message
            key={i}
            msg={m}
            isLast={i === messages.length - 1}
            onFollowup={send}
            busy={loading}
          />
        ))}

        {loading && (
          <div className="msg msg-assistant">
            <div className="msg-bubble">
              <div className="typing">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        {error && <div className="error">⚠️ {error}</div>}
        <div ref={endRef} />
      </main>

      <footer className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a question about Ray Peat's work..."
          rows={1}
          disabled={loading}
        />
        <button onClick={() => send()} disabled={loading || !input.trim()}>
          Send
        </button>
      </footer>

      {remaining !== null && (
        <p className="quota">
          {remaining > 0
            ? `${remaining} question${remaining === 1 ? "" : "s"} left today`
            : "Daily limit reached — resets at midnight UTC"}
        </p>
      )}

      <p className="disclaimer">
        Answers are grounded in Ray Peat's corpus and may be incomplete. Not
        medical advice. PeatLearn is an unofficial, educational project and is
        not affiliated with Ray Peat or his estate.
      </p>
    </div>
  );
}
