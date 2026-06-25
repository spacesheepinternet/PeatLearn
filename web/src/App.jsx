import { useState, useRef, useEffect, Children } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ask, fetchDocument } from "./api.js";

const SUGGESTIONS = [
  "What did Ray Peat think about polyunsaturated fats?",
  "How does thyroid affect metabolism?",
  "Is coffee good or bad according to Peat?",
  "What is the role of CO₂ in the body?",
];

// Warm incandescent "sun / filament" mark — the brand glyph. No blue.
function SunMark({ className }) {
  return (
    <svg className={className} viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <defs>
        <radialGradient id="sun" cx="50%" cy="45%" r="55%">
          <stop offset="0%" stopColor="#fbe3a8" />
          <stop offset="55%" stopColor="#e3982f" />
          <stop offset="100%" stopColor="#d2622c" />
        </radialGradient>
      </defs>
      <g stroke="#d2622c" strokeWidth="2.2" strokeLinecap="round">
        {Array.from({ length: 12 }).map((_, i) => {
          const a = (i * Math.PI) / 6;
          const x = 24 + Math.cos(a) * 20;
          const y = 24 + Math.sin(a) * 20;
          const x2 = 24 + Math.cos(a) * 23.5;
          const y2 = 24 + Math.sin(a) * 23.5;
          return <line key={i} x1={x} y1={y} x2={x2} y2={y2} opacity="0.85" />;
        })}
      </g>
      <circle cx="24" cy="24" r="13" fill="url(#sun)" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 19V5" />
      <path d="M5 12l7-7 7 7" />
    </svg>
  );
}

function ConfidenceBadge({ tier }) {
  if (!tier) return null;
  return <span className={"badge badge-" + tier.toLowerCase()}>{tier}</span>;
}

// Wrap [S1] / [S1, S5] citation tokens in styled pills, leaving other inline
// nodes (bold, links) untouched.
const CITE_RE = /\[S\d+(?:\s*,\s*S\d+)*\]/g;
function withCitations(children) {
  return Children.map(children, (child) => {
    if (typeof child !== "string") return child;
    const out = [];
    let last = 0;
    let m;
    CITE_RE.lastIndex = 0;
    while ((m = CITE_RE.exec(child))) {
      if (m.index > last) out.push(child.slice(last, m.index));
      out.push(
        <sup className="cite" key={m.index}>
          {m[0]}
        </sup>
      );
      last = m.index + m[0].length;
    }
    if (out.length === 0) return child;
    if (last < child.length) out.push(child.slice(last));
    return out;
  });
}

const MD_COMPONENTS = {
  p: ({ children }) => <p>{withCitations(children)}</p>,
  li: ({ children }) => <li>{withCitations(children)}</li>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
};

function Answer({ text }) {
  return (
    <div className="answer">
      <Markdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
        {text}
      </Markdown>
    </div>
  );
}

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
  const hasText =
    (s.context && s.context.trim()) || (s.ray_peat_response && s.ray_peat_response.trim());
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
        {s.context && s.context.trim() && <p className="src-context">{s.context.trim()}</p>}
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
      <summary>
        {sources.length} source{sources.length > 1 ? "s" : ""}
      </summary>
      <div className="source-list">
        {sources.map((s, i) => (
          <SourceItem key={i} s={s} n={i + 1} />
        ))}
      </div>
    </details>
  );
}

function FollowUps({ items, onPick, disabled }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="followups">
      <span className="followups-label">Continue exploring</span>
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
        {isUser ? <div className="msg-content">{msg.content}</div> : <Answer text={msg.content} />}
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
        <SunMark className="brand-mark" />
        <div className="brand-text">
          <h1>PeatLearn</h1>
          <span className="tag">Dr. Ray Peat's bioenergetics, grounded in his own words</span>
        </div>
      </header>

      <main className="chat">
        {messages.length === 0 && !loading && (
          <div className="empty">
            <SunMark className="empty-glyph" />
            <h2>Ask about metabolism, hormones &amp; health</h2>
            <p>
              Every answer is drawn from Ray Peat's interviews, articles, and newsletters — and
              cited so you can check his actual words.
            </p>
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
          placeholder="Ask a question about Ray Peat's work…"
          rows={1}
          disabled={loading}
        />
        <button onClick={() => send()} disabled={loading || !input.trim()} aria-label="Send">
          <SendIcon />
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
        Answers reflect Ray Peat's views and may be incomplete. Not medical advice. PeatLearn is an
        unofficial, educational project, not affiliated with Ray Peat or his estate.
      </p>
    </div>
  );
}
