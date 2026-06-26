import { useState, useRef, useEffect, useId, Children } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ask, fetchDocument } from "./api.js";
import Admin from "./Admin.jsx";
import Privacy from "./Privacy.jsx";

const SUGGESTIONS = [
  "What did Ray Peat think about polyunsaturated fats?",
  "How does thyroid affect metabolism?",
  "Is coffee good or bad according to Peat?",
  "What is the role of CO₂ in the body?",
];

/* Sol — a medieval/alchemical sun with a face and alternating rays. */
function Sol({ className }) {
  const id = useId();
  const rays = [];
  const N = 16;
  for (let i = 0; i < N; i++) {
    const a = (i * 2 * Math.PI) / N;
    const tipR = i % 2 === 0 ? 23.5 : 20;
    const bw = 0.17;
    const bx1 = (24 + Math.cos(a - bw) * 15.5).toFixed(1);
    const by1 = (24 + Math.sin(a - bw) * 15.5).toFixed(1);
    const bx2 = (24 + Math.cos(a + bw) * 15.5).toFixed(1);
    const by2 = (24 + Math.sin(a + bw) * 15.5).toFixed(1);
    const tx = (24 + Math.cos(a) * tipR).toFixed(1);
    const ty = (24 + Math.sin(a) * tipR).toFixed(1);
    rays.push(<polygon key={i} points={`${bx1},${by1} ${tx},${ty} ${bx2},${by2}`} />);
  }
  return (
    <svg className={className} viewBox="0 0 48 48" aria-hidden="true">
      <defs>
        <radialGradient id={`${id}d`} cx="50%" cy="45%" r="60%">
          <stop offset="0%" stopColor="#fce6ad" />
          <stop offset="55%" stopColor="#e3982f" />
          <stop offset="100%" stopColor="#d2622c" />
        </radialGradient>
      </defs>
      <g fill="#e08a2e">{rays}</g>
      <circle cx="24" cy="24" r="15" fill={`url(#${id}d)`} />
      <g fill="#8a3f17" opacity="0.9">
        <ellipse cx="20.3" cy="22.3" rx="1.15" ry="1.5" />
        <ellipse cx="27.7" cy="22.3" rx="1.15" ry="1.5" />
      </g>
      <g stroke="#8a3f17" strokeWidth="1.1" fill="none" strokeLinecap="round" opacity="0.85">
        <path d="M24 23.4 v2.3" />
        <path d="M20.6 28.2 q3.4 3 6.8 0" />
      </g>
    </svg>
  );
}

/* Luna — a medieval crescent moon with a profile face and stars. */
function Luna({ className }) {
  const id = useId();
  return (
    <svg className={className} viewBox="0 0 48 48" aria-hidden="true">
      <defs>
        <radialGradient id={`${id}g`} cx="40%" cy="40%" r="70%">
          <stop offset="0%" stopColor="#f6d8a0" />
          <stop offset="60%" stopColor="#e0a04a" />
          <stop offset="100%" stopColor="#c9772f" />
        </radialGradient>
        <mask id={`${id}m`}>
          <rect width="48" height="48" fill="black" />
          <circle cx="22" cy="24" r="15" fill="white" />
          <circle cx="31" cy="20" r="13.5" fill="black" />
        </mask>
      </defs>
      <g fill="#e0a04a">
        <path d="M34 12.5 l0.8 2 2 0.8 -2 0.8 -0.8 2 -0.8 -2 -2 -0.8 2 -0.8z" />
        <path d="M39.5 21.5 l0.6 1.5 1.5 0.6 -1.5 0.6 -0.6 1.5 -0.6 -1.5 -1.5 -0.6 1.5 -0.6z" />
      </g>
      <circle cx="22" cy="24" r="15" fill={`url(#${id}g)`} mask={`url(#${id}m)`} />
      <circle cx="16.6" cy="20.6" r="1.15" fill="#6e3414" opacity="0.8" />
      <path
        d="M15 27.4 q2.4 2 4.6 0.3"
        fill="none"
        stroke="#6e3414"
        strokeWidth="1.1"
        strokeLinecap="round"
        opacity="0.75"
      />
    </svg>
  );
}

function BrandMark({ theme, className }) {
  return theme === "dark" ? <Luna className={className} /> : <Sol className={className} />;
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

// Reveal `text` progressively (typewriter) when `animate` is true. Respects
// reduced-motion. Calls onTick each step so the view can keep scrolling.
function useTyped(text, animate, onTick) {
  const reduce =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [len, setLen] = useState(animate && !reduce ? 0 : text.length);
  useEffect(() => {
    if (!animate || reduce) {
      setLen(text.length);
      return;
    }
    const total = text.length;
    const chunk = Math.max(3, Math.ceil(total / 45)); // ~45 steps -> ~1.3s
    let i = 0;
    const timer = setInterval(() => {
      i += chunk;
      if (i >= total) {
        setLen(total);
        clearInterval(timer);
      } else {
        setLen(i);
      }
      onTick && onTick();
    }, 28);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, animate]);
  return len;
}

function Answer({ text, animate, onTick }) {
  const len = useTyped(text, animate, onTick);
  let shown = text.slice(0, len);
  // Hide half-revealed markdown/citation tokens so they don't flicker as raw
  // characters (e.g. a dangling "**" or "[S1").
  const open = (shown.match(/\[/g) || []).length;
  const close = (shown.match(/\]/g) || []).length;
  if (open > close) shown = shown.slice(0, shown.lastIndexOf("["));
  if (((shown.match(/\*\*/g) || []).length) % 2 === 1) {
    shown = shown.slice(0, shown.lastIndexOf("**"));
  }
  return (
    <div className={"answer" + (len < text.length ? " is-typing" : "")}>
      <Markdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
        {shown}
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

function Message({ msg, isLast, onFollowup, busy, onTick }) {
  const isUser = msg.role === "user";
  return (
    <div className={`msg ${isUser ? "msg-user" : "msg-assistant"}`}>
      <div className="msg-bubble">
        {!isUser && <ConfidenceBadge tier={msg.confidence} />}
        {isUser ? (
          <div className="msg-content">{msg.content}</div>
        ) : (
          <Answer text={msg.content} animate={isLast} onTick={onTick} />
        )}
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
  const [sending, setSending] = useState(false);
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("peatlearn_theme") || "light";
    } catch {
      return "light";
    }
  });
  const [route, setRoute] = useState(() => window.location.hash.replace(/^#/, ""));
  const endRef = useRef(null);
  const taRef = useRef(null);

  useEffect(() => {
    const onHash = () => setRoute(window.location.hash.replace(/^#/, ""));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Auto-grow the composer so typed text is never clipped.
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, [input]);

  const scrollToEnd = () => endRef.current?.scrollIntoView({ behavior: "auto" });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("peatlearn_theme", theme);
    } catch {
      /* ignore */
    }
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#1a0f0b" : "#fbf4e9");
  }, [theme]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

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

  // Send via the arrow button — plays the launch animation on the icon.
  function handleSendClick() {
    if (loading || !input.trim()) return;
    setSending(true);
    setTimeout(() => setSending(false), 500);
    send();
  }

  if (route === "admin") {
    return <Admin onExit={() => (window.location.hash = "")} />;
  }
  if (route === "privacy") {
    return <Privacy onExit={() => (window.location.hash = "")} />;
  }

  return (
    <div className="app">
      <header className="header">
        <button
          className="brand-toggle"
          onClick={toggleTheme}
          aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          title={theme === "dark" ? "Switch to day (Sol)" : "Switch to night (Luna)"}
        >
          <BrandMark theme={theme} className="brand-mark" />
        </button>
        <div className="brand-text">
          <h1>PeatLearn</h1>
          <span className="tag">Dr. Ray Peat's bioenergetics, grounded in his own words</span>
        </div>
      </header>

      <main className="chat">
        {messages.length === 0 && !loading && (
          <div className="empty">
            <BrandMark theme={theme} className="empty-glyph" />
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
            onTick={scrollToEnd}
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
          ref={taRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask Ray Peat anything…"
          rows={1}
          disabled={loading}
        />
        <button
          className={sending ? "sending" : ""}
          onClick={handleSendClick}
          disabled={loading || !input.trim()}
          aria-label="Send"
        >
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
        unofficial, educational project, not affiliated with Ray Peat or his estate. Questions may be
        logged to monitor and improve answer quality —{" "}
        <a href="#privacy">Privacy</a>.
      </p>
    </div>
  );
}
