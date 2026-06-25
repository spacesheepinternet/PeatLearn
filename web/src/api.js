// API client for the PeatLearn Web API.
//
// In dev, Vite proxies /api -> http://localhost:8080 (see vite.config.js).
// In prod, Caddy proxies /api -> the api container. Either way the frontend
// just calls same-origin "/api/...", so no base URL or CORS config is needed.
// An optional override is supported for unusual deployments.
const BASE = import.meta.env.VITE_API_URL ?? "";

// localStorage key holding the admin token (set via /?admin=TOKEN). When
// present, it's sent as X-Admin-Token so the backend skips the rate limit.
export const ADMIN_TOKEN_KEY = "peatlearn_admin_token";

/**
 * Ask the RAG pipeline a question.
 * @param {string} query
 * @param {{role: string, content: string}[]} chatHistory
 * @returns {Promise<{answer: string, sources: {source_file: string, score: number, rerank_score: number|null}[], confidence: string|null, followups: string[]}>}
 */
export async function ask(query, chatHistory = []) {
  const headers = { "Content-Type": "application/json" };
  const adminToken = localStorage.getItem(ADMIN_TOKEN_KEY);
  if (adminToken) headers["X-Admin-Token"] = adminToken;

  const res = await fetch(`${BASE}/api/ask`, {
    method: "POST",
    headers,
    body: JSON.stringify({ query, chat_history: chatHistory }),
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body — keep the status message */
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }

  const data = await res.json();
  // Daily quota remaining, surfaced by the API as a header (may be absent).
  // Admins get "unlimited" (non-numeric) — treat as null so no quota is shown.
  const remaining = res.headers.get("X-RateLimit-Remaining");
  const n = Number(remaining);
  data.remaining = remaining !== null && Number.isFinite(n) ? n : null;
  return data;
}

/**
 * Fetch the full text of a source document.
 * @param {string} file  the source_file path from a search result
 * @returns {Promise<{file: string, content: string}>}
 */
export async function fetchDocument(file) {
  const res = await fetch(`${BASE}/api/document?file=${encodeURIComponent(file)}`);
  if (!res.ok) {
    let detail = `Couldn't load document (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* keep status message */
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}
