// API client for the PeatLearn Web API.
//
// In dev, Vite proxies /api -> http://localhost:8080 (see vite.config.js).
// In prod, Caddy proxies /api -> the api container. Either way the frontend
// just calls same-origin "/api/...", so no base URL or CORS config is needed.
// An optional override is supported for unusual deployments.
const BASE = import.meta.env.VITE_API_URL ?? "";

/**
 * Ask the RAG pipeline a question.
 * @param {string} query
 * @param {{role: string, content: string}[]} chatHistory
 * @returns {Promise<{answer: string, sources: {source_file: string, score: number, rerank_score: number|null}[], confidence: string|null}>}
 */
export async function ask(query, chatHistory = []) {
  const res = await fetch(`${BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
    throw new Error(detail);
  }

  return res.json();
}
