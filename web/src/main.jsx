import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";
import { ADMIN_TOKEN_KEY } from "./api.js";

// Admin bypass: visiting /?admin=TOKEN once stores the token and removes it from
// the URL (so it's not left in the address bar, history, or shared links). The
// API client then sends it on every request to skip the rate limit.
(function captureAdminToken() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("admin");
  if (token) {
    localStorage.setItem(ADMIN_TOKEN_KEY, token);
    params.delete("admin");
    const qs = params.toString();
    window.history.replaceState(
      {},
      "",
      window.location.pathname + (qs ? `?${qs}` : "") + window.location.hash
    );
  }
})();

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
