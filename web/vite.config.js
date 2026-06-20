import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, proxy /api to the local FastAPI server (uvicorn app.web_api:app --port 8080)
// so the frontend can call same-origin paths exactly like it will in production
// (where Caddy proxies /api to the api container).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
  },
});
