import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` on :5173 proxies API + files to the FastAPI server on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/files": "http://127.0.0.1:8000",
      "/samples": "http://127.0.0.1:8000",
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
