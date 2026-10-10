// Dev-only harness: renders TouchMode alone. Not imported by the app.
// Usage: build with a vite config whose input is src/touch/dev-harness.html, open ?job=<id>.
import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@fontsource/inter/800.css";
import "@fontsource/inter/900.css";
import TouchMode from "./TouchMode";

const jobId = new URLSearchParams(window.location.search).get("job") || "mock";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <TouchMode jobId={jobId} onExit={() => ((window as unknown as { __feelableExited?: boolean }).__feelableExited = true)} />
  </React.StrictMode>,
);
