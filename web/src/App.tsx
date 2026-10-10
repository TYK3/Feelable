// App shell + hash routes. OWNER: Frontend.
import { lazy, Suspense, useEffect } from "react";
import "@fontsource-variable/bricolage-grotesque/wght.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/app.css";
import { navigate, routeKey, useRoute } from "./router";
import HomePage from "./pages/HomePage";
import ReviewPage from "./pages/ReviewPage";
import NotFound from "./pages/NotFound";
import { ToastHost } from "./components/Toast";
import { ErrorBoundary } from "./ErrorBoundary";
import { getEditToken } from "./api";

// Touch mode is a separate, phone-first experience: load it on demand.
const TouchMode = lazy(() => import("./touch/TouchMode"));

export default function App() {
  const route = useRoute();
  const key = routeKey(route);

  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [key]);

  if (route.name === "touch") {
    const canEdit = !!getEditToken(route.id);
    return (
      <ErrorBoundary resetKey={key}>
        <Suspense fallback={<div className="touch-loading" role="status">Opening touch mode…</div>}>
          <TouchMode key={key} jobId={route.id} onExit={canEdit ? () => navigate(`#/job/${route.id}`) : undefined} />
        </Suspense>
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary resetKey={key}>
      <a
        className="skip-link"
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
      {route.name === "home" && <HomePage key={key} />}
      {route.name === "job" && <ReviewPage key={key} id={route.id} urlToken={route.token} />}
      {route.name === "notfound" && <NotFound />}
      <ToastHost />
    </ErrorBoundary>
  );
}
