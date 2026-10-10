// Error boundary (F03). At the app root it replaces a blank page with a recovery screen; around lazy
// widgets (3D viewer) it keeps the rest of the page usable and offers Retry / Reload.
import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = {
  children: ReactNode;
  /** Render a contained fallback instead of the full-page one. */
  fallback?: (args: { error: Error; retry: () => void }) => ReactNode;
  onRetry?: () => void;
  resetKey?: unknown;
};
type State = { error: Error | null };

export function isChunkError(e: unknown) {
  const m = String((e as Error)?.message ?? e);
  return /dynamically imported module|Importing a module script failed|ChunkLoadError|Loading chunk|error loading dynamically/i.test(m);
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Feelable UI error:", error, info.componentStack);
  }

  componentDidUpdate(prev: Props) {
    if (this.state.error && prev.resetKey !== this.props.resetKey) this.setState({ error: null });
  }

  retry = () => {
    this.props.onRetry?.();
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    if (this.props.fallback) return this.props.fallback({ error, retry: this.retry });
    const chunk = isChunkError(error);
    return (
      <main className="crash" role="alert">
        <div className="crash__card">
          <svg width="56" height="56" viewBox="0 0 56 56" aria-hidden="true">
            <rect width="56" height="56" rx="16" fill="#FFD23F" />
            <circle cx="20" cy="18" r="5" fill="#17171C" />
            <circle cx="20" cy="38" r="5" fill="none" stroke="#17171C" strokeWidth="2.5" />
            <circle cx="36" cy="28" r="5" fill="#17171C" />
          </svg>
          <h1 className="crash__title">{chunk ? "Feelable was updated" : "This screen stopped working"}</h1>
          <p className="crash__msg">
            {chunk
              ? "A newer version is on the server. Reload to get it — your saved lessons and drafts are kept."
              : "Your drafts are saved in this browser. Reload to carry on, or start a new diagram."}
          </p>
          <div className="crash__actions">
            <button type="button" className="btn btn--primary btn--lg" onClick={() => window.location.reload()}>
              <span>Reload</span>
            </button>
            <a className="btn btn--secondary btn--lg" href="#/" onClick={() => this.setState({ error: null })}>
              <span>New diagram</span>
            </a>
          </div>
        </div>
      </main>
    );
  }
}
