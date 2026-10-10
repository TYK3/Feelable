import type { ReactNode } from "react";
import { Logo } from "./Logo";

export function AppBar({ right }: { right?: ReactNode }) {
  return (
    <header className="appbar">
      <a href="#/" className="appbar__brand" aria-label="Feelable home">
        <Logo size={30} withWordmark />
      </a>
      <div className="appbar__right">{right}</div>
    </header>
  );
}
