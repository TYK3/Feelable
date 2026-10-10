// Minimal global toast: call toast("Link copied") from anywhere; <ToastHost/> lives in App.
import { useEffect, useState } from "react";
import { IconCheck, IconWarn } from "./Icons";

type Tone = "ok" | "warn";
type ToastMsg = { id: number; text: string; tone: Tone };
let listeners: ((t: ToastMsg) => void)[] = [];
let nextId = 1;

export function toast(text: string, tone: Tone = "ok") {
  const t = { id: nextId++, text, tone };
  listeners.forEach((l) => l(t));
}

export function ToastHost() {
  const [items, setItems] = useState<ToastMsg[]>([]);
  useEffect(() => {
    const l = (t: ToastMsg) => {
      setItems((xs) => [...xs.slice(-2), t]);
      window.setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== t.id)), 2800);
    };
    listeners.push(l);
    return () => {
      listeners = listeners.filter((x) => x !== l);
    };
  }, []);
  return (
    <div className="toasts" role="status" aria-live="polite">
      {items.map((t) => (
        <div key={t.id} className={`toast toast--${t.tone}`}>
          {t.tone === "ok" ? <IconCheck size={18} /> : <IconWarn size={18} />}
          {t.text}
        </div>
      ))}
    </div>
  );
}
