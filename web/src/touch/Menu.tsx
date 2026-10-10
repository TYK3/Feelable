import { useEffect, useRef, useState } from "react";
import type { Part } from "../types";
import { currentVoiceName, englishVoices, initSpeech, naturalClipCount, naturalSample, naturalVoiceEnabled, onVoicesChanged, setNaturalVoice, setVoiceByName, speak } from "./speech";

export interface TouchSettings {
  rate: number;
  showHighlights: boolean;
  vibration: boolean;
}

interface MenuProps {
  parts: Part[];
  settings: TouchSettings;
  canVibrate: boolean;
  finding: Part | null;
  startOnFind?: boolean;
  onChange(s: TouchSettings): void;
  onClose(): void;
  onExit(): void;
  onList(): void;
  onFind(part: Part): void;
  onStopFind(): void;
}

function VoicePicker({ rate }: { rate: number }) {
  const [, force] = useState(0);
  useEffect(() => {
    initSpeech();
    return onVoicesChanged(() => force((n) => n + 1));
  }, []);
  const voices = englishVoices();
  const natural = naturalClipCount() > 0;
  if (!voices.length && !natural) return null;
  const cur = currentVoiceName() ?? "";
  const mode: "natural" | "device" = natural && naturalVoiceEnabled() ? "natural" : "device";
  const DEVICE_LINE = "Hello. This is how I will read the diagram to you.";
  // UX10: Test plays the voice that is actually active — a recorded natural clip, or the device voice.
  const test = () => speak(mode === "natural" ? naturalSample() ?? DEVICE_LINE : DEVICE_LINE, { rate });
  const setMode = (m: "natural" | "device") => {
    setNaturalVoice(m === "natural");
    force((n) => n + 1);
  };
  return (
    <div className="ts-t-setting">
      <span className="ts-t-setting-title" id="ts-t-voice-title">
        Voice
      </span>
      {natural && (
        <div className="ts-t-seg" role="radiogroup" aria-labelledby="ts-t-voice-title">
          {(["natural", "device"] as const).map((m) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={mode === m}
              className={mode === m ? "ts-t-seg-btn is-on" : "ts-t-seg-btn"}
              onClick={() => setMode(m)}
            >
              {m === "natural" ? "Natural voice" : "Device voice"}
            </button>
          ))}
        </div>
      )}
      {natural && mode === "natural" && (
        <p className="ts-t-btn-sub">Part names and descriptions use a recorded natural voice. Other phrases use the device voice below.</p>
      )}
      {voices.length > 0 && (
        <>
          <label htmlFor="ts-t-voice" className="ts-t-setting-sub">
            Device voice
          </label>
          <select
            id="ts-t-voice"
            className="ts-t-select"
            value={cur}
            onChange={(e) => {
              setVoiceByName(e.target.value || null);
              speak(DEVICE_LINE, { rate });
            }}
          >
            {voices.slice(0, 30).map((v, i) => (
              <option key={v.name} value={v.name}>
                {i === 0 ? "★ " : ""}
                {v.name.replace(/^Microsoft /, "").replace(/ - English.*$/, "")} ({v.lang})
              </option>
            ))}
          </select>
        </>
      )}
      <button type="button" className="ts-t-btn" style={{ minHeight: 44 }} onClick={test}>
        Test {mode === "natural" ? "natural" : "device"} voice
      </button>
    </div>
  );
}

export function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

export default function Menu(props: MenuProps) {
  const { settings, onChange } = props;
  const [view, setView] = useState<"main" | "find">(props.startOnFind ? "find" : "main");
  const firstRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    firstRef.current?.focus();
  }, [view]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      if (view === "find") setView("main");
      else props.onClose();
      return;
    }
    if (e.key === "Tab" && dialogRef.current) {
      // keep focus inside the dialog
      const items = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>("button:not([disabled]), input:not([disabled])"),
      );
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  };

  return (
    <div className="ts-t-scrim" onKeyDown={onKeyDown}>
      <div
        ref={dialogRef}
        className="ts-t-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="ts-t-menu-title"
      >
        {view === "main" ? (
          <>
            <div className="ts-t-sheet-head">
              <h2 id="ts-t-menu-title">Menu</h2>
              <button ref={firstRef} className="ts-t-btn ts-t-btn-primary" onClick={props.onClose}>
                Back to diagram
              </button>
            </div>
            <div className="ts-t-menu-grid">
              <button className="ts-t-btn ts-t-btn-big" onClick={props.onList}>
                List parts
                <span className="ts-t-btn-sub">Big buttons, works with screen readers</span>
              </button>
              {props.finding ? (
                <button className="ts-t-btn ts-t-btn-big" onClick={props.onStopFind}>
                  Stop finding {props.finding.name}
                  <span className="ts-t-btn-sub">Or tap with three fingers</span>
                </button>
              ) : (
                <button className="ts-t-btn ts-t-btn-big" onClick={() => setView("find")}>
                  Find a part…
                  <span className="ts-t-btn-sub">Beeps get faster as you get closer</span>
                </button>
              )}
            </div>

            <div className="ts-t-setting">
              <label htmlFor="ts-t-rate">
                Speech speed <strong>{settings.rate.toFixed(1)}×</strong>
              </label>
              <input
                id="ts-t-rate"
                type="range"
                min={0.7}
                max={1.5}
                step={0.1}
                value={settings.rate}
                onChange={(e) => onChange({ ...settings, rate: Number(e.target.value) })}
              />
            </div>
            <VoicePicker rate={settings.rate} />
            <Toggle
              label="Show highlights"
              hint="Yellow shading for sighted helpers"
              on={settings.showHighlights}
              onToggle={() => onChange({ ...settings, showHighlights: !settings.showHighlights })}
            />
            <Toggle
              label="Vibration"
              hint={props.canVibrate ? "Buzz on outlines and parts" : "Not available on this device"}
              on={settings.vibration && props.canVibrate}
              disabled={!props.canVibrate}
              onToggle={() => onChange({ ...settings, vibration: !settings.vibration })}
            />
            <button className="ts-t-btn ts-t-btn-exit" onClick={props.onExit}>
              Exit touch mode
            </button>
          </>
        ) : (
          <>
            <div className="ts-t-sheet-head">
              <h2 id="ts-t-menu-title">Find a part</h2>
              <button ref={firstRef} className="ts-t-btn" onClick={() => setView("main")}>
                Back
              </button>
            </div>
            <ul className="ts-t-list" role="list">
              {props.parts.map((part) => (
                <li key={part.id}>
                  <button className="ts-t-btn ts-t-btn-big ts-t-part" onClick={() => props.onFind(part)}>
                    <span>Find the {part.name}</span>
                    {part.key && <span className="ts-t-key" aria-label={`braille key ${part.key}`}>{part.key}</span>}
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}

function Toggle(props: { label: string; hint: string; on: boolean; disabled?: boolean; onToggle(): void }) {
  return (
    <button
      className="ts-t-toggle"
      role="switch"
      aria-checked={props.on}
      disabled={props.disabled}
      onClick={props.onToggle}
    >
      <span className="ts-t-toggle-text">
        <span className="ts-t-toggle-label">{props.label}</span>
        <span className="ts-t-btn-sub">{props.hint}</span>
      </span>
      <span className="ts-t-switch" aria-hidden="true">
        <span className="ts-t-knob" />
      </span>
    </button>
  );
}
