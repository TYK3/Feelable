// Hero mini demo: a small tactile graphic (original illustration) you can touch, hover or tab through.
// The selected part lifts, its braille key "rises", and its name can be spoken.
import { useCallback, useEffect, useRef, useState } from "react";
import { BrailleWord } from "./BrailleWord";
import { IconSpeaker, IconSpeakerOff } from "./Icons";
import { initSpeech, speak } from "../touch/speech";

type DemoPart = { id: string; key: string; name: string; says: string; tex: string };

const PARTS: DemoPart[] = [
  { id: "petal", key: "p", name: "Petals", says: "Five round petals ring the centre. They feel dotted.", tex: "url(#md-dots)" },
  { id: "centre", key: "c", name: "Centre", says: "The centre of the flower, at the top. It feels criss-crossed.", tex: "url(#md-cross)" },
  { id: "stem", key: "s", name: "Stem", says: "A long smooth stem runs down from the flower to the ground.", tex: "#17171C" },
  { id: "leaf", key: "l", name: "Leaf", says: "One leaf grows to the right of the stem. It has slanted lines.", tex: "url(#md-diag)" },
];

const PETALS = [0, 72, 144, 216, 288];

export function MiniDemo() {
  const [sel, setSel] = useState<string>("petal");
  const [touched, setTouched] = useState(false);
  const [speakOn, setSpeakOn] = useState(false);
  const canSpeak = typeof window !== "undefined" && "speechSynthesis" in window;
  const speakRef = useRef(speakOn);
  speakRef.current = speakOn;

  const choose = useCallback((id: string, fromUser = true) => {
    setSel(id);
    if (fromUser) setTouched(true);
    if (fromUser && speakRef.current && canSpeak) {
      const p = PARTS.find((x) => x.id === id)!;
      initSpeech();
      speak(`${p.name}. ${p.says}`);
    }
  }, [canSpeak]);

  // Gentle auto-tour until someone interacts (skipped with reduced motion).
  useEffect(() => {
    if (touched) return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const t = window.setInterval(() => {
      setSel((s) => PARTS[(PARTS.findIndex((p) => p.id === s) + 1) % PARTS.length].id);
    }, 2600);
    return () => window.clearInterval(t);
  }, [touched]);

  useEffect(() => () => {
    if (canSpeak) window.speechSynthesis.cancel();
  }, [canSpeak]);

  const part = PARTS.find((p) => p.id === sel)!;
  const cls = (id: string) => "md__part" + (sel === id ? " is-on" : "");
  const hover = (id: string) => ({
    onPointerEnter: (e: React.PointerEvent) => e.pointerType === "mouse" && choose(id),
    onPointerDown: () => choose(id),
  });

  return (
    <figure className="md" aria-label="Try it: a tactile flower diagram">
      <div className="md__paper">
        <svg className="md__svg" viewBox="0 0 320 250" role="img" aria-label={`Tactile flower. Selected: ${part.name}`}>
          <defs>
            <pattern id="md-dots" width="9" height="9" patternUnits="userSpaceOnUse">
              <circle cx="4.5" cy="4.5" r="1.9" fill="#17171C" />
            </pattern>
            <pattern id="md-cross" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <path d="M0 0V7M0 0H7" stroke="#17171C" strokeWidth="1.6" />
            </pattern>
            <pattern id="md-diag" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(-40)">
              <path d="M0 0V8" stroke="#17171C" strokeWidth="2" />
            </pattern>
          </defs>

          <g className={cls("stem")} {...hover("stem")}>
            <path className="md__hit" d="M160 112 C 158 160, 166 200, 160 240" />
            <path className="md__line md__line--stem" d="M160 112 C 158 160, 166 200, 160 240" />
          </g>
          <g className={cls("leaf")} {...hover("leaf")}>
            <path className="md__glow" d="M163 196 C 190 160, 238 158, 262 168 C 240 200, 200 214, 163 196 Z" />
            <path className="md__fill" d="M163 196 C 190 160, 238 158, 262 168 C 240 200, 200 214, 163 196 Z" fill="url(#md-diag)" />
            <path className="md__line" d="M163 196 C 190 160, 238 158, 262 168 C 240 200, 200 214, 163 196 Z" />
          </g>
          <g className={cls("petal")} {...hover("petal")}>
            {PETALS.map((a) => (
              <g key={a} transform={`rotate(${a} 160 82)`}>
                <ellipse className="md__glow" cx="160" cy="44" rx="21" ry="29" />
                <ellipse className="md__fill" cx="160" cy="44" rx="21" ry="29" fill="url(#md-dots)" />
                <ellipse className="md__line" cx="160" cy="44" rx="21" ry="29" />
              </g>
            ))}
          </g>
          <g className={cls("centre")} {...hover("centre")}>
            <circle className="md__glow" cx="160" cy="82" r="20" />
            <circle className="md__fill md__fill--solid" cx="160" cy="82" r="20" />
            <circle className="md__fill" cx="160" cy="82" r="20" fill="url(#md-cross)" />
            <circle className="md__line" cx="160" cy="82" r="20" />
          </g>

          {/* braille keys, as on a real tactile graphic */}
          {[
            { k: "p", x: 84, y: 30 },
            { k: "s", x: 128, y: 214 },
            { k: "l", x: 250, y: 204 },
          ].map(({ k, x, y }) => (
            <BrailleSvg key={k} letter={k} x={x} y={y} on={part.key === k} />
          ))}
        </svg>
      </div>

      <figcaption className="md__readout" aria-live="polite">
        <BrailleWord key={part.key} word={part.key} className="md__cell" />
        <span className="md__text">
          <span className="md__name">{part.name}</span>
          <span className="md__says">{part.says}</span>
        </span>
        {canSpeak && (
          <button
            type="button"
            className={speakOn ? "md__speak is-on" : "md__speak"}
            aria-pressed={speakOn}
            onClick={() => {
              const next = !speakOn;
              setSpeakOn(next);
              speakRef.current = next;
              if (next) choose(sel);
              else window.speechSynthesis.cancel();
            }}
          >
            {speakOn ? <IconSpeaker size={18} /> : <IconSpeakerOff size={18} />}
            <span className="sr-only">Speak part names</span>
          </button>
        )}
      </figcaption>

      <div className="md__chips" role="group" aria-label="Parts of the flower">
        {PARTS.map((p) => (
          <button
            key={p.id}
            type="button"
            className={sel === p.id ? "md__chip is-on" : "md__chip"}
            aria-pressed={sel === p.id}
            onClick={() => choose(p.id)}
            onFocus={() => choose(p.id, false)}
          >
            {p.name}
          </button>
        ))}
      </div>
    </figure>
  );
}

/** One braille cell drawn in SVG (dots 1-6). */
function BrailleSvg({ letter, x, y, on }: { letter: string; x: number; y: number; on: boolean }) {
  const map: Record<string, number[]> = { p: [1, 2, 3, 4], s: [2, 3, 4], l: [1, 2, 3], c: [1, 4] };
  const dots = map[letter] ?? [];
  const pos: Record<number, [number, number]> = { 1: [0, 0], 2: [0, 7], 3: [0, 14], 4: [7, 0], 5: [7, 7], 6: [7, 14] };
  return (
    <g className={on ? "md__key is-on" : "md__key"} transform={`translate(${x} ${y})`} aria-hidden="true">
      {dots.map((d) => (
        <circle key={d} cx={pos[d][0]} cy={pos[d][1]} r="2.6" />
      ))}
    </g>
  );
}
