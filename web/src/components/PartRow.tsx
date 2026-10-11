import { useEffect, useRef } from "react";
import type { Part } from "../types";
import type { PartDraft } from "../lib/drafts";
import { isUncertain, whyText } from "../lib/readiness";
import { BrailleWord } from "./BrailleWord";
import { Switch } from "./Controls";
import { IconChevronDown, IconSparkle, IconSplit, IconWarn } from "./Icons";
import { polysToPath, rgb } from "./PartOverlay";
import { TextureSwatch, textureName } from "./TextureSwatch";
import { sentenceCase } from "./text";

export type { PartDraft };

export function PartRow({
  part,
  draft,
  selected,
  hovered,
  canEdit,
  limits,
  flagged,
  merge,
  sourceUrl,
  sourceSize,
  datalistId,
  busy,
  lockReason,
  onSelect,
  onHover,
  onChange,
  onSplit,
  onFixOutline,
}: {
  part: Part;
  draft: PartDraft;
  selected: boolean;
  hovered: boolean;
  canEdit: boolean;
  limits: { name: number; description: number };
  flagged: boolean;
  merge: { on: boolean; checked: boolean; toggle: () => void } | null;
  sourceUrl: string;
  sourceSize: [number, number];
  datalistId?: string;
  busy: boolean;
  lockReason: string | null;
  onSelect: (id: string | null) => void;
  onHover: (id: string | null) => void;
  onChange: (patch: Partial<PartDraft>) => void;
  onSplit: () => void;
  onFixOutline?: () => void;
}) {
  const rowRef = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (selected) rowRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selected]);

  const changed = draft.name !== part.name || draft.description !== part.description || draft.include !== part.include;
  const descFromAi = part.description_source === "ai" && draft.description === part.description;
  const name = sentenceCase(draft.name) || "Unnamed part";
  const key = part.key || "";
  const brailleKey = key.replace(/[^a-z]/gi, "").slice(0, 2);
  const nameId = `name-${part.id}`;
  const descId = `desc-${part.id}`;

  return (
    <li
      ref={rowRef}
      className={["prow", selected && "is-sel", hovered && "is-hover", !draft.include && "is-off", changed && "is-changed"].filter(Boolean).join(" ")}
      onMouseEnter={() => onHover(part.id)}
      onMouseLeave={() => onHover(null)}
      data-part={part.id}
    >
      <div className="prow__head">
        {merge?.on && (
          <input
            type="checkbox"
            className="prow__check"
            checked={merge.checked}
            onChange={merge.toggle}
            aria-label={`Merge ${name}`}
          />
        )}
        <button
          type="button"
          className="prow__main"
          aria-expanded={selected}
          aria-controls={`pbody-${part.id}`}
          onClick={() => (merge?.on ? merge.toggle() : onSelect(selected ? null : part.id))}
        >
          <span className="prow__key" style={{ ["--c" as string]: rgb(part.color) }}>
            {brailleKey ? <BrailleWord word={brailleKey} className="prow__braille" /> : null}
            <span className="prow__letter">{key || "–"}</span>
          </span>
          <span className="prow__name">
            {name}
            {flagged && (
              <span className="prow__flag" title="Needs a check">
                <IconWarn size={14} strokeWidth={2.5} />
                <span className="sr-only"> (needs a check)</span>
              </span>
            )}
          </span>
          <span className="prow__meta">
            {draft.include ? (
              <span className="tex" title={`Texture: ${textureName(part.texture)}`}>
                <TextureSwatch texture={part.texture} />
                <span className="tex__label">{textureName(part.texture)}</span>
              </span>
            ) : (
              <span className="tag tag--off">Removed</span>
            )}
            {part.count > 1 && <span className="tag" title={`${part.count} separate shapes`}>×{part.count}</span>}
            {changed && <span className="tag tag--edit">Edited</span>}
          </span>
          <IconChevronDown size={18} className="prow__chev" />
        </button>
        {canEdit && !merge?.on && (
          <Switch
            label={draft.include ? `${name}: in the print` : `${name}: left out`}
            hideLabel
            checked={draft.include}
            disabled={busy}
            onChange={(v) => onChange({ include: v })}
          />
        )}
      </div>

      {selected && !merge?.on && (
        <div className="prow__body" id={`pbody-${part.id}`}>
          <div className="prow__context">
            <Crop part={part} url={sourceUrl} size={sourceSize} />
            <p className={isUncertain(part.match_method) ? "why is-unsure" : "why"}>
              <span className="why__label">Why this name</span>
              {whyText(part.match_method, part.label_text)}
            </p>
          </div>

          {canEdit ? (
            <>
              <Field id={nameId} label="Name the student hears" value={draft.name} max={limits.name}>
                <input
                  id={nameId}
                  className="input"
                  value={draft.name}
                  maxLength={limits.name}
                  list={datalistId}
                  disabled={busy}
                  autoComplete="off"
                  onChange={(e) => onChange({ name: e.target.value })}
                />
              </Field>
              <Field
                id={descId}
                label="Description"
                value={draft.description}
                max={limits.description}
                badge={descFromAi ? <AiBadge /> : null}
              >
                <textarea
                  id={descId}
                  className="input input--area"
                  value={draft.description}
                  rows={3}
                  maxLength={limits.description}
                  disabled={busy}
                  placeholder="One or two sentences: what it is and where it is."
                  onChange={(e) => onChange({ description: e.target.value })}
                />
              </Field>
              <div className="prow__remove">
                {draft.include ? (
                  <>
                    {onFixOutline && (
                      <button type="button" className="btn btn--secondary btn--sm" onClick={onFixOutline} disabled={busy}>
                        Fix outline
                      </button>
                    )}
                    <button type="button" className="btn btn--ghost btn--sm btn--danger" onClick={() => onChange({ include: false })} disabled={busy}>
                      Remove part
                    </button>
                    <span className="prow__hint">Wrong shape or not needed? It leaves the student view, Braille key, audio and downloads. You can restore it.</span>
                  </>
                ) : (
                  <>
                    <span className="prow__hint" role="status">
                      Removed{part.include ? " (not saved yet)" : ""}. The student won't find this part.
                    </span>
                    <button type="button" className="btn btn--secondary btn--sm" onClick={() => onChange({ include: true })} disabled={busy}>
                      {part.include ? "Undo" : "Restore part"}
                    </button>
                  </>
                )}
              </div>
              {part.count > 1 && (
                <div className="prow__split">
                  <p>
                    This part is {part.count} separate shapes. If they are different things, split them into their own parts.
                  </p>
                  <button type="button" className="btn btn--secondary btn--sm" onClick={onSplit} disabled={busy || !!lockReason} title={lockReason ?? undefined}>
                    <IconSplit size={16} />
                    <span>Split into {part.count}</span>
                  </button>
                  {lockReason && <span className="prow__lock">{lockReason}</span>}
                </div>
              )}
            </>
          ) : (
            <div className="prow__read">
              <p className="prow__desc">
                {descFromAi && <AiBadge />} {part.description || <em>No description.</em>}
              </p>
            </div>
          )}
        </div>
      )}
    </li>
  );
}

export function AiBadge() {
  return (
    <span className="ai-badge" title="Written by AI. Check it before teaching.">
      <IconSparkle size={12} /> AI
      <span className="sr-only"> written — check before teaching</span>
    </span>
  );
}

function Field({ id, label, value, max, badge, children }: { id: string; label: string; value: string; max: number; badge?: React.ReactNode; children: React.ReactNode }) {
  const n = value.length;
  const near = n > max * 0.85;
  return (
    <div className="pfield">
      <div className="pfield__top">
        <label htmlFor={id} className="pfield__label">
          {label} {badge}
        </label>
        <span className={near ? "pfield__count is-near" : "pfield__count"} aria-live={near ? "polite" : undefined}>
          <span className="tabular">{n}</span>/{max}
          <span className="sr-only"> characters</span>
        </span>
      </div>
      {children}
    </div>
  );
}

/** A close-up of the part in the original photo, outlined (keeps context visible on phones). */
function Crop({ part, url, size }: { part: Part; url: string; size: [number, number] }) {
  const [x0, y0, x1, y1] = part.source_bbox;
  const pad = Math.max(24, (x1 - x0 + y1 - y0) * 0.18);
  const vb = [x0 - pad, y0 - pad, x1 - x0 + pad * 2, y1 - y0 + pad * 2];
  return (
    <svg className="crop" viewBox={vb.join(" ")} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <image href={url} x={0} y={0} width={size[0]} height={size[1]} />
      <path d={polysToPath(part.source_polygons)} fillRule="evenodd" className="crop__shape" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
