import { useState } from "react";
import type { Part, Readiness } from "../types";
import type { PartDraft } from "../lib/drafts";
import { Button } from "./Button";
import { IconMerge, IconPencil, IconSparkle, IconX, IconZoom } from "./Icons";
import { PartRow } from "./PartRow";

export function PartsPanel({
  parts,
  drafts,
  selectedId,
  hoverId,
  canEdit,
  limits,
  readiness,
  sourceUrl,
  sourceSize,
  busy,
  dirty,
  ai,
  onSelect,
  onHover,
  onChange,
  onSplit,
  onMerge,
  onDraw,
  onFixOutline,
  onCloseup,
  onAiDescribe,
  onNewPhoto,
  mergeMode,
  setMergeMode,
}: {
  parts: Part[];
  drafts: Record<string, PartDraft>;
  selectedId: string | null;
  hoverId: string | null;
  canEdit: boolean;
  limits: { name: number; description: number };
  readiness: Readiness;
  sourceUrl: string;
  sourceSize: [number, number];
  busy: boolean;
  dirty: boolean;
  ai: { available: boolean | null; running: boolean; failed: string | null };
  onSelect: (id: string | null) => void;
  onHover: (id: string | null) => void;
  onChange: (id: string, patch: Partial<PartDraft>) => void;
  onSplit: (id: string) => void;
  onMerge: (ids: string[]) => void;
  onDraw?: () => void;
  /** open the draw tool on an existing part to add or trim area */
  onFixOutline?: (id: string) => void;
  onCloseup?: () => void;
  onAiDescribe: () => void;
  onNewPhoto: () => void;
  mergeMode: boolean;
  setMergeMode: (v: boolean) => void;
}) {
  const [picked, setPicked] = useState<string[]>([]);
  const draftOf = (p: Part): PartDraft => drafts[p.id] ?? { name: p.name, description: p.description, include: p.include };
  const included = parts.filter((p) => draftOf(p).include).length;
  const flagged = new Set(readiness.issues.filter((i) => i.part_id).map((i) => i.part_id!));
  const lock = dirty ? "Save or discard your edits first." : null;
  const unmatched = readiness.unmatched_labels;

  if (parts.length === 0) {
    return (
      <section className="pcard" aria-labelledby="parts-title">
        <h2 id="parts-title" className="pcard__title">Parts</h2>
        <div className="empty-parts">
          <p>
            <strong>No parts were found in this picture.</strong> Feelable looks for coloured or outlined shapes. Photograph
            the diagram straight on, in good light, filling the frame — or use a screenshot.
          </p>
          <Button variant="primary" onClick={onNewPhoto}>Try another photo</Button>
        </div>
      </section>
    );
  }

  return (
    <section className="pcard" aria-labelledby="parts-title">
      <div className="pcard__head">
        <div>
          <h2 id="parts-title" className="pcard__title">Parts</h2>
          <p className="pcard__count">
            {included} of {parts.length} in the print{included > 8 && <strong className="pcard__over"> · 8 or fewer reads best</strong>}
          </p>
        </div>
        {canEdit && !mergeMode && (
          <div className="pcard__tools">
            <Button
              variant="secondary"
              size="sm"
              icon={<IconSparkle size={16} />}
              onClick={onAiDescribe}
              busy={ai.running}
              disabled={ai.available !== true || ai.running || busy || !!lock}
              title={lock ?? (ai.available === false ? "AI is not set up on this server" : undefined)}
            >
              {ai.running ? "Describing…" : "Describe with AI"}
            </Button>
            <Button variant="secondary" size="sm" icon={<IconMerge size={16} />} onClick={() => { setPicked([]); setMergeMode(true); }} disabled={busy || !!lock || parts.length < 2} title={lock ?? undefined}>
              Merge
            </Button>
            {onDraw && (
              <Button variant="secondary" size="sm" icon={<IconPencil size={16} />} onClick={onDraw} disabled={busy || !!lock} title={lock ?? "Outline a part the scan missed"}>
                Add missing part
              </Button>
            )}
            {onCloseup && (
              <Button variant="secondary" size="sm" icon={<IconZoom size={16} />} onClick={onCloseup} disabled={busy} title="Crowded? Make an enlarged detail lesson of one area">
                Close-up lesson
              </Button>
            )}
          </div>
        )}
      </div>

      {canEdit && ai.available === false && <p className="pcard__note">AI isn't set up on this server, so descriptions come from the labels. You can write your own.</p>}
      {canEdit && ai.failed && <p className="pcard__note pcard__note--warn" role="alert">{ai.failed}</p>}
      {canEdit && lock && !mergeMode && <p className="pcard__note sr-only">{lock}</p>}

      {mergeMode && (
        <div className="mergebar" role="region" aria-label="Merge parts">
          <p>Tick the parts that are really one thing. The first one you tick keeps its name.</p>
          <div className="mergebar__actions">
            <Button variant="primary" size="sm" disabled={picked.length < 2 || busy} busy={busy} onClick={() => onMerge(picked)}>
              Merge {picked.length >= 2 ? picked.length : ""} parts
            </Button>
            <Button variant="ghost" size="sm" icon={<IconX size={16} />} onClick={() => setMergeMode(false)}>
              Cancel
            </Button>
          </div>
        </div>
      )}

      {unmatched.length > 0 && (
        <datalist id="unmatched-labels">
          {unmatched.map((u) => <option key={u} value={u} />)}
        </datalist>
      )}

      <ul className="plist">
        {parts.map((p) => (
          <PartRow
            key={p.id}
            part={p}
            draft={draftOf(p)}
            selected={selectedId === p.id}
            hovered={hoverId === p.id}
            canEdit={canEdit}
            limits={limits}
            flagged={flagged.has(p.id)}
            merge={mergeMode ? { on: true, checked: picked.includes(p.id), toggle: () => setPicked((xs) => (xs.includes(p.id) ? xs.filter((x) => x !== p.id) : [...xs, p.id])) } : null}
            sourceUrl={sourceUrl}
            sourceSize={sourceSize}
            datalistId={unmatched.length ? "unmatched-labels" : undefined}
            busy={busy}
            lockReason={lock}
            onSelect={onSelect}
            onHover={onHover}
            onChange={(patch) => onChange(p.id, patch)}
            onSplit={() => onSplit(p.id)}
            onFixOutline={onFixOutline && !lock ? () => onFixOutline(p.id) : undefined}
          />
        ))}
      </ul>
    </section>
  );
}
