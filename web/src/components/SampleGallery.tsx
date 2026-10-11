// Prepared examples (precision audit): ten FEATURED lessons - five basic, five complex, each from a
// different subject - with a Basic / Complex filter, then an extended library of earlier examples
// that were not re-reviewed after the latest engine changes (said plainly on each card).
import { useState } from "react";
import type { Sample } from "../types";
import { IconArrowRight } from "./Icons";
import { sentenceCase } from "./text";

const KIND_LABEL: Record<string, string> = {
  colour: "Colour",
  lineart: "Line art",
  chart: "Chart",
  photo: "Photo",
  anatomy: "Dense anatomy",
};
type Filter = "all" | "simple" | "complex";

function Card({ s, onPick }: { s: Sample; onPick: (s: Sample) => void }) {
  return (
    <li>
      <button type="button" className={`sample${s.group === "complex" ? " sample--complex" : ""}`} onClick={() => onPick(s)}>
        <span className="sample__thumb">
          <img src={s.thumb_url} alt="" loading="lazy" />
        </span>
        <span className="sample__meta">
          <span className="sample__title">{sentenceCase(s.title)}</span>
          <span className="sample__kind">{KIND_LABEL[s.kind] ?? s.kind}</span>
          {s.objective && <span className="sample__obj">{s.objective}</span>}
          <span className="sample__prep">
            {s.featured === false
              ? "Earlier example · outlines not re-checked"
              : s.corrections
                ? "Prepared lesson · reviewed and corrected in Feelable"
                : "Prepared lesson · reviewed"}
          </span>
          {s.credit && <span className="sample__credit">{s.credit}</span>}
          <span className="sample__go" aria-hidden="true"><IconArrowRight size={18} /></span>
        </span>
        <span className="sr-only">Explore this example</span>
      </button>
    </li>
  );
}

export function SampleGallery({
  samples,
  failed,
  onPick,
}: {
  samples: Sample[] | null;
  failed: boolean;
  onPick: (s: Sample) => void;
}) {
  const [all, setAll] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  if (failed) {
    return <p className="empty-note">Examples are unavailable right now. Upload your own diagram above.</p>;
  }
  if (samples === null) {
    return (
      <ul className="samples" aria-busy="true" aria-label="Loading examples">
        {[0, 1, 2, 3].map((i) => (
          <li key={i} className="sample sample--skeleton" aria-hidden="true">
            <div className="sample__thumb" />
            <div className="sample__meta"><span className="skeleton-line" /></div>
          </li>
        ))}
      </ul>
    );
  }
  if (samples.length === 0) {
    return <p className="empty-note">No examples on this server yet. Upload your own diagram above.</p>;
  }
  const anyFeatured = samples.some((s) => s.featured);
  const featured = anyFeatured ? samples.filter((s) => s.featured) : samples;
  const extended = anyFeatured ? samples.filter((s) => !s.featured) : [];
  const shown = featured.filter((s) => filter === "all" || (filter === "complex" ? s.group === "complex" : s.group !== "complex"));
  const nBasic = featured.filter((s) => s.group !== "complex").length;
  const nComplex = featured.length - nBasic;
  return (
    <div className="gallery">
      <p className="gallery__note">
        Lessons open instantly because they were prepared and reviewed in advance. Your own photos take longer to
        read, and you check what Feelable found.
      </p>
      <div className="segmented gallery__filter" role="group" aria-label="Show lessons">
        {(
          [
            ["all", `All ${featured.length}`],
            ["simple", `Basic ${nBasic}`],
            ["complex", `Complex ${nComplex}`],
          ] as [Filter, string][]
        ).map(([k, label]) => (
          <button key={k} type="button" className={`segmented__opt${filter === k ? " is-on" : ""}`} aria-pressed={filter === k} onClick={() => setFilter(k)}>
            {label}
          </button>
        ))}
      </div>
      <ul className="samples" aria-label="Featured lessons">
        {shown.map((s) => (
          <Card key={s.id} s={s} onPick={onPick} />
        ))}
      </ul>
      {extended.length > 0 && (
        <>
          <button type="button" className="gallery__more" aria-expanded={all} onClick={() => setAll((v) => !v)}>
            {all ? "Hide the extended library" : `Extended library: ${extended.length} earlier examples`}
          </button>
          {all && (
            <ul className="samples" aria-label="Extended library">
              {extended.map((s) => (
                <Card key={s.id} s={s} onPick={onPick} />
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
