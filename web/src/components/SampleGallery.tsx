// Prepared examples (handover: exactly 15 — ten to start with, five complex systems). Each card
// says what a student learns from it and, honestly, that it was prepared in advance: it opens
// instantly, unlike a fresh photo, and complex ones were checked and corrected in Feelable.
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
const START_WITH = 4;

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
            {s.corrections ? "Prepared example · checked and corrected in Feelable" : "Prepared example"}
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
  const simple = samples.filter((s) => s.group !== "complex");
  const complex = samples.filter((s) => s.group === "complex");
  const shownSimple = all ? simple : simple.slice(0, START_WITH);
  return (
    <div className="gallery">
      <p className="gallery__note">
        Examples open instantly because they were prepared in advance. Your own photos take longer to read.
      </p>
      <h3 className="gallery__h" id="g-simple">Start with a simple diagram</h3>
      <ul className="samples" aria-labelledby="g-simple">
        {shownSimple.map((s) => (
          <Card key={s.id} s={s} onPick={onPick} />
        ))}
      </ul>
      {complex.length > 0 && (
        <>
          <h3 className="gallery__h" id="g-complex">Explore {complex.length === 5 ? "five" : complex.length} complex systems</h3>
          <ul className="samples samples--complex" aria-labelledby="g-complex">
            {complex.map((s) => (
              <Card key={s.id} s={s} onPick={onPick} />
            ))}
          </ul>
        </>
      )}
      {simple.length > START_WITH && (
        <button type="button" className="gallery__more" aria-expanded={all} onClick={() => setAll((v) => !v)}>
          {all ? "Show fewer examples" : `Show all ${samples.length} examples`}
        </button>
      )}
    </div>
  );
}
