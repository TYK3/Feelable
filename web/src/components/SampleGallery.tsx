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

export function SampleGallery({
  samples,
  failed,
  onPick,
}: {
  samples: Sample[] | null;
  failed: boolean;
  onPick: (s: Sample) => void;
}) {
  if (failed) {
    return <p className="empty-note">Samples are unavailable right now. Upload your own diagram above.</p>;
  }
  if (samples === null) {
    return (
      <ul className="samples" aria-busy="true" aria-label="Loading samples">
        {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((i) => (
          <li key={i} className="sample sample--skeleton" aria-hidden="true">
            <div className="sample__thumb" />
            <div className="sample__meta"><span className="skeleton-line" /></div>
          </li>
        ))}
      </ul>
    );
  }
  if (samples.length === 0) {
    return <p className="empty-note">No samples on this server yet. Upload your own diagram above.</p>;
  }
  return (
    <ul className="samples">
      {samples.map((s) => (
        <li key={s.id}>
          <button type="button" className="sample" onClick={() => onPick(s)}>
            <span className="sample__thumb">
              <img src={s.thumb_url} alt="" loading="lazy" />
            </span>
            <span className="sample__meta">
              <span className="sample__title">{sentenceCase(s.title)}</span>
              <span className="sample__kind">{KIND_LABEL[s.kind] ?? s.kind}</span>
              {s.credit && <span className="sample__credit">{s.credit}</span>}
              <span className="sample__go" aria-hidden="true"><IconArrowRight size={18} /></span>
            </span>
            <span className="sr-only">Make a tactile graphic from this sample</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
