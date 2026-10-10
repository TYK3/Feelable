// Decorative braille rendered as real dot cells (CSS circles). Always aria-hidden.
import { dotsOf } from "./braille";

export function BrailleWord({ word, className }: { word: string; className?: string }) {
  return (
    <span className={["braille", className].filter(Boolean).join(" ")} aria-hidden="true">
      {[...word].map((ch, i) => {
        const dots = dotsOf(ch);
        // grid order: row-major 1 4 / 2 5 / 3 6
        const order = [1, 4, 2, 5, 3, 6];
        return (
          <span className="braille__cell" key={i}>
            {order.map((d) => (
              <span key={d} className={dots.includes(d) ? "braille__dot is-on" : "braille__dot"} />
            ))}
          </span>
        );
      })}
    </span>
  );
}
