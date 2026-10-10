import { useEffect } from "react";
import { AppBar } from "../components/AppBar";
import { BrailleWord } from "../components/BrailleWord";
import { LinkButton } from "../components/Button";
import { IconArrowLeft } from "../components/Icons";

export default function NotFound({
  title = "We can't find that page.",
  message = "The link may be mistyped, or the page has moved.",
}: {
  title?: string;
  message?: string;
}) {
  useEffect(() => {
    document.title = "Not found · Feelable";
  }, []);
  return (
    <>
      <AppBar />
      <main className="page notfound" id="main" tabIndex={-1}>
        <div className="card notfound__card">
          <BrailleWord word="lost" className="notfound__braille" />
          <p className="eyebrow">404</p>
          <h1 className="notfound__title">{title}</h1>
          <p className="notfound__msg">{message}</p>
          <LinkButton href="#/" variant="primary" size="lg" icon={<IconArrowLeft size={20} />}>
            Make a new tactile graphic
          </LinkButton>
        </div>
      </main>
    </>
  );
}
