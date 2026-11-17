"use client";

import { type ReactNode, useEffect, useRef } from "react";
import type { Caption, NotTranslated } from "@/hooks/useStream";
import { dropReasonText } from "@/lib/protocol";

interface CaptionsProps {
  committed: Caption[];
  partial: Caption | null;
  active: boolean;
  onClear: () => void;
  /** True while committed sentences are being read aloud by the page itself. */
  spokenAloud: boolean;
  /** Read-aloud controls, shown above the captions. */
  controls?: ReactNode;
  /** Stretches the service did not translate in this session. */
  notTranslated: NotTranslated | null;
  /** True after Stop while the service is still finishing the last sentence. */
  finishing: boolean;
}

/** Committed sentences (final, never change) and, below them, the unstable partial line. */
export function Captions({ committed, partial, active, onClear, spokenAloud, controls, notTranslated, finishing }: CaptionsProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [committed.length, partial?.text, finishing]);

  const empty = committed.length === 0 && partial === null && !finishing;

  return (
    <section className="panel captions-panel" aria-labelledby="captions-heading">
      <div className="section-heading">
        <div>
          <p className="step">02 · CAPTIONS</p>
          <h2 id="captions-heading">English</h2>
        </div>
        <button className="link-button" onClick={onClear} disabled={empty}>
          Clear
        </button>
      </div>

      {controls}

      <div className="captions-scroll" ref={scrollRef}>
        {empty ? (
          <div className="captions-empty">
            <p>{active ? "Waiting for the first sentence…" : "Captions appear here while you sign."}</p>
            <small>Lower your hands to rest between sentences so each one is committed.</small>
          </div>
        ) : (
          <>
            {/* When the page reads sentences aloud itself, the list is not a live region, so a
                screen reader does not say each sentence a second time. It stays readable. */}
            <ol
              className="captions-final"
              aria-live={spokenAloud ? "off" : "polite"}
              aria-label="Committed sentences"
              data-testid="captions-final"
            >
              {committed.map((caption) => (
                <li key={caption.key} className="caption-final" data-testid="caption-final">
                  {caption.text}
                </li>
              ))}
            </ol>
            {partial && partial.text !== "" && (
              <p className="caption-partial" data-testid="caption-partial" aria-label="Sentence in progress, may change">
                {partial.text}
                <span className="caption-partial-tag">in progress · may change</span>
              </p>
            )}
            {finishing && (
              <p className="caption-finishing" role="status" data-testid="caption-finishing">
                Finishing the last sentence…
              </p>
            )}
          </>
        )}
      </div>

      {notTranslated && (
        <p className="notice" data-testid="not-translated">
          {notTranslated.count === 1 ? "1 stretch was" : `${notTranslated.count} stretches were`} not translated; the
          last one because {dropReasonText(notTranslated.lastReason)}.
        </p>
      )}

      <p className="notice" data-testid="experimental-notice">
        Experimental output. Captions can be wrong, including when they read fluently.
      </p>
    </section>
  );
}
