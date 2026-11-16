import { domainInfo } from "@/lib/domain";

const EXAMPLE_COUNT = 12;

/** What the demo covers: the one domain, example sentences, and its limits. */
export function DomainPanel() {
  const { sentences, counts } = domainInfo;
  const domainName = domainInfo.domain.replace(/_/g, " ");
  const examples = sentences.slice(0, EXAMPLE_COUNT);
  const rest = sentences.slice(EXAMPLE_COUNT);

  return (
    <section className="panel domain-panel" aria-labelledby="domain-heading" data-testid="domain-panel">
      <p className="step">WHAT THIS DEMO COVERS</p>
      <h2 id="domain-heading">One domain: {domainName}</h2>
      <p className="domain-intro">
        This demo is limited to one conversation domain: {counts.sentences} short sentences about introductions and
        everyday small talk, using about {counts.vocabulary} signs, signed by the one signer it is tuned to. Outside
        this domain, or with another signer, expect poor output.
      </p>
      <p className="result-label">EXAMPLE SENTENCES</p>
      <ul className="domain-list" data-testid="domain-examples">
        {examples.map((s) => (
          <li key={s.id}>{s.text}</li>
        ))}
      </ul>
      {rest.length > 0 && (
        <details className="domain-more">
          <summary>Show all {counts.sentences} sentences</summary>
          <ul className="domain-list">
            {rest.map((s) => (
              <li key={s.id}>{s.text}</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
