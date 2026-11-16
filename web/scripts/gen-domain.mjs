// Build-time step: turns ../domain/sentences.yaml into src/generated/domain.json for DomainPanel.
// Only the English text, id, and role are copied. The `signs`, `asl`, and `source` fields stay out
// of the page: they are planning aids, not something the demo should present as ASL.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourcePath = resolve(webRoot, "..", "domain", "sentences.yaml");
const outPath = resolve(webRoot, "src", "generated", "domain.json");

function fail(message) {
  console.error(`gen-domain: ${message}`);
  process.exit(1);
}

const doc = parse(readFileSync(sourcePath, "utf8"));
if (!doc || typeof doc !== "object") fail(`${sourcePath} is not a YAML mapping`);
if (typeof doc.domain !== "string" || doc.domain === "") fail("missing `domain`");
if (!Array.isArray(doc.sentences) || doc.sentences.length === 0) fail("no sentences");

const seen = new Set();
const sentences = doc.sentences.map((s, i) => {
  if (!s || typeof s.id !== "string" || typeof s.text !== "string" || s.text.trim() === "") {
    fail(`sentence ${i} needs a string id and a non-empty text`);
  }
  if (s.role !== "train" && s.role !== "unseen_test") fail(`sentence ${s.id}: unknown role ${s.role}`);
  if (seen.has(s.id)) fail(`duplicate sentence id ${s.id}`);
  seen.add(s.id);
  return { id: s.id, text: s.text, role: s.role };
});

const out = {
  generated_from: "domain/sentences.yaml",
  domain: doc.domain,
  sign_language: doc.sign_language ?? null,
  text_language: doc.text_language ?? null,
  counts: {
    sentences: sentences.length,
    train: sentences.filter((s) => s.role === "train").length,
    unseen_test: sentences.filter((s) => s.role === "unseen_test").length,
    vocabulary: Array.isArray(doc.vocabulary) ? doc.vocabulary.length : 0,
  },
  sentences,
};

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(out, null, 2)}\n`);
console.log(`gen-domain: wrote ${sentences.length} sentences to src/generated/domain.json`);
