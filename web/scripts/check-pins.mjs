// Fails when package.json has a dependency that is not an exact version (CLAUDE.md, pinned-version policy).
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(readFileSync(resolve(webRoot, "package.json"), "utf8"));
const exact = /^\d+\.\d+\.\d+$/;
const bad = [];
for (const group of ["dependencies", "devDependencies", "optionalDependencies", "peerDependencies"]) {
  for (const [name, version] of Object.entries(pkg[group] ?? {})) {
    if (!exact.test(version)) bad.push(`${group}.${name}: ${version}`);
  }
}
if (bad.length > 0) {
  console.error(`Not exact pins in web/package.json:\n  ${bad.join("\n  ")}`);
  process.exit(1);
}
console.log("check-pins: every dependency in web/package.json is an exact version");
