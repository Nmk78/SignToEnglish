import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

// Generates the synthetic camera and clip files (ffmpeg test patterns) if they are missing.
export default function globalSetup(): void {
  execFileSync("node", [resolve(__dirname, "..", "..", "scripts", "make-test-media.mjs")], { stdio: "inherit" });
}
