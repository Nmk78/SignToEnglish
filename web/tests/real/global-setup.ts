import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { health, startService, startedByTests, stopService } from "./service";

export default async function globalSetup(): Promise<void> {
  const webRoot = resolve(__dirname, "..", "..");
  execFileSync("node", [resolve(webRoot, "scripts", "make-test-media.mjs")], { stdio: "inherit" });
  if (!existsSync(resolve(webRoot, "tests", "media", "asl_camera.mjpeg"))) {
    throw new Error(
      "tests/media/asl_camera.mjpeg is missing: the public-domain source video " +
        "tests/fixtures/clips/commons_source/COVID-19_Prevention_Tips_in_ASL.webm was not found.",
    );
  }
  if (startedByTests()) await stopService(); // left over from an interrupted run
  if ((await health()) !== null) {
    throw new Error("Something already answers on 127.0.0.1:8000. Stop that service first; these tests start their own.");
  }
  const seconds = await startService();
  console.log(`[measured] real service ready ${seconds.toFixed(1)} s after it was started`);
}
