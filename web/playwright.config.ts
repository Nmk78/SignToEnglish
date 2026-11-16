// Browser tests for the web app. They run against the development fake server in dev/, with a
// synthetic test-pattern video as the webcam. They check the page's behavior against the
// documented protocol; they do not check translation, and they do not satisfy the Step 4.1
// acceptance gate, which needs the real service (Wave 3).
import { resolve } from "node:path";
import { defineConfig } from "@playwright/test";

export const FAKE_SERVER_URL = "http://127.0.0.1:8787";
export const APP_URL = "http://127.0.0.1:3100";

const cameraFile = resolve(__dirname, "tests", "media", "camera.mjpeg");

const appEnv = {
  S2E_DIST_DIR: ".next-e2e",
  NEXT_PUBLIC_S2E_SERVICE_URL: FAKE_SERVER_URL,
  NEXT_TELEMETRY_DISABLED: "1",
};

export default defineConfig({
  testDir: "./tests/e2e",
  globalSetup: "./tests/e2e/global-setup.ts",
  fullyParallel: false,
  workers: 1, // the fake server's session log is shared state
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: [["list"]],
  use: { baseURL: APP_URL, trace: "retain-on-failure" },
  projects: [
    {
      name: "camera-allowed",
      testIgnore: /camera-denied\.spec\.ts/,
      use: {
        browserName: "chromium",
        launchOptions: {
          args: [
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream", // answers the permission prompt with "allow"
            `--use-file-for-fake-video-capture=${cameraFile}`,
          ],
        },
      },
    },
    {
      name: "camera-denied",
      testMatch: /camera-denied\.spec\.ts/,
      use: {
        browserName: "chromium",
        launchOptions: {
          args: ["--use-fake-device-for-media-stream", "--deny-permission-prompts"],
        },
      },
    },
  ],
  webServer: [
    {
      command: "node dev/fake-server.mjs",
      url: `${FAKE_SERVER_URL}/health`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command: "npm run gen && npx next build && npx next start --hostname 127.0.0.1 --port 3100",
      url: APP_URL,
      env: appEnv,
      reuseExistingServer: false,
      timeout: 300_000,
    },
  ],
});
