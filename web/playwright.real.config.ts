// Browser tests against the REAL translation service (PLAN.md Step 4.1 acceptance). They start
// `uv run --frozen s2e serve` from the repository root, wait until the model is loaded, and stop
// it when they finish. The camera is a local copy of public-domain ASL video, so real signing
// reaches the real model. These tests are slow and load the model; the fast, model-free set
// against the development fake server is `npm run test:e2e` (playwright.config.ts).
//
//   npm run test:real        functional acceptance (about 4 minutes)
//   npm run test:real:soak   the 5-minute send-rate and memory measurement
//
// The page is served on 127.0.0.1:3000 because that is the origin the service allows.
import { resolve } from "node:path";
import { defineConfig } from "@playwright/test";

const soak = process.env.S2E_SOAK === "1";
const cameraFile = resolve(__dirname, "tests", "media", "asl_camera.mjpeg");

export default defineConfig({
  testDir: "./tests/real",
  outputDir: "./test-results-real/artifacts",
  globalSetup: "./tests/real/global-setup.ts",
  globalTeardown: "./tests/real/global-teardown.ts",
  fullyParallel: false,
  workers: 1, // one model, one service
  retries: 0,
  timeout: soak ? 480_000 : 240_000,
  expect: { timeout: 60_000 },
  reporter: [["list"]],
  use: { baseURL: "http://127.0.0.1:3000", trace: "retain-on-failure" },
  projects: soak
    ? [
        {
          name: "soak",
          testMatch: /soak\.spec\.ts/,
          use: {
            browserName: "chromium",
            launchOptions: {
              args: [
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                `--use-file-for-fake-video-capture=${cameraFile}`,
                "--js-flags=--expose-gc",
              ],
            },
          },
        },
      ]
    : [
        {
          name: "real-camera-allowed",
          testMatch: /acceptance\.spec\.ts/,
          use: {
            browserName: "chromium",
            launchOptions: {
              args: [
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                `--use-file-for-fake-video-capture=${cameraFile}`,
              ],
            },
          },
        },
        {
          name: "real-camera-denied",
          testMatch: /camera-denied\.spec\.ts/,
          use: {
            browserName: "chromium",
            launchOptions: {
              args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream=deny"],
            },
          },
        },
      ],
  webServer: [
    {
      // A production build with the default settings: service at 127.0.0.1:8000, 10 fps.
      command: "npm run gen && npx next build && npx next start --hostname 127.0.0.1 --port 3000",
      url: "http://127.0.0.1:3000",
      env: { S2E_DIST_DIR: ".next-real", NEXT_TELEMETRY_DISABLED: "1" },
      reuseExistingServer: false,
      timeout: 300_000,
    },
  ],
});
