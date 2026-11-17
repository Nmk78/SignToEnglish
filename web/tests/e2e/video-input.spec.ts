import { expect, test } from "@playwright/test";
import { BAD_FILE, CLIP_FILE, CLIP_MS, SEND_FPS, fakeSessions, readAloudOff, resetFake, stampedFps } from "./helpers";

test.beforeEach(async ({ page, request }) => {
  await resetFake(request);
  await readAloudOff(page);
});

test("translate file: the upload goes to POST /translate and each sentence is listed with its time range", async ({ page, request }) => {
  await page.goto("/");
  await page.getByTestId("mode-file").click();
  await expect(page.getByTestId("translate-file")).toBeDisabled();
  await page.getByTestId("file-input").setInputFiles(CLIP_FILE);

  const [upload] = await Promise.all([
    page.waitForRequest((r) => r.url().endsWith("/translate") && r.method() === "POST"),
    page.getByTestId("translate-file").click(),
  ]);
  expect(upload.headers()["content-type"]).toContain("multipart/form-data");

  const segments = page.getByTestId("translate-segment");
  await expect(segments).toHaveCount(2);
  await expect(segments.nth(0)).toContainText("0:00.4 – 0:02.6");
  await expect(segments.nth(0)).toContainText("[canned] Hello, how are you?");
  await expect(segments.nth(1)).toContainText("0:03.3 – 0:05.4");
  await expect(segments.nth(1)).toContainText("[canned] Nice to meet you.");
  // The result sits beside the video, and a time range jumps the video to that sentence.
  await expect(page.getByTestId("file-video")).toBeVisible();
  await segments.nth(1).locator(".segment-time").click();
  await expect.poll(() => page.getByTestId("file-video").evaluate((v: HTMLVideoElement) => v.currentTime)).toBeCloseTo(3.3, 1);
  // Translate file does not open a stream.
  expect(await fakeSessions(request)).toHaveLength(0);
});

test("translate file: a file the service rejects shows the service's reason", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("mode-file").click();
  await page.getByTestId("file-input").setInputFiles(BAD_FILE);
  await page.getByTestId("translate-file").click();
  const error = page.getByTestId("translate-error");
  await expect(error).toBeVisible();
  await expect(error).toContainText("Unsupported file");
  await expect(error).toContainText("MP4, WebM, or MOV");
  await expect(page.getByTestId("translate-result")).toHaveCount(0);
});

test("translate file: an empty result says why nothing was translated", async ({ page, request }) => {
  await request.post("http://127.0.0.1:8787/__dev/set", { data: { translate: "empty" } });
  await page.goto("/");
  await page.getByTestId("mode-file").click();
  await page.getByTestId("file-input").setInputFiles(CLIP_FILE);
  await page.getByTestId("translate-file").click();
  await expect(page.getByTestId("translate-empty")).toContainText("No sentence was found");
  await expect(page.getByTestId("translate-skipped")).toContainText("0:00.0 – 0:06.0 was not translated because no signing was seen");
  await expect(page.getByTestId("translate-model")).toContainText("FAKE-DEV-SERVER · not adapted · none · sentence ends: fixed length");
});

test("play as live: the file plays in the page and its frames are streamed with media timestamps", async ({ page, request }) => {
  await page.goto("/");
  await page.getByTestId("mode-file").click();
  await page.getByTestId("file-input").setInputFiles(CLIP_FILE);
  await page.getByTestId("play-as-live").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");

  // Captions arrive while the video is still playing.
  const finals = page.getByTestId("caption-final");
  await expect(finals.first()).toHaveText("[canned] Hello, how are you?");
  expect(await page.getByTestId("file-video").evaluate((v: HTMLVideoElement) => !v.ended)).toBe(true);

  // When the file ends the page sends "stop" and the session finishes by itself.
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle", { timeout: 20_000 });
  await expect(page.getByTestId("stream-error")).toHaveCount(0);
  await expect(page.getByTestId("play-as-live")).toBeVisible();

  const sessions = await fakeSessions(request);
  expect(sessions).toHaveLength(1);
  const s = sessions[0]!;
  expect(s.start).toEqual({ type: "start", width: 640, height: 480, client_fps: SEND_FPS });
  expect(s.bad_frames).toBe(0);
  expect(s.non_increasing_timestamps).toBe(0);
  expect(s.stop_received).toBe(true);
  // Media timestamps: they start at the beginning of the file and never pass its end, which
  // wall-clock capture times (about 1.8e12) could not do.
  expect(s.first_ts_ms).toBeGreaterThanOrEqual(0);
  expect(s.first_ts_ms).toBeLessThan(200);
  expect(s.last_ts_ms).toBeGreaterThan(CLIP_MS - 500);
  expect(s.last_ts_ms).toBeLessThanOrEqual(CLIP_MS);
  // The whole 6 s file was sent at the configured rate (60 frames at 10 fps).
  const expectedFrames = (CLIP_MS / 1000) * SEND_FPS;
  const fps = stampedFps(s);
  console.log(`[measured] play as live: ${s.frames} frames (expected ${expectedFrames}), ${fps.toFixed(2)} fps by media timestamps, first ${s.first_ts_ms} ms, last ${s.last_ts_ms} ms`);
  expect(s.frames).toBeGreaterThanOrEqual(expectedFrames * 0.9);
  expect(s.frames).toBeLessThanOrEqual(expectedFrames + 1);
  // Every final the server sent is on the page.
  await expect(finals).toHaveCount(s.finals_sent);
});
