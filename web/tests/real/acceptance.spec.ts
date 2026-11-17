// PLAN.md Step 4.1 acceptance against the real service. The model is the unadapted base
// checkpoint and sentence ends are fixed 8-second windows, so the TEXT is not checked for
// correctness: these tests check that text from the real model arrives and is shown, not that it
// is right. Nothing here asserts a speed; times are measured and printed.
import { expect, test } from "@playwright/test";
import { instrumentSpeech, speechLog } from "../e2e/speech-instrument";
import { NOT_A_VIDEO, PATTERN_CLIP, SIGNING_CLIP, SIGNING_CLIP_MS, SIGNING_SHORT, ofType, recordServerMessages } from "./helpers";
import { ensureService, health, stopService } from "./service";

test.beforeEach(async () => ensureService());

test("real service: the webcam path (signing video as the camera) shows partial and committed text, and Stop delivers the last sentence", async ({ page }) => {
  await instrumentSpeech(page);
  const messages = recordServerMessages(page);
  const info = await health();
  await page.goto("/");
  await expect(page.getByTestId("service-status")).toContainText("Service ready");
  await expect(page.getByTestId("checkpoint-id")).toContainText(info!.checkpoint_id!);

  const t0 = Date.now();
  await page.getByTestId("camera-start").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");
  await expect(page.getByTestId("send-settings")).toHaveText("640×360 · JPEG quality 0.80 · 10 fps");

  const partial = page.getByTestId("caption-partial");
  await expect(partial).toBeVisible();
  const firstPartialS = (Date.now() - t0) / 1000;
  await expect(partial).toHaveCSS("font-style", "italic");

  const finals = page.getByTestId("caption-final");
  await expect(finals.first()).toBeVisible();
  const firstFinalS = (Date.now() - t0) / 1000;
  await expect(finals.first()).toHaveCSS("font-style", "normal");
  const firstText = (await finals.first().textContent()) ?? "";
  expect(firstText.trim().length).toBeGreaterThan(0);
  await expect(finals.nth(1)).toBeVisible();
  // Committed text does not change.
  await expect(finals.first()).toHaveText(firstText);

  // Stop in the middle of a sentence: the page waits for the service to finish it.
  await expect(partial).toBeVisible();
  const before = await finals.count();
  const stopAt = Date.now();
  await page.getByTestId("camera-stop").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Finishing the last sentence…");
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  const stopS = (Date.now() - stopAt) / 1000;
  await expect(page.getByTestId("stream-error")).toHaveCount(0);
  await expect(page.getByTestId("last-session")).toContainText("frames received");
  expect(await page.getByTestId("camera-video").evaluate((v: HTMLVideoElement) => v.srcObject === null)).toBe(true);

  const summary = ofType(messages, "summary")[0]!;
  expect(summary).toBeTruthy();
  expect(summary.complete).toBe(true);
  const frames = summary.frames as Record<string, number>;
  expect(frames.rejected_bad).toBe(0); // every frame was a JPEG of the announced size
  expect(frames.rejected_non_increasing).toBe(0);
  const serverFinals = ofType(messages, "final");
  const flushed = serverFinals.filter((m) => m.closed_by === "flush");
  // The sentence that was open at Stop came back as a final and is on the page.
  expect(flushed.length).toBe(1);
  expect(await finals.count()).toBe(before + 1);
  expect(await finals.count()).toBe(serverFinals.filter((m) => String(m.text).trim() !== "").length);
  expect(ofType(messages, "error")).toHaveLength(0);

  const statuses = ofType(messages, "status");
  const lastStatus = statuses[statuses.length - 1]!;
  const speech = await speechLog(page);
  console.log(
    `[measured] real camera path: first partial ${firstPartialS.toFixed(1)} s and first final ${firstFinalS.toFixed(1)} s after Start; ` +
      `Stop to closed ${stopS.toFixed(2)} s; frames received ${frames.received}, dropped ${frames.dropped_behind}, processed ${frames.processed}; ` +
      `stream ${(Number(summary.stream_ms) / 1000).toFixed(1)} s => ${((frames.received! - 1) * 1000 / Number(summary.stream_ms)).toFixed(2)} fps received; ` +
      `last status rtf ${String(lastStatus.rtf)}; final latency_ms ${JSON.stringify(serverFinals.map((m) => m.latency_ms))}, decode_ms ${JSON.stringify(serverFinals.map((m) => m.decode_ms))}; ` +
      `partials ${ofType(messages, "partial").length}, dropped stretches ${ofType(messages, "dropped").length}`,
  );
  console.log(`[measured] real camera path, model output (unadapted model, not expected to be right): ${JSON.stringify(serverFinals.map((m) => m.text))}`);
  console.log(`[measured] read-aloud with real pacing: ${speech.spoken.length} sentences handed to speech for ${serverFinals.length} finals; skip notice shown: ${await page.getByTestId("speech-skipped").count()}`);
  // Every committed sentence was handed to speech once, none skipped.
  expect(speech.spoken.map((s) => s.text)).toEqual(serverFinals.map((m) => String(m.text).trim()).filter((t) => t !== ""));
  await expect(page.getByTestId("speech-skipped")).toHaveCount(0);
});

test("real service: play as live streams the file's frames with media timestamps and ends by itself", async ({ page }) => {
  await instrumentSpeech(page);
  const messages = recordServerMessages(page);
  await page.goto("/");
  await page.getByTestId("mode-file").click();
  await page.getByTestId("file-input").setInputFiles(SIGNING_CLIP);
  await page.getByTestId("play-as-live").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");

  const finals = page.getByTestId("caption-final");
  await expect(finals.first()).toBeVisible();
  // The first sentence is committed while the file is still playing.
  expect(await page.getByTestId("file-video").evaluate((v: HTMLVideoElement) => !v.ended)).toBe(true);

  await expect(page.getByTestId("stream-phase")).toHaveText("Idle", { timeout: 120_000 });
  await expect(page.getByTestId("stream-error")).toHaveCount(0);
  await expect(page.getByTestId("play-as-live")).toBeVisible();

  const summary = ofType(messages, "summary")[0]!;
  expect(summary.complete).toBe(true);
  const frames = summary.frames as Record<string, number>;
  expect(frames.rejected_bad).toBe(0);
  expect(frames.rejected_non_increasing).toBe(0);
  // Media timestamps: the stream is as long as the file, not as long as the wall clock since 1970.
  expect(Number(summary.stream_ms)).toBeGreaterThan(SIGNING_CLIP_MS - 1500);
  expect(Number(summary.stream_ms)).toBeLessThanOrEqual(SIGNING_CLIP_MS);
  const serverFinals = ofType(messages, "final");
  for (const final of serverFinals) {
    expect(Number(final.start_ms)).toBeGreaterThanOrEqual(0);
    expect(Number(final.end_ms)).toBeLessThanOrEqual(SIGNING_CLIP_MS);
  }
  expect(serverFinals.length).toBeGreaterThanOrEqual(2); // two 8-second windows
  await expect(finals).toHaveCount(serverFinals.filter((m) => String(m.text).trim() !== "").length);
  console.log(
    `[measured] real play as live (16 s file): frames received ${frames.received} (160 expected at 10 fps), dropped ${frames.dropped_behind}, ` +
      `stream ${summary.stream_ms} ms; finals ${JSON.stringify(serverFinals.map((m) => ({ start: m.start_ms, end: m.end_ms, by: m.closed_by, text: m.text })))}`,
  );
  // Read aloud, including the last sentence that arrives after the file has ended.
  await expect.poll(async () => (await speechLog(page)).spoken.length, { timeout: 30_000 }).toBe(serverFinals.filter((m) => String(m.text).trim() !== "").length);
  await expect(page.getByTestId("speech-skipped")).toHaveCount(0);
});

test("real service: translate file lists each sentence with its time range", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("mode-file").click();
  await page.getByTestId("file-input").setInputFiles(SIGNING_SHORT);
  const t0 = Date.now();
  await page.getByTestId("translate-file").click();
  await expect(page.getByTestId("translate-file")).toHaveText("Translating…");
  const segments = page.getByTestId("translate-segment");
  await expect(segments.first()).toBeVisible({ timeout: 180_000 });
  const seconds = (Date.now() - t0) / 1000;
  const rows = await segments.allTextContents();
  for (const row of rows) expect(row).toMatch(/^\d+:\d\d\.\d – \d+:\d\d\.\d/);
  await expect(page.getByTestId("translate-model")).toContainText("not adapted");
  await expect(page.getByTestId("translate-model")).toContainText("sentence ends: fixed length");
  await expect(page.getByTestId("file-video")).toBeVisible();
  await expect(page.getByTestId("segment-speak").first()).toBeVisible();
  console.log(`[measured] real translate file (9 s clip): answered in ${seconds.toFixed(1)} s; rows ${JSON.stringify(rows)}; ${await page.getByTestId("translate-model").textContent()}`);
});

test("real service: a video with nobody in it gives an empty result that says why", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("mode-file").click();
  await page.getByTestId("file-input").setInputFiles(PATTERN_CLIP);
  await page.getByTestId("translate-file").click();
  await expect(page.getByTestId("translate-empty")).toBeVisible({ timeout: 180_000 });
  await expect(page.getByTestId("translate-skipped")).toContainText("was not translated because");
  console.log(`[measured] real translate file, test pattern: ${JSON.stringify(await page.getByTestId("translate-skipped").locator("li").allTextContents())}`);
});

test("real service: a file that is not a video is refused with the service's reason", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("mode-file").click();
  await page.getByTestId("file-input").setInputFiles(NOT_A_VIDEO);
  await page.getByTestId("translate-file").click();
  const error = page.getByTestId("translate-error");
  await expect(error).toContainText("Unsupported file");
  await expect(error).toContainText("MP4, WebM, or MOV");
});

// Last in this file: it kills the service. beforeEach starts it again for whatever runs next.
test("real service: when the service dies mid-stream the page says so, and says so again when it cannot reconnect", async ({ page }) => {
  await page.addInitScript(() => {
    window.sessionStorage.setItem("s2e.speech", JSON.stringify({ enabled: false, voiceUri: null, rate: 1 }));
  });
  await page.goto("/");
  await page.getByTestId("camera-start").click();
  await expect(page.getByTestId("caption-partial")).toBeVisible();

  await stopService(true); // SIGKILL, as a crash would be

  const error = page.getByTestId("stream-error");
  await expect(error).toContainText("connection to the translation service was lost");
  await expect(error).toHaveAttribute("role", "alert");
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  await expect(page.getByTestId("caption-partial")).toHaveCount(0);
  expect(await page.getByTestId("camera-video").evaluate((v: HTMLVideoElement) => v.srcObject === null)).toBe(true);
  await expect(page.getByTestId("service-status")).toContainText("Service unreachable");

  await page.getByTestId("camera-start").click();
  await expect(error).toContainText("Could not connect to the translation service");
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");

  // The service comes back; the page notices and streaming works again.
  await ensureService();
  await expect(page.getByTestId("service-status")).toContainText("Service ready");
  await page.getByTestId("camera-start").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");
  await expect(error).toHaveCount(0);
  await expect(page.getByTestId("caption-partial")).toBeVisible();
  await page.getByTestId("camera-stop").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
});

