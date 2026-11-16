import { expect, test } from "@playwright/test";
import { SEND_FPS, fakeSessions, resetFake, stampedFps } from "./helpers";

test.beforeEach(async ({ request }) => resetFake(request));

test("webcam path: frames from the camera produce partial and committed captions", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.getByTestId("service-status")).toContainText("Service ready");
  await expect(page.getByTestId("framing-guide")).toBeVisible();
  await expect(page.getByTestId("experimental-notice")).toBeVisible();
  await expect(page.getByTestId("domain-examples").locator("li").first()).toBeVisible();

  const wallBefore = Date.now();
  await page.getByTestId("camera-start").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");

  // Partial text shows up first and is styled differently from committed text.
  const partial = page.getByTestId("caption-partial");
  await expect(partial).toBeVisible();
  await expect(partial).toContainText("[canned]");
  await expect(partial).toContainText("may change");
  await expect(partial).toHaveCSS("font-style", "italic");

  const finals = page.getByTestId("caption-final");
  await expect(finals.first()).toHaveText("[canned] Hello, how are you?");
  await expect(finals.first()).toHaveCSS("font-style", "normal");
  await expect(finals.nth(1)).toHaveText("[canned] Nice to meet you.");
  // Committed text does not change once shown.
  await expect(finals.first()).toHaveText("[canned] Hello, how are you?");

  // Let it run a little so the rate measurement has something to work with.
  await page.waitForTimeout(4000);
  await page.getByTestId("camera-stop").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  await expect(page.getByTestId("stream-error")).toHaveCount(0);
  // The camera is released on stop.
  expect(await page.getByTestId("camera-video").evaluate((v: HTMLVideoElement) => v.srcObject === null)).toBe(true);

  const sessions = await fakeSessions(request);
  expect(sessions).toHaveLength(1);
  const s = sessions[0]!;
  expect(s.start).toEqual({ type: "start", width: 640, height: 480, client_fps: SEND_FPS });
  expect(s.frames).toBeGreaterThan(40);
  expect(s.bad_frames).toBe(0); // every binary message was 8 timestamp bytes plus a whole JPEG
  expect(s.non_increasing_timestamps).toBe(0);
  expect(s.min_jpeg_bytes).toBeGreaterThan(1000);
  expect(s.stop_received).toBe(true);
  expect(s.closed).toBe(true);
  // Camera frames carry capture time on the wall clock (milliseconds since the Unix epoch).
  expect(s.first_ts_ms).toBeGreaterThan(wallBefore - 2000);
  expect(s.last_ts_ms).toBeLessThan(Date.now() + 2000);

  const fps = stampedFps(s);
  test.info().annotations.push({ type: "measured", description: `camera: ${s.frames} frames, ${fps.toFixed(2)} fps by frame timestamps (configured ${SEND_FPS})` });
  console.log(`[measured] camera path: ${s.frames} frames, ${fps.toFixed(2)} fps by frame timestamps (configured ${SEND_FPS})`);
  expect(fps).toBeGreaterThan(SEND_FPS * 0.8);
  expect(fps).toBeLessThan(SEND_FPS * 1.1);
});

test("status bar reports a service that is not ready, with the reason", async ({ page, request }) => {
  await request.post("http://127.0.0.1:8787/__dev/health", { data: { mode: "not_ready" } });
  await page.goto("/");
  await expect(page.getByTestId("service-status")).toContainText("Service not ready");
  await expect(page.getByTestId("service-status")).toContainText("not-ready mode");
});
