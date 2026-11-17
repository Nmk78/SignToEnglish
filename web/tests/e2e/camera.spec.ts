import { expect, test } from "@playwright/test";
import { FAKE, SEND_FPS, fakeSessions, readAloudOff, resetFake, stampedFps } from "./helpers";

test.beforeEach(async ({ page, request }) => {
  await resetFake(request);
  await readAloudOff(page);
});

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
  // The send settings the model depends on are readable by a presenter.
  await expect(page.getByTestId("send-settings")).toHaveText(`640×480 · JPEG quality 0.80 · ${SEND_FPS} fps`);
  await expect(page.getByTestId("frames-dropped")).toHaveText("0 (0.0%)");
  await expect(page.getByTestId("dropping-hint")).toHaveCount(0);
  // Committed text does not change once shown.
  await expect(finals.first()).toHaveText("[canned] Hello, how are you?");

  // Let it run a little so the rate measurement has something to work with.
  await page.waitForTimeout(4000);
  const committedBeforeStop = await finals.count();
  await page.getByTestId("camera-stop").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  await expect(page.getByTestId("stream-error")).toHaveCount(0);
  // The service's summary, sent just before it closed the socket, is shown.
  await expect(page.getByTestId("last-session")).toContainText("frames received, 0 dropped");
  expect(await finals.count()).toBeGreaterThanOrEqual(committedBeforeStop);
  // The camera is released on stop.
  expect(await page.getByTestId("camera-video").evaluate((v: HTMLVideoElement) => v.srcObject === null)).toBe(true);

  const sessions = await fakeSessions(request);
  expect(sessions).toHaveLength(1);
  const s = sessions[0]!;
  expect(s.start).toEqual({ type: "start", width: 640, height: 480, client_fps: SEND_FPS });
  expect(s.frames).toBeGreaterThan(40);
  expect(s.bad_frames).toBe(0); // every binary message was 8 timestamp bytes plus a whole JPEG
  expect(s.wrong_size_frames).toBe(0); // and of the size the start message announced
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
  await request.post(`${FAKE}/__dev/health`, { data: { mode: "not_ready" } });
  await page.goto("/");
  await expect(page.getByTestId("service-status")).toContainText("Service not ready");
  await expect(page.getByTestId("service-status")).toContainText("not-ready mode");
});

// The base model has no left-right flip handling, so the service must get the picture as the
// camera delivers it. The synthetic camera has a red band on its left edge and a blue band on
// its right edge (scripts/make-test-media.mjs).
test("the preview is mirrored for the signer, but the frames sent to the service are not", async ({ page, request }) => {
  await page.goto("/");
  await page.getByTestId("camera-start").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");
  await expect.poll(async () => (await fakeSessions(request))[0]?.frames ?? 0).toBeGreaterThan(5);

  // What the signer sees is flipped horizontally.
  const transform = await page.getByTestId("camera-video").evaluate((v) => getComputedStyle(v).transform);
  expect(transform).toBe("matrix(-1, 0, 0, 1, 0, 0)");

  // What the service received is not: decode the last JPEG it got and look at both edges.
  const pixels = await page.evaluate(async (url) => {
    const blob = await (await fetch(url)).blob();
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(bitmap, 0, 0);
    const at = (x: number) => Array.from(ctx.getImageData(x, Math.floor(bitmap.height / 2), 1, 1).data.slice(0, 3));
    return { width: bitmap.width, height: bitmap.height, left: at(80), right: at(bitmap.width - 80) };
  }, `${FAKE}/__dev/last-frame`);
  await page.getByTestId("camera-stop").click();

  expect(pixels.width).toBe(640);
  expect(pixels.height).toBe(480);
  const [lr, lg, lb] = pixels.left as [number, number, number];
  const [rr, rg, rb] = pixels.right as [number, number, number];
  console.log(`[measured] sent frame: left edge rgb(${lr},${lg},${lb}), right edge rgb(${rr},${rg},${rb})`);
  // Left edge red, right edge blue, as the camera delivers it.
  expect(lr).toBeGreaterThan(180);
  expect(lb).toBeLessThan(80);
  expect(rb).toBeGreaterThan(180);
  expect(rr).toBeLessThan(80);
});

test("frames dropped by the service are counted and explained in plain words", async ({ page, request }) => {
  await request.post(`${FAKE}/__dev/set`, { data: { drop_frames: true } });
  await page.goto("/");
  await page.getByTestId("camera-start").click();
  await expect(page.getByTestId("dropping-hint")).toContainText("cannot keep up and is dropping frames");
  await expect(page.getByTestId("frames-dropped")).toContainText("%");
  await expect(page.getByTestId("frames-dropped")).not.toHaveText("0 (0.0%)");
  await page.getByTestId("camera-stop").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  await expect(page.getByTestId("last-session")).not.toContainText(", 0 dropped");
});

test("a stream the service refuses shows the service's own reason", async ({ page, request }) => {
  await request.post(`${FAKE}/__dev/set`, { data: { refuse: true } });
  await page.goto("/");
  await page.getByTestId("camera-start").click();
  const error = page.getByTestId("stream-error");
  await expect(error).toContainText("already has 2 open stream(s); close one and try again");
  await expect(error).not.toContainText("connection to the translation service was lost");
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  expect(await page.getByTestId("camera-video").evaluate((v: HTMLVideoElement) => v.srcObject === null)).toBe(true);
});

test("a stretch the service did not translate is reported, and the stream goes on", async ({ page, request }) => {
  await request.post(`${FAKE}/__dev/captions`, { data: { mode: "manual" } });
  await page.goto("/");
  await page.getByTestId("camera-start").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");
  await request.post(`${FAKE}/__dev/send`, {
    data: { messages: [{ type: "dropped", segment_id: 0, start_ms: 0, end_ms: 8000, reason: "no_signing" }] },
  });
  await expect(page.getByTestId("not-translated")).toContainText("1 stretch was not translated");
  await expect(page.getByTestId("not-translated")).toContainText("no signing was seen");
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");
  await expect(page.getByTestId("stream-error")).toHaveCount(0);
  await page.getByTestId("camera-stop").click();
});
