import { expect, test } from "@playwright/test";
import { FAKE, fakeSessions, resetFake } from "./helpers";

test.beforeEach(async ({ request }) => resetFake(request));

test("a dropped connection gives a clear message, keeps committed text, and can be restarted", async ({ page, request }) => {
  await page.goto("/");
  await page.getByTestId("camera-start").click();
  const finals = page.getByTestId("caption-final");
  await expect(finals.first()).toHaveText("[canned] Hello, how are you?");

  // The fake server cuts the socket without a close handshake.
  const dropped = (await (await request.post(`${FAKE}/__dev/drop`)).json()) as { dropped: number };
  expect(dropped.dropped).toBe(1);

  const error = page.getByTestId("stream-error");
  await expect(error).toBeVisible();
  await expect(error).toContainText("connection to the translation service was lost");
  await expect(error).toHaveAttribute("role", "alert");
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  // Unfinished partial text is removed; committed text stays.
  await expect(page.getByTestId("caption-partial")).toHaveCount(0);
  await expect(finals.first()).toHaveText("[canned] Hello, how are you?");
  // The camera is released and the page stops sending.
  expect(await page.getByTestId("camera-video").evaluate((v: HTMLVideoElement) => v.srcObject === null)).toBe(true);
  const committedBefore = await finals.count();
  const framesAfterDrop = (await fakeSessions(request))[0]!.frames;
  await page.waitForTimeout(1000);
  expect((await fakeSessions(request))[0]!.frames).toBe(framesAfterDrop);

  // Starting again reconnects, clears the message, and appends to the same caption list.
  await page.getByTestId("camera-start").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");
  await expect(error).toHaveCount(0);
  await expect(finals.nth(committedBefore)).toBeVisible();
  await page.getByTestId("camera-stop").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  expect(await fakeSessions(request)).toHaveLength(2);
});
