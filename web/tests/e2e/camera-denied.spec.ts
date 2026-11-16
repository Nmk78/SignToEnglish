import { expect, test } from "@playwright/test";
import { fakeSessions, resetFake } from "./helpers";

test.beforeEach(async ({ request }) => resetFake(request));

// This project launches Chromium so that it answers the camera prompt with "block": getUserMedia
// is really refused with NotAllowedError; nothing in the page is replaced or mocked.
test("refused camera permission gives a clear message and opens no stream", async ({ page, request }) => {
  await page.goto("/");
  await page.getByTestId("camera-start").click();

  const error = page.getByTestId("camera-error");
  await expect(error).toBeVisible();
  await expect(error).toContainText("Camera permission was refused");
  await expect(error).toContainText("site settings");
  await expect(error).toHaveAttribute("role", "alert");

  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  await expect(page.getByTestId("camera-start")).toBeEnabled();
  expect(await fakeSessions(request)).toHaveLength(0);
});
