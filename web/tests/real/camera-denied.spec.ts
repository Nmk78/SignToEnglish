import { expect, test } from "@playwright/test";
import { ensureService } from "./service";

test.beforeEach(async () => ensureService());

// Chromium is launched so that it answers the camera prompt with "block": a real NotAllowedError.
test("real service: refused camera permission gives a clear message and opens no stream", async ({ page }) => {
  let sockets = 0;
  page.on("websocket", () => (sockets += 1));
  await page.goto("/");
  await expect(page.getByTestId("service-status")).toContainText("Service ready");
  await page.getByTestId("camera-start").click();
  const error = page.getByTestId("camera-error");
  await expect(error).toContainText("Camera permission was refused");
  await expect(error).toHaveAttribute("role", "alert");
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  await expect(page.getByTestId("camera-start")).toBeEnabled();
  expect(sockets).toBe(0);
});
