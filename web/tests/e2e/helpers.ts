import { resolve } from "node:path";
import { type APIRequestContext, type Page, expect } from "@playwright/test";

export const FAKE = "http://127.0.0.1:8787";
export const CLIP_FILE = resolve(__dirname, "..", "media", "clip.webm");
export const BAD_FILE = resolve(__dirname, "..", "media", "not-a-video.txt");
export const CLIP_MS = 6000;
/** Must match the default in src/lib/config.ts; the test build does not override it. */
export const SEND_FPS = 15;

export interface FakeSession {
  id: number;
  start: { type: string; width: number; height: number; client_fps: number } | null;
  frames: number;
  bad_frames: number;
  non_increasing_timestamps: number;
  first_ts_ms: number | null;
  last_ts_ms: number | null;
  first_frame_wall_ms: number | null;
  last_frame_wall_ms: number | null;
  min_jpeg_bytes: number | null;
  max_jpeg_bytes: number | null;
  stop_received: boolean;
  finals_sent: number;
  partials_sent: number;
  closed: boolean;
}

/**
 * Switches read-aloud off through the page's own remembered setting, so tests that are not about
 * speech do not make this computer talk. (Read-aloud is on by default.)
 */
export async function readAloudOff(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.sessionStorage.setItem("s2e.speech", JSON.stringify({ enabled: false, voiceUri: null, rate: 1 }));
  });
}

export async function resetFake(request: APIRequestContext): Promise<void> {
  expect((await request.post(`${FAKE}/__dev/reset`)).ok()).toBeTruthy();
}

export async function fakeSessions(request: APIRequestContext): Promise<FakeSession[]> {
  const response = await request.get(`${FAKE}/__dev/sessions`);
  expect(response.ok()).toBeTruthy();
  return (await response.json()) as FakeSession[];
}

/** Frames per second according to the timestamps the page put on the frames. */
export function stampedFps(session: FakeSession): number {
  if (session.first_ts_ms === null || session.last_ts_ms === null || session.frames < 2) return 0;
  return ((session.frames - 1) * 1000) / (session.last_ts_ms - session.first_ts_ms);
}
