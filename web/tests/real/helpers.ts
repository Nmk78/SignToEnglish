import { resolve } from "node:path";
import type { Page } from "@playwright/test";

const MEDIA = resolve(__dirname, "..", "media");
export const SIGNING_CLIP = resolve(MEDIA, "asl_clip.webm"); // 16 s of real signing
export const SIGNING_SHORT = resolve(MEDIA, "asl_short.webm"); // 9 s of real signing
export const PATTERN_CLIP = resolve(MEDIA, "clip.webm"); // 6 s test pattern, nobody in view
export const NOT_A_VIDEO = resolve(MEDIA, "not-a-video.txt");
export const SIGNING_CLIP_MS = 16_000;

export interface ServerMessage {
  type: string;
  at: number;
  [key: string]: unknown;
}

/** Test-side observation only: records every JSON message the service sends on /stream. */
export function recordServerMessages(page: Page): ServerMessage[] {
  const messages: ServerMessage[] = [];
  page.on("websocket", (ws) => {
    ws.on("framereceived", (frame) => {
      if (typeof frame.payload !== "string") return;
      try {
        messages.push({ ...(JSON.parse(frame.payload) as { type: string }), at: Date.now() });
      } catch {
        // not JSON
      }
    });
  });
  return messages;
}

export function ofType(messages: ServerMessage[], type: string): ServerMessage[] {
  return messages.filter((m) => m.type === type);
}

export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}
