// Grabs frames from a playing <video>, reduces them to the send rate by timestamp, encodes each
// as JPEG, and hands "timestamp + JPEG" messages to a sink. The webcam and "play as live" use
// this same class; the only difference is which clock stamps the frame.

import { encodeFrame } from "./protocol";

/**
 * "capture": the frame's capture time on the wall clock, in milliseconds since the Unix epoch
 *            (webcam). "media": the frame's presentation time inside the file (play as live).
 */
export type ClockKind = "capture" | "media";

export interface FramePumpOptions {
  video: HTMLVideoElement;
  clock: ClockKind;
  fps: number;
  width: number;
  height: number;
  jpegQuality: number;
  /** Bytes currently waiting in the transport's send buffer. */
  bufferedBytes: () => number;
  maxBufferedBytes: number;
  sink: (message: ArrayBuffer) => void;
}

export interface PumpStats {
  framesSent: number;
  /** Frames due for sending but skipped because the previous frame was still being encoded. */
  skippedBusy: number;
  /** Frames due for sending but skipped because the socket's send buffer was full. */
  skippedBackpressure: number;
  /** Measured send rate over the last two seconds. */
  sendFps: number;
}

export const ZERO_STATS: PumpStats = { framesSent: 0, skippedBusy: 0, skippedBackpressure: 0, sendFps: 0 };

const RATE_WINDOW_MS = 2000;
// Media timestamps are floats (66.666 for the frame due at 66.667); allow that much slack.
const DUE_SLACK_MS = 2;

export function frameSize(video: HTMLVideoElement, maxWidth: number): { width: number; height: number } {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (w <= maxWidth) return { width: w, height: h };
  return { width: maxWidth, height: Math.round((h * maxWidth) / w) };
}

export class FramePump {
  private readonly opts: FramePumpOptions;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D | null;
  private readonly intervalMs: number;
  private running = false;
  private busy = false;
  private callbackHandle: number | null = null;
  private timerHandle: number | null = null;
  private nextDueMs: number | null = null;
  private lastSentTs: number | null = null;
  private sentWallTimes: number[] = [];
  private framesSent = 0;
  private skippedBusy = 0;
  private skippedBackpressure = 0;

  constructor(opts: FramePumpOptions) {
    this.opts = opts;
    this.intervalMs = 1000 / opts.fps;
    this.canvas = document.createElement("canvas");
    this.canvas.width = opts.width;
    this.canvas.height = opts.height;
    this.ctx = this.canvas.getContext("2d", { alpha: false });
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.schedule();
  }

  stop(): void {
    this.running = false;
    const video = this.opts.video;
    if (this.callbackHandle !== null && "cancelVideoFrameCallback" in video) {
      video.cancelVideoFrameCallback(this.callbackHandle);
    }
    if (this.timerHandle !== null) window.clearTimeout(this.timerHandle);
    this.callbackHandle = null;
    this.timerHandle = null;
  }

  stats(): PumpStats {
    const now = performance.now();
    this.sentWallTimes = this.sentWallTimes.filter((t) => now - t <= RATE_WINDOW_MS);
    const n = this.sentWallTimes.length;
    const first = this.sentWallTimes[0];
    const last = this.sentWallTimes[n - 1];
    const sendFps = n >= 2 && first !== undefined && last !== undefined && last > first ? ((n - 1) * 1000) / (last - first) : 0;
    return {
      framesSent: this.framesSent,
      skippedBusy: this.skippedBusy,
      skippedBackpressure: this.skippedBackpressure,
      sendFps,
    };
  }

  private schedule(): void {
    if (!this.running) return;
    const video = this.opts.video;
    if ("requestVideoFrameCallback" in video) {
      // Fires once per frame the video element presents, with that frame's own timestamps.
      this.callbackHandle = video.requestVideoFrameCallback((now, meta) => {
        this.callbackHandle = null;
        const ts =
          this.opts.clock === "media"
            ? meta.mediaTime * 1000
            : performance.timeOrigin + (meta.captureTime ?? meta.expectedDisplayTime ?? now);
        this.onFrame(ts);
        this.schedule();
      });
      return;
    }
    // Fallback for browsers without per-frame callbacks: poll at twice the send rate.
    const el: HTMLVideoElement = video;
    this.timerHandle = window.setTimeout(() => {
      this.timerHandle = null;
      if (!el.paused && !el.ended && el.readyState >= 2) {
        this.onFrame(this.opts.clock === "media" ? el.currentTime * 1000 : performance.timeOrigin + performance.now());
      }
      this.schedule();
    }, this.intervalMs / 2);
  }

  private onFrame(tsMs: number): void {
    if (!this.running) return;
    if (this.nextDueMs !== null && tsMs < this.nextDueMs - DUE_SLACK_MS) return; // rate reduction
    const rounded = Math.round(tsMs);
    // Timestamps on the wire are whole milliseconds and must be strictly increasing.
    if (this.lastSentTs !== null && rounded <= this.lastSentTs) return;
    if (this.busy) {
      this.skippedBusy += 1;
      return;
    }
    if (this.opts.bufferedBytes() > this.opts.maxBufferedBytes) {
      this.skippedBackpressure += 1;
      return;
    }
    const due = this.nextDueMs === null ? tsMs : this.nextDueMs;
    this.nextDueMs = due + this.intervalMs;
    if (this.nextDueMs < tsMs) this.nextDueMs = tsMs + this.intervalMs;
    this.lastSentTs = rounded;
    void this.grab(rounded);
  }

  private async grab(tsMs: number): Promise<void> {
    if (!this.ctx) return;
    this.busy = true;
    try {
      // The pixels are copied now, in the same task as the frame callback, so they are the frame
      // the timestamp belongs to. CSS mirroring of the preview does not affect drawImage.
      this.ctx.drawImage(this.opts.video, 0, 0, this.canvas.width, this.canvas.height);
      const blob = await new Promise<Blob | null>((resolve) =>
        this.canvas.toBlob(resolve, "image/jpeg", this.opts.jpegQuality),
      );
      if (!blob || !this.running) return;
      const jpeg = await blob.arrayBuffer();
      if (!this.running) return;
      this.opts.sink(encodeFrame(tsMs, jpeg));
      this.framesSent += 1;
      this.sentWallTimes.push(performance.now());
    } finally {
      this.busy = false;
    }
  }
}
