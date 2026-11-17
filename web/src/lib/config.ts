// The one place the web app reads its settings. NEXT_PUBLIC_* values are fixed at build time.

function unitFromEnv(value: string | undefined, fallback: number): number {
  const n = value === undefined || value === "" ? NaN : Number(value);
  return Number.isFinite(n) && n > 0 && n <= 1 ? n : fallback;
}

function numberFromEnv(value: string | undefined, fallback: number): number {
  if (value === undefined || value === "") return fallback;
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

const serviceUrl = (process.env.NEXT_PUBLIC_S2E_SERVICE_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

export const config = {
  /** Base URL of the local translation service. Port 8000 is an assumed default (uvicorn's). */
  serviceUrl,
  /** WebSocket endpoint derived from the service URL. */
  streamUrl: `${serviceUrl.replace(/^http/, "ws")}/stream`,
  /**
   * Frames per second sent to the service. 10 is the user's decision of 2026-11-17 (PLAN.md
   * section 0): measured on this Mac, the service kept up at 10 and dropped 28% of frames at 15.
   * The send rate changes what the model sees, so it is fixed at build time
   * (NEXT_PUBLIC_S2E_SEND_FPS) and shown in the status bar.
   */
  sendFps: numberFromEnv(process.env.NEXT_PUBLIC_S2E_SEND_FPS, 10),
  /** Frames wider than this are scaled down before JPEG encoding. */
  maxFrameWidth: numberFromEnv(process.env.NEXT_PUBLIC_S2E_MAX_FRAME_WIDTH, 1280),
  /**
   * JPEG quality of the frames sent, 0 to 1. Like the send rate it changes what the model sees
   * (Step 3.3 finding), so it is fixed at build time (NEXT_PUBLIC_S2E_JPEG_QUALITY) and shown in
   * the status bar.
   */
  jpegQuality: unitFromEnv(process.env.NEXT_PUBLIC_S2E_JPEG_QUALITY, 0.8),
  /** A frame is skipped while more than this many bytes wait in the socket's send buffer. */
  maxBufferedBytes: 1_000_000,
  /**
   * After "stop" the service finishes the frames it holds, sends the last sentence and a summary,
   * and closes the socket itself. This is only the fallback for a service that never closes; it
   * is longer than the service's own 30 s stop timeout.
   */
  stopGraceMs: 35_000,
  healthPollMs: 3000,
  /** Camera request from ARCHITECTURE.md 6.2. */
  camera: { width: 1280, height: 720, frameRate: 30, minWidth: 640, minHeight: 480 },
  maxCommittedCaptions: 200,
} as const;
