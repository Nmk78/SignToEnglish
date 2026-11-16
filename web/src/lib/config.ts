// The one place the web app reads its settings. NEXT_PUBLIC_* values are fixed at build time.

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
   * Frames per second sent to the service. ARCHITECTURE.md 6.2 says the browser sends at the
   * model frame rate, which is "per base model" and not known to this track. 15 is an ASSUMED
   * placeholder; set NEXT_PUBLIC_S2E_SEND_FPS from docs/base_model_notes.md at Wave 3. The
   * service reduces by timestamp again, so a higher value costs bandwidth, not correctness.
   */
  sendFps: numberFromEnv(process.env.NEXT_PUBLIC_S2E_SEND_FPS, 15),
  /** Frames wider than this are scaled down before JPEG encoding. */
  maxFrameWidth: numberFromEnv(process.env.NEXT_PUBLIC_S2E_MAX_FRAME_WIDTH, 1280),
  jpegQuality: 0.8,
  /** A frame is skipped while more than this many bytes wait in the socket's send buffer. */
  maxBufferedBytes: 1_000_000,
  /** After "stop", how long to wait for the service to flush and close before closing ourselves. */
  stopGraceMs: 4000,
  healthPollMs: 3000,
  /** Camera request from ARCHITECTURE.md 6.2. */
  camera: { width: 1280, height: 720, frameRate: 30, minWidth: 640, minHeight: 480 },
  maxCommittedCaptions: 200,
} as const;
