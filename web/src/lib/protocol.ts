// Wire format of the local service: ARCHITECTURE.md section 6.3 plus the details fixed by the
// service itself (src/signtoenglish/serve/ws.py and app.py). Types and parsing only.

export type SegmentId = string | number;

export interface StartMessage {
  type: "start";
  width: number;
  height: number;
  client_fps: number;
}

export interface CaptionMessage {
  type: "partial" | "final";
  segment_id: SegmentId;
  text: string;
  start_ms: number | null;
  end_ms: number | null;
  latency_ms: number | null;
  decode_ms: number | null;
  /** On a final: what ended the sentence (for example "boundary", "max_length", "flush"). */
  closed_by: string | null;
}

export interface StatusMessage {
  type: "status";
  fps_in: number | null;
  queue_depth: number | null;
  rtf: number | null;
  signing: boolean;
  frames_received: number | null;
  /** Frames the service dropped unprocessed since the session began, because it was behind. */
  frames_dropped: number | null;
  frames_processed: number | null;
  pose_fps: number | null;
}

/** A stretch of the stream the service did not translate, and why. */
export interface DroppedMessage {
  type: "dropped";
  segment_id: SegmentId | null;
  start_ms: number | null;
  end_ms: number | null;
  reason: string;
}

/** Sent once after "stop", just before the service closes the socket. */
export interface SummaryMessage {
  type: "summary";
  /** False when the service gave up waiting for the last frames or the last decode. */
  complete: boolean | null;
  stream_ms: number | null;
  frames_received: number | null;
  frames_dropped: number | null;
  frames_processed: number | null;
}

export interface ErrorMessage {
  type: "error";
  detail: string;
}

export type ServerMessage = CaptionMessage | StatusMessage | ErrorMessage | DroppedMessage | SummaryMessage;

export interface HealthResponse {
  status: "ready" | "not_ready";
  detail: string | null;
  checkpoint_id: string | null;
  preprocessing_hash: string | null;
  device: string | null;
  /** The base checkpoint's name (an addition of the service; absent from the architecture document). */
  checkpoint: string | null;
}

export interface TranslateSegment {
  text: string;
  start_ms: number;
  end_ms: number;
}

export interface SkippedStretch {
  start_ms: number | null;
  end_ms: number | null;
  reason: string;
}

export interface TranslateResponse {
  text: string;
  segments: TranslateSegment[];
  /** Stretches of the file that were not sent to the model, and why. */
  skipped: SkippedStretch[];
  timings: Record<string, number>;
  /** One line describing the model, built from the service's `model` object. */
  model: string | null;
  /** How sentence ends were found (for example "fixed_length"). */
  boundary: string | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function strOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

/**
 * One binary frame message: 8 bytes little-endian timestamp in whole milliseconds (written as a
 * 64-bit integer), then the JPEG bytes.
 */
export function encodeFrame(timestampMs: number, jpeg: ArrayBuffer): ArrayBuffer {
  const out = new ArrayBuffer(8 + jpeg.byteLength);
  new DataView(out).setBigInt64(0, BigInt(Math.round(timestampMs)), true);
  new Uint8Array(out, 8).set(new Uint8Array(jpeg));
  return out;
}

export function parseServerMessage(raw: unknown): ServerMessage | null {
  if (typeof raw !== "string") return null;
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  if (value.type === "partial" || value.type === "final") {
    const id = value.segment_id;
    if ((typeof id !== "string" && typeof id !== "number") || typeof value.text !== "string") return null;
    return {
      type: value.type,
      segment_id: id,
      text: value.text,
      start_ms: numOrNull(value.start_ms),
      end_ms: numOrNull(value.end_ms),
      latency_ms: numOrNull(value.latency_ms),
      decode_ms: numOrNull(value.decode_ms),
      closed_by: strOrNull(value.closed_by),
    };
  }
  if (value.type === "status") {
    return {
      type: "status",
      fps_in: numOrNull(value.fps_in),
      queue_depth: numOrNull(value.queue_depth),
      rtf: numOrNull(value.rtf),
      signing: value.signing === true,
      frames_received: numOrNull(value.frames_received),
      frames_dropped: numOrNull(value.frames_dropped),
      frames_processed: numOrNull(value.frames_processed),
      pose_fps: numOrNull(value.pose_fps),
    };
  }
  if (value.type === "dropped") {
    const id = value.segment_id;
    return {
      type: "dropped",
      segment_id: typeof id === "string" || typeof id === "number" ? id : null,
      start_ms: numOrNull(value.start_ms),
      end_ms: numOrNull(value.end_ms),
      reason: strOrNull(value.reason) ?? "unknown",
    };
  }
  if (value.type === "summary") {
    const frames = isRecord(value.frames) ? value.frames : {};
    return {
      type: "summary",
      complete: typeof value.complete === "boolean" ? value.complete : null,
      stream_ms: numOrNull(value.stream_ms),
      frames_received: numOrNull(frames.received),
      frames_dropped: numOrNull(frames.dropped_behind),
      frames_processed: numOrNull(frames.processed),
    };
  }
  if (value.type === "error") {
    return { type: "error", detail: detailText(value.detail) ?? "The service reported an error." };
  }
  return null;
}

export function parseHealth(value: unknown): HealthResponse | null {
  if (!isRecord(value)) return null;
  if (value.status !== "ready" && value.status !== "not_ready") return null;
  return {
    status: value.status,
    detail: strOrNull(value.detail),
    checkpoint_id: strOrNull(value.checkpoint_id),
    preprocessing_hash: strOrNull(value.preprocessing_hash),
    device: strOrNull(value.device),
    checkpoint: strOrNull(value.checkpoint),
  };
}

export function parseTranslateResponse(value: unknown): TranslateResponse | null {
  if (!isRecord(value) || typeof value.text !== "string" || !Array.isArray(value.segments)) return null;
  const segments: TranslateSegment[] = [];
  for (const item of value.segments) {
    if (!isRecord(item) || typeof item.text !== "string") return null;
    const start = numOrNull(item.start_ms);
    const end = numOrNull(item.end_ms);
    if (start === null || end === null) return null;
    segments.push({ text: item.text, start_ms: start, end_ms: end });
  }
  const timings: Record<string, number> = {};
  if (isRecord(value.timings)) {
    for (const [key, v] of Object.entries(value.timings)) {
      if (typeof v === "number" && Number.isFinite(v)) timings[key] = v;
    }
  }
  const skipped: SkippedStretch[] = [];
  if (Array.isArray(value.skipped)) {
    for (const item of value.skipped) {
      if (!isRecord(item)) continue;
      skipped.push({
        start_ms: numOrNull(item.start_ms),
        end_ms: numOrNull(item.end_ms),
        reason: strOrNull(item.reason) ?? "unknown",
      });
    }
  }
  // The service sends `model` as an object: {checkpoint, checkpoint_id, device, fine_tuned, ...}.
  let model: string | null = strOrNull(value.model);
  if (model === null && isRecord(value.model)) {
    const m = value.model;
    const parts = [
      strOrNull(m.checkpoint) ?? strOrNull(m.checkpoint_id),
      m.fine_tuned === true ? "adapted" : m.fine_tuned === false ? "not adapted" : null,
      strOrNull(m.device),
    ].filter((part): part is string => part !== null);
    model = parts.length > 0 ? parts.join(" · ") : null;
  }
  return { text: value.text, segments, skipped, timings, model, boundary: strOrNull(value.boundary) };
}

/** Turns an error body's `detail` (a string, or FastAPI's list of validation errors) into text. */
export function detailText(detail: unknown): string | null {
  if (typeof detail === "string" && detail !== "") return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => (isRecord(d) && typeof d.msg === "string" ? d.msg : null))
      .filter((d): d is string => d !== null);
    if (parts.length > 0) return parts.join("; ");
  }
  return null;
}

/** Why the service did not translate a stretch, in plain words. */
export function dropReasonText(reason: string): string {
  switch (reason) {
    case "no_signing":
      return "no signing was seen (hands not in view)";
    case "too_short":
      return "it was too short to be a sentence";
    case "empty":
      return "it held no frames";
    case "backlog":
      return "the translator was too far behind";
    default:
      return reason;
  }
}

export function formatMediaTime(ms: number): string {
  const totalSeconds = Math.max(0, ms) / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}
