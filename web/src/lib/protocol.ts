// Wire format of the local service, from ARCHITECTURE.md section 6.3. Types and parsing only.

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
}

export interface StatusMessage {
  type: "status";
  fps_in: number | null;
  queue_depth: number | null;
  rtf: number | null;
  signing: boolean;
}

export interface ErrorMessage {
  type: "error";
  detail: string;
}

export type ServerMessage = CaptionMessage | StatusMessage | ErrorMessage;

export interface HealthResponse {
  status: "ready" | "not_ready";
  detail: string | null;
  checkpoint_id: string | null;
  preprocessing_hash: string | null;
  device: string | null;
}

export interface TranslateSegment {
  text: string;
  start_ms: number;
  end_ms: number;
}

export interface TranslateResponse {
  text: string;
  segments: TranslateSegment[];
  timings: Record<string, number>;
  model: string | null;
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
    };
  }
  if (value.type === "status") {
    return {
      type: "status",
      fps_in: numOrNull(value.fps_in),
      queue_depth: numOrNull(value.queue_depth),
      rtf: numOrNull(value.rtf),
      signing: value.signing === true,
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
  // The shape of `model` is not fixed by the architecture document; show a string if there is one.
  let model: string | null = strOrNull(value.model);
  if (model === null && isRecord(value.model)) model = strOrNull(value.model.checkpoint_id);
  return { text: value.text, segments, timings, model };
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

export function formatMediaTime(ms: number): string {
  const totalSeconds = Math.max(0, ms) / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}
