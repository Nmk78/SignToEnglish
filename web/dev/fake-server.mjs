// DEVELOPMENT AID ONLY. NOT THE TRANSLATION SERVICE.
//
// A local fake that speaks the protocol of the real service (ARCHITECTURE.md section 6.3 and the
// details fixed in src/signtoenglish/serve/ws.py and app.py: GET /health, POST /translate,
// WS /stream) and answers with CANNED TEXT. It loads no model, looks at no
// pixels beyond the JPEG header, and its output says nothing about translation quality.
// It exists so the web app can be developed and browser-tested quickly, without the model.
// The browser tests against the real service are in tests/real/ (npm run test:real).
// Passing tests against this server does not satisfy any phase gate (CLAUDE.md, "Tests must
// exercise the real thing"); acceptance against the real service is PLAN.md Step 4.1, Wave 3.
//
// Every canned sentence starts with "[canned]" so it cannot be mistaken for model output.
//
// Usage: node dev/fake-server.mjs            (binds 127.0.0.1:8787; S2E_FAKE_PORT overrides)
//
// Test-only control endpoints, which the real service does not have:
//   GET  /__dev/sessions   what each WebSocket session received
//   GET  /__dev/last-frame the most recent JPEG received on any session (image/jpeg)
//   POST /__dev/drop       cut every open WebSocket without a close handshake
//   POST /__dev/reset      forget sessions, health back to ready
//   POST /__dev/health     body {"mode": "ready" | "not_ready"}
//   POST /__dev/captions   body {"mode": "auto" | "manual"}; in manual mode frames produce no
//                          captions and a test sends them itself with /__dev/send
//   POST /__dev/send       body {"messages": [...]}; each is sent as JSON on every open socket
//   POST /__dev/set        body with any of {"translate": "normal" | "empty",
//                          "refuse": true | false (accept, send an error, close with 1013),
//                          "drop_frames": true | false (report every 4th frame as dropped)}

import { createServer } from "node:http";
import { WebSocketServer } from "ws";

const HOST = "127.0.0.1";
const PORT = Number(process.env.S2E_FAKE_PORT ?? 8787);

const PARTIAL_EVERY_FRAMES = 5;
const FINAL_EVERY_FRAMES = 20;
const STATUS_EVERY_MS = 1000;
const ACCEPTED_EXTENSIONS = [".mp4", ".webm", ".mov"];
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;

const CANNED = [
  "[canned] Hello, how are you?",
  "[canned] Nice to meet you.",
  "[canned] Where do you live?",
  "[canned] I am learning to sign.",
];

let healthMode = "ready";
let captionMode = "auto";
let translateMode = "normal";
let refuseStreams = false;
let dropFrames = false;

const MODEL_INFO = {
  checkpoint: "FAKE-DEV-SERVER",
  checkpoint_id: "fake-0000",
  preprocessing_hash: "0000000000000000",
  device: "none",
  fine_tuned: false,
};

/** Pixel size from a JPEG's start-of-frame marker, or null. */
function jpegSize(buf) {
  let i = 2;
  while (i + 9 < buf.length) {
    if (buf[i] !== 0xff) return null;
    const marker = buf[i + 1];
    const length = buf.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      return { height: buf.readUInt16BE(i + 5), width: buf.readUInt16BE(i + 7) };
    }
    i += 2 + length;
  }
  return null;
}
/** @type {Buffer | null} */
let lastFrame = null;
let nextSessionId = 1;
/** @type {Array<Record<string, unknown>>} */
let sessions = [];
/** @type {Set<import("ws").WebSocket>} */
const sockets = new Set();

function sendJson(res, status, body) {
  const data = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(data),
    "access-control-allow-origin": "*",
  });
  res.end(data);
}

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const head = [];
    let headSize = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error("too large"));
        req.destroy();
        return;
      }
      // Only the first few kilobytes are kept: enough to read the multipart part headers.
      if (headSize < 8192) {
        head.push(chunk);
        headSize += chunk.length;
      }
    });
    req.on("end", () => resolve({ size, head: Buffer.concat(head).subarray(0, 8192) }));
    req.on("error", reject);
  });
}

function health() {
  if (healthMode === "not_ready") {
    return {
      status: "not_ready",
      detail: "fake server: not-ready mode switched on by a test",
      checkpoint_id: null,
      preprocessing_hash: null,
      device: null,
      checkpoint: null,
      active_sessions: sockets.size,
    };
  }
  return {
    status: "ready",
    detail: "development fake server, canned text, no model loaded",
    checkpoint_id: MODEL_INFO.checkpoint_id,
    preprocessing_hash: MODEL_INFO.preprocessing_hash,
    device: MODEL_INFO.device,
    checkpoint: MODEL_INFO.checkpoint,
    active_sessions: sockets.size,
  };
}

async function handleTranslate(req, res) {
  if (healthMode !== "ready") {
    sendJson(res, 503, { detail: `The translation service is not ready: ${health().detail}` });
    return;
  }
  const contentType = String(req.headers["content-type"] ?? "");
  let body;
  try {
    body = await readBody(req, MAX_UPLOAD_BYTES);
  } catch {
    sendJson(res, 413, { detail: "Upload is too large." });
    return;
  }
  if (!contentType.startsWith("multipart/form-data")) {
    sendJson(res, 400, { detail: "Expected a multipart upload with a `video` file field." });
    return;
  }
  const headText = body.head.toString("latin1");
  const match = /name="video"; filename="([^"]*)"/.exec(headText);
  if (!match) {
    sendJson(res, 400, { detail: "Expected a multipart upload with a `video` file field." });
    return;
  }
  const filename = match[1] ?? "";
  const lower = filename.toLowerCase();
  if (!ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
    sendJson(res, 415, { detail: `Unsupported file '${filename}'. Use an MP4, WebM, or MOV video.` });
    return;
  }
  const empty = translateMode === "empty";
  const segments = empty
    ? []
    : [
        { text: CANNED[0], start_ms: 400, end_ms: 2600, segment_id: 0, closed_by: "max_length" },
        { text: CANNED[1], start_ms: 3300, end_ms: 5400, segment_id: 1, closed_by: "flush" },
      ];
  sendJson(res, 200, {
    text: segments.map((s) => s.text).join(" "),
    segments,
    skipped: empty ? [{ start_ms: 0, end_ms: 6000, reason: "no_signing" }] : [],
    timings: { total_ms: 0, receive_ms: 0, keypoints_ms: 0, translate_ms: 0, video_ms: 6000, n_frames: 180, n_segments: segments.length, real_time_factor: 0 },
    keypoints: "extracted",
    boundary: "fixed_length",
    model: MODEL_INFO,
  });
}

const server = createServer((req, res) => {
  const url = new URL(req.url ?? "/", `http://${HOST}`);
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET, POST, OPTIONS",
      "access-control-allow-headers": "content-type",
    });
    res.end();
    return;
  }
  if (req.method === "GET" && url.pathname === "/health") {
    const body = health();
    sendJson(res, body.status === "ready" ? 200 : 503, body);
    return;
  }
  if (req.method === "POST" && url.pathname === "/translate") {
    void handleTranslate(req, res);
    return;
  }
  if (req.method === "GET" && url.pathname === "/__dev/sessions") {
    sendJson(res, 200, sessions);
    return;
  }
  if (req.method === "GET" && url.pathname === "/__dev/last-frame") {
    if (lastFrame === null) {
      sendJson(res, 404, { detail: "No frame received yet." });
      return;
    }
    res.writeHead(200, {
      "content-type": "image/jpeg",
      "content-length": lastFrame.length,
      "access-control-allow-origin": "*",
    });
    res.end(lastFrame);
    return;
  }
  if (req.method === "POST" && url.pathname === "/__dev/drop") {
    const n = sockets.size;
    for (const ws of sockets) ws.terminate();
    sendJson(res, 200, { dropped: n });
    return;
  }
  if (req.method === "POST" && url.pathname === "/__dev/reset") {
    for (const ws of sockets) ws.terminate();
    sessions = [];
    healthMode = "ready";
    captionMode = "auto";
    translateMode = "normal";
    refuseStreams = false;
    dropFrames = false;
    lastFrame = null;
    sendJson(res, 200, { ok: true });
    return;
  }
  if (req.method === "POST" && url.pathname === "/__dev/set") {
    readBody(req, 4096).then(
      (body) => {
        try {
          const parsed = JSON.parse(body.head.toString("utf8"));
          if (parsed.translate === "normal" || parsed.translate === "empty") translateMode = parsed.translate;
          if (typeof parsed.refuse === "boolean") refuseStreams = parsed.refuse;
          if (typeof parsed.drop_frames === "boolean") dropFrames = parsed.drop_frames;
          sendJson(res, 200, { translate: translateMode, refuse: refuseStreams, drop_frames: dropFrames });
        } catch {
          sendJson(res, 400, { detail: "bad request body" });
        }
      },
      () => sendJson(res, 400, { detail: "bad body" }),
    );
    return;
  }
  if (req.method === "POST" && (url.pathname === "/__dev/captions" || url.pathname === "/__dev/send")) {
    readBody(req, 65536).then(
      (body) => {
        try {
          const parsed = JSON.parse(body.head.toString("utf8"));
          if (url.pathname === "/__dev/captions") {
            if (parsed.mode !== "auto" && parsed.mode !== "manual") throw new Error("bad mode");
            captionMode = parsed.mode;
            sendJson(res, 200, { mode: captionMode });
            return;
          }
          if (!Array.isArray(parsed.messages)) throw new Error("bad messages");
          let sent = 0;
          for (const ws of sockets) {
            if (ws.readyState !== ws.OPEN) continue;
            for (const message of parsed.messages) ws.send(JSON.stringify(message));
            sent += 1;
          }
          sendJson(res, 200, { sockets: sent, messages: parsed.messages.length });
        } catch {
          sendJson(res, 400, { detail: "bad request body" });
        }
      },
      () => sendJson(res, 400, { detail: "bad body" }),
    );
    return;
  }
  if (req.method === "POST" && url.pathname === "/__dev/health") {
    readBody(req, 4096).then(
      (body) => {
        try {
          const mode = JSON.parse(body.head.toString("utf8")).mode;
          if (mode !== "ready" && mode !== "not_ready") throw new Error("bad mode");
          healthMode = mode;
          sendJson(res, 200, { mode });
        } catch {
          sendJson(res, 400, { detail: 'body must be {"mode": "ready" | "not_ready"}' });
        }
      },
      () => sendJson(res, 400, { detail: "bad body" }),
    );
    return;
  }
  sendJson(res, 404, { detail: "Not found" });
});

const wss = new WebSocketServer({ noServer: true });

server.on("upgrade", (req, socket, head) => {
  const url = new URL(req.url ?? "/", `http://${HOST}`);
  if (url.pathname !== "/stream") {
    socket.destroy();
    return;
  }
  wss.handleUpgrade(req, socket, head, (ws) => wss.emit("connection", ws));
});

wss.on("connection", (ws) => {
  if (refuseStreams || healthMode !== "ready") {
    // The real service refuses the same way: accept, say why, close with 1013.
    const detail = refuseStreams
      ? "The translation service already has 2 open stream(s); close one and try again."
      : `The translation service is not ready: ${health().detail}`;
    ws.send(JSON.stringify({ type: "error", detail }));
    ws.close(1013);
    return;
  }
  sockets.add(ws);
  const record = {
    id: nextSessionId++,
    start: null,
    frames: 0,
    bad_frames: 0,
    wrong_size_frames: 0,
    reported_dropped: 0,
    non_increasing_timestamps: 0,
    first_ts_ms: null,
    last_ts_ms: null,
    first_frame_wall_ms: null,
    last_frame_wall_ms: null,
    min_jpeg_bytes: null,
    max_jpeg_bytes: null,
    stop_received: false,
    finals_sent: 0,
    partials_sent: 0,
    closed: false,
  };
  sessions.push(record);

  let segmentId = 0;
  let segmentFrames = 0;
  let segmentStartTs = null;
  let framesSinceStatus = 0;

  const send = (obj) => {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
  };

  const emit = (type, closedBy = "max_length") => {
    const sentence = CANNED[segmentId % CANNED.length];
    let text = sentence;
    if (type === "partial") {
      const words = sentence.split(" ");
      const shown = Math.min(words.length, 1 + Math.floor(segmentFrames / PARTIAL_EVERY_FRAMES));
      text = words.slice(0, shown).join(" ");
      record.partials_sent += 1;
    } else {
      record.finals_sent += 1;
    }
    send({
      type,
      segment_id: segmentId,
      text,
      start_ms: segmentStartTs,
      end_ms: record.last_ts_ms,
      latency_ms: 0,
      decode_ms: 0,
      ...(type === "final" ? { closed_by: closedBy } : {}),
    });
  };

  const statusTimer = setInterval(() => {
    send({
      type: "status",
      fps_in: framesSinceStatus / (STATUS_EVERY_MS / 1000),
      queue_depth: 0,
      rtf: 0,
      signing: segmentFrames > 0,
      frames_received: record.frames + record.bad_frames,
      frames_dropped: record.reported_dropped,
      frames_processed: record.frames - record.reported_dropped,
      pose_fps: framesSinceStatus / (STATUS_EVERY_MS / 1000),
    });
    framesSinceStatus = 0;
  }, STATUS_EVERY_MS);

  ws.on("message", (data, isBinary) => {
    if (!isBinary) {
      let msg;
      try {
        msg = JSON.parse(data.toString("utf8"));
      } catch {
        send({ type: "error", detail: "Text message was not JSON." });
        return;
      }
      if (msg.type === "start") {
        record.start = msg;
        return;
      }
      if (msg.type === "stop") {
        record.stop_received = true;
        if (segmentFrames > 0 && captionMode === "auto") emit("final", "flush");
        // Like the real service: a summary, then the service closes the socket itself.
        send({
          type: "summary",
          complete: true,
          stream_ms: record.first_ts_ms === null ? 0 : record.last_ts_ms - record.first_ts_ms,
          frames: {
            received: record.frames + record.bad_frames,
            dropped_behind: record.reported_dropped,
            rejected_bad: record.bad_frames,
            rejected_non_increasing: record.non_increasing_timestamps,
            processed: record.frames - record.reported_dropped,
          },
        });
        ws.close(1000, "stopped");
        return;
      }
      send({ type: "error", detail: `Unknown message type ${JSON.stringify(msg.type)}.` });
      return;
    }

    if (record.start === null) {
      send({ type: "error", detail: "Frame received before the start message; it was not used." });
      return;
    }
    const buf = /** @type {Buffer} */ (data);
    // 8 bytes little-endian timestamp in milliseconds, then one JPEG (starts FF D8, ends FF D9).
    const isJpeg =
      buf.length > 12 &&
      buf[8] === 0xff &&
      buf[9] === 0xd8 &&
      buf[buf.length - 2] === 0xff &&
      buf[buf.length - 1] === 0xd9;
    if (!isJpeg) {
      record.bad_frames += 1;
      send({ type: "error", detail: "Frame was not 8 timestamp bytes followed by a JPEG." });
      return;
    }
    const ts = Number(buf.readBigInt64LE(0));
    if (record.last_ts_ms !== null && ts <= record.last_ts_ms) {
      record.non_increasing_timestamps += 1;
      send({ type: "error", detail: `frame timestamp ${ts} does not come after the previous one (${record.last_ts_ms}); the frame was not used` });
      return;
    }
    // The real service refuses a frame whose pixel size differs from the start message.
    const size = jpegSize(buf.subarray(8));
    if (!size || size.width !== record.start.width || size.height !== record.start.height) {
      record.wrong_size_frames += 1;
      record.bad_frames += 1;
      send({ type: "error", detail: `frame is ${size?.width}x${size?.height} but the start message said ${record.start.width}x${record.start.height}; the frame was not used` });
      return;
    }
    const now = Date.now();
    if (record.first_ts_ms === null) {
      record.first_ts_ms = ts;
      record.first_frame_wall_ms = now;
    }
    record.last_ts_ms = ts;
    record.last_frame_wall_ms = now;
    const jpegBytes = buf.length - 8;
    record.min_jpeg_bytes = record.min_jpeg_bytes === null ? jpegBytes : Math.min(record.min_jpeg_bytes, jpegBytes);
    record.max_jpeg_bytes = record.max_jpeg_bytes === null ? jpegBytes : Math.max(record.max_jpeg_bytes, jpegBytes);
    record.frames += 1;
    if (dropFrames && record.frames % 4 === 0) record.reported_dropped += 1;
    lastFrame = Buffer.from(buf.subarray(8));
    framesSinceStatus += 1;

    if (captionMode !== "auto") return;
    if (segmentFrames === 0) segmentStartTs = ts;
    segmentFrames += 1;
    if (segmentFrames >= FINAL_EVERY_FRAMES) {
      emit("final");
      segmentId += 1;
      segmentFrames = 0;
      segmentStartTs = null;
    } else if (segmentFrames % PARTIAL_EVERY_FRAMES === 0) {
      emit("partial");
    }
  });

  ws.on("close", () => {
    clearInterval(statusTimer);
    sockets.delete(ws);
    record.closed = true;
  });
  ws.on("error", () => {});
});

server.listen(PORT, HOST, () => {
  console.log(`FAKE development server (canned text, not the translation service) on http://${HOST}:${PORT}`);
});
