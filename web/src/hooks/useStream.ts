"use client";

// Connection hook for WS /stream: socket lifecycle, the frame pump, back-pressure, and the
// caption state. One session at a time. The webcam and "play as live" both go through start().

import { useCallback, useEffect, useRef, useState } from "react";
import { config } from "@/lib/config";
import { type ClockKind, FramePump, type PumpStats, ZERO_STATS, frameSize } from "@/lib/framePump";
import {
  type SegmentId,
  type StartMessage,
  type StatusMessage,
  type SummaryMessage,
  parseServerMessage,
} from "@/lib/protocol";

export type StreamPhase = "idle" | "connecting" | "streaming" | "stopping";

/**
 * Why a session ended: the user (or the end of the file) stopped it; the connection was lost;
 * the service could not be reached; or the service ended the stream itself and said why.
 */
export type EndReason = "stopped" | "lost" | "unreachable" | "refused";

/** Stretches the service reported as not translated in this session. */
export interface NotTranslated {
  count: number;
  lastReason: string;
}

export interface Caption {
  key: number;
  segmentId: SegmentId;
  text: string;
}

export interface StreamSource {
  /** A video element that is ready to play (metadata loaded, so its frame size is known). */
  video: HTMLVideoElement;
  clock: ClockKind;
}

export interface StreamApi {
  phase: StreamPhase;
  committed: Caption[];
  partial: Caption | null;
  serverStatus: StatusMessage | null;
  /** True while the service's dropped-frame count is still rising. */
  dropping: boolean;
  notTranslated: NotTranslated | null;
  /** The service's summary of the last session that ended with "stop". */
  summary: SummaryMessage | null;
  /** Pixel size of the frames being sent (or last sent). */
  frame: { width: number; height: number } | null;
  clientStats: PumpStats;
  error: string | null;
  /** Resolves true once the socket is open and frames are flowing, false if it could not connect. */
  start: (source: StreamSource, onEnd?: (reason: EndReason) => void) => Promise<boolean>;
  /** Stops sending, asks the service to flush the open segment, then closes. */
  stop: () => void;
  clearCaptions: () => void;
}

/** Hook-level callbacks, for things that follow every session (read-aloud). */
export interface StreamEvents {
  onSessionStart?: () => void;
  /** Called once for each committed ("final") sentence. Partial text never reaches it. */
  onFinal?: (text: string) => void;
  onSessionEnd?: (reason: EndReason) => void;
}

interface Session {
  ws: WebSocket;
  pump: FramePump | null;
  opened: boolean;
  stopping: boolean;
  graceTimer: number | null;
  statsTimer: number | null;
  lastDropped: number;
  lastServerError: { detail: string; at: number } | null;
}

/** A close this soon after an "error" message is taken to be caused by it. */
const ERROR_THEN_CLOSE_MS = 3000;
/** WebSocket close code the service uses when it refuses a stream ("try again later"). */
const CLOSE_TRY_LATER = 1013;

export const MESSAGES = {
  unreachable: `Could not connect to the translation service at ${config.serviceUrl}. Check that it is running (and that it allows pages from this address), then press start again.`,
  lost: "The connection to the translation service was lost. Captioning has stopped; press start to reconnect.",
} as const;

function teardown(session: Session): void {
  session.pump?.stop();
  if (session.graceTimer !== null) window.clearTimeout(session.graceTimer);
  if (session.statsTimer !== null) window.clearInterval(session.statsTimer);
  session.graceTimer = null;
  session.statsTimer = null;
}

export function useStream(events: StreamEvents = {}): StreamApi {
  const [phase, setPhase] = useState<StreamPhase>("idle");
  const [committed, setCommitted] = useState<Caption[]>([]);
  const [partial, setPartial] = useState<Caption | null>(null);
  const [serverStatus, setServerStatus] = useState<StatusMessage | null>(null);
  const [dropping, setDropping] = useState(false);
  const [notTranslated, setNotTranslated] = useState<NotTranslated | null>(null);
  const [summary, setSummary] = useState<SummaryMessage | null>(null);
  const [frame, setFrame] = useState<{ width: number; height: number } | null>(null);
  const [clientStats, setClientStats] = useState<PumpStats>(ZERO_STATS);
  const [error, setError] = useState<string | null>(null);
  const sessionRef = useRef<Session | null>(null);
  const keyRef = useRef(0);
  const eventsRef = useRef<StreamEvents>(events);
  useEffect(() => {
    eventsRef.current = events;
  });

  const start = useCallback(
    (source: StreamSource, onEnd?: (reason: EndReason) => void) =>
      new Promise<boolean>((resolve) => {
        if (sessionRef.current) {
          resolve(false);
          return;
        }
        setError(null);
        setPartial(null);
        setServerStatus(null);
        setDropping(false);
        setNotTranslated(null);
        setSummary(null);
        setClientStats(ZERO_STATS);
        setPhase("connecting");

        let ws: WebSocket;
        try {
          ws = new WebSocket(config.streamUrl);
        } catch {
          setPhase("idle");
          setError(MESSAGES.unreachable);
          onEnd?.("unreachable");
          resolve(false);
          return;
        }
        ws.binaryType = "arraybuffer";
        const session: Session = {
          ws,
          pump: null,
          opened: false,
          stopping: false,
          graceTimer: null,
          statsTimer: null,
          lastDropped: 0,
          lastServerError: null,
        };
        sessionRef.current = session;

        ws.onopen = () => {
          session.opened = true;
          if (session.stopping) {
            ws.close(1000);
            return;
          }
          const size = frameSize(source.video, config.maxFrameWidth);
          setFrame(size);
          const hello: StartMessage = { type: "start", width: size.width, height: size.height, client_fps: config.sendFps };
          ws.send(JSON.stringify(hello));
          const pump = new FramePump({
            video: source.video,
            clock: source.clock,
            fps: config.sendFps,
            width: size.width,
            height: size.height,
            jpegQuality: config.jpegQuality,
            bufferedBytes: () => ws.bufferedAmount,
            maxBufferedBytes: config.maxBufferedBytes,
            sink: (message) => {
              if (ws.readyState === WebSocket.OPEN && !session.stopping) ws.send(message);
            },
          });
          session.pump = pump;
          pump.start();
          session.statsTimer = window.setInterval(() => setClientStats(pump.stats()), 500);
          setPhase("streaming");
          eventsRef.current.onSessionStart?.();
          resolve(true);
        };

        ws.onmessage = (event: MessageEvent) => {
          const message = parseServerMessage(event.data);
          if (!message) return;
          if (message.type === "status") {
            setServerStatus(message);
            const dropped = message.frames_dropped ?? 0;
            setDropping(dropped > session.lastDropped);
            session.lastDropped = dropped;
          } else if (message.type === "error") {
            session.lastServerError = { detail: message.detail, at: performance.now() };
            setError(`The translation service reported a problem: ${message.detail}`);
          } else if (message.type === "dropped") {
            // A stretch the service did not translate. The stream goes on.
            const reason = message.reason;
            setPartial((p) => (p && p.segmentId === message.segment_id ? null : p));
            setNotTranslated((prev) => ({ count: (prev?.count ?? 0) + 1, lastReason: reason }));
          } else if (message.type === "summary") {
            setSummary(message);
          } else if (message.type === "partial") {
            // Display-only text for the open segment; replaced by every later partial.
            setPartial({ key: -1, segmentId: message.segment_id, text: message.text });
          } else {
            const segmentId = message.segment_id;
            const text = message.text.trim();
            setPartial((p) => (p && p.segmentId === segmentId ? null : p));
            if (text !== "") {
              keyRef.current += 1;
              const caption: Caption = { key: keyRef.current, segmentId, text };
              // Committed text is only ever appended, never edited.
              setCommitted((prev) => [...prev, caption].slice(-config.maxCommittedCaptions));
              eventsRef.current.onFinal?.(text);
            }
          }
        };

        ws.onclose = (event: CloseEvent) => {
          teardown(session);
          if (sessionRef.current === session) sessionRef.current = null;
          if (session.pump) setClientStats(session.pump.stats());
          setPartial(null);
          setServerStatus(null);
          setDropping(false);
          setPhase("idle");
          let reason: EndReason = "stopped";
          if (!session.stopping) {
            const said = session.lastServerError;
            if (!session.opened) {
              reason = "unreachable";
              setError(MESSAGES.unreachable);
            } else if (said && (event.code === CLOSE_TRY_LATER || performance.now() - said.at < ERROR_THEN_CLOSE_MS)) {
              // The service said why before it closed; keep its words.
              reason = "refused";
              setError(`The translation service ended the stream: ${said.detail}`);
            } else {
              reason = "lost";
              setError(MESSAGES.lost);
            }
          }
          onEnd?.(reason);
          eventsRef.current.onSessionEnd?.(reason);
          if (!session.opened) resolve(false);
        };
      }),
    [],
  );

  const stop = useCallback(() => {
    const session = sessionRef.current;
    if (!session || session.stopping) return;
    session.stopping = true;
    session.pump?.stop();
    setPhase("stopping");
    const ws = session.ws;
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
      // The service finishes the frames it holds, sends the last "final" and a "summary", and
      // closes the socket itself; that close is what ends the session. The timer is a fallback.
      session.graceTimer = window.setTimeout(() => ws.close(1000), config.stopGraceMs);
    } else if (ws.readyState === WebSocket.CONNECTING) {
      // onopen closes it as soon as the handshake finishes.
    } else {
      ws.close();
    }
  }, []);

  const clearCaptions = useCallback(() => {
    setCommitted([]);
    setPartial(null);
  }, []);

  useEffect(
    () => () => {
      const session = sessionRef.current;
      if (!session) return;
      sessionRef.current = null;
      teardown(session);
      session.ws.onopen = null;
      session.ws.onmessage = null;
      session.ws.onclose = null;
      session.ws.close();
    },
    [],
  );

  return {
    phase,
    committed,
    partial,
    serverStatus,
    dropping,
    notTranslated,
    summary,
    frame,
    clientStats,
    error,
    start,
    stop,
    clearCaptions,
  };
}
