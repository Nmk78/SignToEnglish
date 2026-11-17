"use client";

import { type ChangeEvent, useEffect, useRef, useState } from "react";
import type { StreamApi } from "@/hooks/useStream";
import { config } from "@/lib/config";
import {
  type TranslateResponse,
  detailText,
  dropReasonText,
  formatMediaTime,
  parseTranslateResponse,
} from "@/lib/protocol";

type UploadState =
  | { kind: "idle" }
  | { kind: "working" }
  | { kind: "done"; result: TranslateResponse }
  | { kind: "error"; message: string };

export const VIDEO_MESSAGES = {
  unreachable: `Could not reach the translation service at ${config.serviceUrl}. Check that it is running, then try again.`,
  badResponse: "The translation service answered in an unexpected format.",
  unplayable:
    "This browser cannot play the file, so “Play as live” is unavailable for it. “Translate file” may still work.",
} as const;

/**
 * Video-file input, also the backup path at the show. Two options for the chosen file:
 * "Translate file" uploads it to POST /translate and lists each sentence with its time range;
 * "Play as live" plays it in the page and streams its frames, stamped with their media time,
 * through the same hook and frame pump as the webcam.
 *
 * Read-aloud: "Play as live" is read aloud like the webcam. A "Translate file" result is not
 * read automatically; each sentence has its own Speak button.
 */
interface VideoInputPanelProps {
  stream: StreamApi;
  /** Called when the user presses Stop during "play as live" (not when the file simply ends). */
  onUserStop?: () => void;
  /** Speaks one sentence of a "translate file" result; absent when no local voice exists. */
  onSpeak?: (text: string) => void;
}

export function VideoInputPanel({ stream, onUserStop, onSpeak }: VideoInputPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const urlRef = useRef<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [upload, setUpload] = useState<UploadState>({ kind: "idle" });
  const [live, setLive] = useState(false);
  const [playError, setPlayError] = useState<string | null>(null);

  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    },
    [],
  );

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const chosen = event.target.files?.[0];
    if (!chosen) return;
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    const nextUrl = URL.createObjectURL(chosen);
    urlRef.current = nextUrl;
    setFile(chosen);
    setUrl(nextUrl);
    setUpload({ kind: "idle" });
    setPlayError(null);
  }

  async function translateFile() {
    if (!file) return;
    setUpload({ kind: "working" });
    const form = new FormData();
    form.append("video", file, file.name);
    let response: Response;
    try {
      response = await fetch(`${config.serviceUrl}/translate`, { method: "POST", body: form });
    } catch {
      setUpload({ kind: "error", message: VIDEO_MESSAGES.unreachable });
      return;
    }
    const body: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = body && typeof body === "object" ? detailText((body as { detail?: unknown }).detail) : null;
      setUpload({
        kind: "error",
        message: detail ?? `The translation service refused the file (HTTP ${response.status}).`,
      });
      return;
    }
    const result = parseTranslateResponse(body);
    setUpload(result ? { kind: "done", result } : { kind: "error", message: VIDEO_MESSAGES.badResponse });
  }

  async function playAsLive() {
    const video = videoRef.current;
    if (!video || !file) return;
    setPlayError(null);
    video.pause();
    if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
      const loaded = await new Promise<boolean>((resolve) => {
        video.addEventListener("loadedmetadata", () => resolve(true), { once: true });
        video.addEventListener("error", () => resolve(false), { once: true });
      });
      if (!loaded) return; // the element's error handler shows the message
    }
    video.currentTime = 0;
    setLive(true);
    const connected = await stream.start({ video, clock: "media" }, () => {
      video.pause();
      setLive(false);
    });
    if (!connected) {
      setLive(false);
      return;
    }
    try {
      // Playback starts only once the socket is open, so the first frames of the file are sent.
      await video.play();
    } catch {
      stream.stop();
      setPlayError(VIDEO_MESSAGES.unplayable);
    }
  }

  function seekTo(ms: number) {
    const video = videoRef.current;
    if (video && !live) video.currentTime = ms / 1000;
  }

  const streamBusy = stream.phase !== "idle";
  const disabled = !file || streamBusy || upload.kind === "working";

  return (
    <div className="video-input">
      <div className="video-row">
        <div className="camera-frame file-frame">
          {url ? (
            <video
              ref={videoRef}
              key={url}
              src={url}
              className="camera-video"
              muted
              playsInline
              preload="auto"
              controls={!live}
              onEnded={() => {
                if (live) stream.stop();
              }}
              onError={() => setPlayError(VIDEO_MESSAGES.unplayable)}
              aria-label="Chosen video"
              data-testid="file-video"
            />
          ) : (
            <div className="camera-empty">
              <strong>No video chosen</strong>
              <small>MP4, WebM, or MOV with one or several sentences</small>
            </div>
          )}
        </div>

        {upload.kind === "done" && (
          <div className="segments" data-testid="translate-result">
            <p className="result-label">SENTENCES IN THIS FILE</p>
            {upload.result.segments.length === 0 ? (
              <p className="segments-empty" data-testid="translate-empty">
                No sentence was found in this video.
              </p>
            ) : (
              <ol>
                {upload.result.segments.map((segment, index) => (
                  <li key={`${segment.start_ms}-${index}`} data-testid="translate-segment">
                    <button className="segment-time" onClick={() => seekTo(segment.start_ms)} title="Jump to this sentence">
                      {formatMediaTime(segment.start_ms)} – {formatMediaTime(segment.end_ms)}
                    </button>
                    <span className="segment-text">{segment.text}</span>
                    {onSpeak && segment.text.trim() !== "" && (
                      <button className="segment-speak" onClick={() => onSpeak(segment.text)} data-testid="segment-speak">
                        Speak
                      </button>
                    )}
                  </li>
                ))}
              </ol>
            )}
            {upload.result.skipped.length > 0 && (
              <ul className="segments-skipped" data-testid="translate-skipped">
                {upload.result.skipped.map((stretch, index) => (
                  <li key={index}>
                    {stretch.start_ms !== null && stretch.end_ms !== null
                      ? `${formatMediaTime(stretch.start_ms)} – ${formatMediaTime(stretch.end_ms)}`
                      : "A stretch"}{" "}
                    was not translated because {dropReasonText(stretch.reason)}.
                  </li>
                ))}
              </ul>
            )}
            {upload.result.model && (
              <p className="segments-meta" data-testid="translate-model">
                Model: {upload.result.model}
                {upload.result.boundary ? ` · sentence ends: ${upload.result.boundary.replace(/_/g, " ")}` : ""}
              </p>
            )}
            {Object.keys(upload.result.timings).length > 0 && (
              <p className="segments-meta">
                {Object.entries(upload.result.timings)
                  .map(([name, value]) => `${name}: ${Math.round(value)}`)
                  .join(" · ")}
              </p>
            )}
          </div>
        )}
      </div>

      <div className="controls">
        <label className={`upload-button ${streamBusy ? "disabled" : ""}`}>
          {file ? "Choose another video" : "Choose video"}
          <input
            type="file"
            accept="video/mp4,video/webm,video/quicktime,.mp4,.webm,.mov"
            onChange={handleFile}
            disabled={streamBusy}
            data-testid="file-input"
          />
        </label>
        <button className="primary-button" onClick={translateFile} disabled={disabled} data-testid="translate-file">
          {upload.kind === "working" ? "Translating…" : "Translate file"}
        </button>
        {live ? (
          <button
            className="stop-button"
            onClick={() => {
              onUserStop?.();
              stream.stop();
            }}
            disabled={stream.phase === "stopping"} data-testid="live-stop">
            <span aria-hidden="true">■</span> {stream.phase === "stopping" ? "Finishing…" : "Stop"}
          </button>
        ) : (
          <button className="record-button" onClick={playAsLive} disabled={disabled || playError !== null} data-testid="play-as-live">
            <span aria-hidden="true">▶</span> Play as live
          </button>
        )}
      </div>
      <p className="format-note">
        {file ? file.name : "Nothing is uploaded anywhere except the translation service on this computer."}
      </p>

      {upload.kind === "error" && (
        <p className="alert" role="alert" data-testid="translate-error">
          {upload.message}
        </p>
      )}
      {playError && (
        <p className="alert" role="alert" data-testid="play-error">
          {playError}
        </p>
      )}
    </div>
  );
}
