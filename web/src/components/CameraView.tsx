"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { StreamApi } from "@/hooks/useStream";
import { config } from "@/lib/config";

type CameraState = "off" | "requesting" | "on";

export const CAMERA_MESSAGES = {
  denied:
    "Camera permission was refused. Allow camera access for this page in the browser's site settings, then press Start camera again.",
  notFound: "No camera was found. Connect a camera, then press Start camera again.",
  busy: "The camera could not be opened. Another application may be using it.",
  unsupported: "This browser cannot open a camera from this page.",
  other: "The camera could not be started.",
} as const;

function cameraErrorText(error: unknown): string {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return CAMERA_MESSAGES.denied;
  if (name === "NotFoundError" || name === "OverconstrainedError") return CAMERA_MESSAGES.notFound;
  if (name === "NotReadableError" || name === "AbortError") return CAMERA_MESSAGES.busy;
  return CAMERA_MESSAGES.other;
}

/** Camera permission, preview with a framing guide, and start/stop of the live stream. */
export function CameraView({ stream }: { stream: StreamApi }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRef = useRef<MediaStream | null>(null);
  const [cameraState, setCameraState] = useState<CameraState>("off");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [aspect, setAspect] = useState<string>("16 / 9");

  const releaseCamera = useCallback(() => {
    mediaRef.current?.getTracks().forEach((track) => track.stop());
    mediaRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => releaseCamera, [releaseCamera]);

  async function handleStart() {
    setCameraError(null);
    setWarning(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(CAMERA_MESSAGES.unsupported);
      return;
    }
    setCameraState("requesting");
    let media: MediaStream;
    try {
      media = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: config.camera.width },
          height: { ideal: config.camera.height },
          frameRate: { ideal: config.camera.frameRate },
        },
      });
    } catch (error) {
      setCameraError(cameraErrorText(error));
      setCameraState("off");
      return;
    }
    const video = videoRef.current;
    if (!video) {
      media.getTracks().forEach((track) => track.stop());
      setCameraState("off");
      return;
    }
    mediaRef.current = media;
    video.srcObject = media;
    try {
      await video.play();
    } catch {
      releaseCamera();
      setCameraError(CAMERA_MESSAGES.other);
      setCameraState("off");
      return;
    }
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      setAspect(`${video.videoWidth} / ${video.videoHeight}`);
      if (video.videoWidth < config.camera.minWidth || video.videoHeight < config.camera.minHeight) {
        setWarning(
          `The camera gives ${video.videoWidth}×${video.videoHeight}, below the ${config.camera.minWidth}×${config.camera.minHeight} minimum. Results may be worse.`,
        );
      }
    }
    setCameraState("on");
    // The camera goes off whenever the session ends, whether stopped or lost.
    const connected = await stream.start({ video, clock: "capture" }, () => {
      releaseCamera();
      setCameraState("off");
    });
    if (!connected) {
      releaseCamera();
      setCameraState("off");
    }
  }

  function handleStop() {
    stream.stop();
    releaseCamera();
    setCameraState("off");
  }

  const running = cameraState !== "off" || stream.phase === "connecting" || stream.phase === "streaming";
  const startDisabled = cameraState === "requesting" || stream.phase !== "idle";

  return (
    <div className="camera-view">
      <div className="camera-frame" style={{ aspectRatio: aspect }}>
        {/* The preview is mirrored with CSS only; the frames sent to the service are not mirrored. */}
        <video ref={videoRef} className="camera-video mirrored" muted playsInline aria-label="Camera preview" data-testid="camera-video" />
        {cameraState !== "on" && (
          <div className="camera-empty">
            <strong>{cameraState === "requesting" ? "Waiting for camera permission…" : "Camera is off"}</strong>
            <small>One signer, upper body in view, face and both hands visible</small>
          </div>
        )}
        <svg className="framing-guide" viewBox="0 0 160 90" preserveAspectRatio="none" aria-hidden="true" data-testid="framing-guide">
          <rect x="22" y="5" width="116" height="82" rx="4" />
          <ellipse cx="80" cy="27" rx="11" ry="14" />
          <path d="M 42 87 C 42 58, 60 50, 80 50 C 100 50, 118 58, 118 87" />
        </svg>
        {cameraState === "on" && <p className="framing-hint">Keep your face and both hands inside the box</p>}
      </div>

      <div className="controls">
        {running ? (
          <button className="stop-button" onClick={handleStop} data-testid="camera-stop">
            <span aria-hidden="true">■</span> Stop
          </button>
        ) : (
          <button className="record-button" onClick={handleStart} disabled={startDisabled} data-testid="camera-start">
            <span aria-hidden="true">●</span> Start camera
          </button>
        )}
      </div>

      {cameraError && (
        <p className="alert" role="alert" data-testid="camera-error">
          {cameraError}
        </p>
      )}
      {warning && <p className="warning">{warning}</p>}
    </div>
  );
}
