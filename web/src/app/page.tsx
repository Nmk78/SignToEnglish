"use client";

import { useState } from "react";
import { CameraView } from "@/components/CameraView";
import { Captions } from "@/components/Captions";
import { DomainPanel } from "@/components/DomainPanel";
import { SpeechOutput } from "@/components/SpeechOutput";
import { StatusBar } from "@/components/StatusBar";
import { VideoInputPanel } from "@/components/VideoInputPanel";
import { useHealth } from "@/hooks/useHealth";
import { useSpeech } from "@/hooks/useSpeech";
import { useStream } from "@/hooks/useStream";

type InputMode = "camera" | "file";

export default function Home() {
  const speech = useSpeech();
  const stream = useStream({
    onSessionStart: speech.sessionStarted,
    // Only committed sentences are read aloud.
    onFinal: speech.sentenceCommitted,
    // A lost connection silences speech. A normal end does not: when a "play as live" file runs
    // out, its last sentences are still read. Pressing Stop silences speech through onUserStop.
    onSessionEnd: (reason) => {
      if (reason !== "stopped") speech.silence();
    },
  });
  const health = useHealth();
  const [mode, setMode] = useState<InputMode>("camera");
  const busy = stream.phase !== "idle";

  return (
    <main>
      <header className="topbar">
        <span className="brand">
          <span aria-hidden="true">◎</span> SignToEnglish
        </span>
        <div className="privacy">
          <span className="privacy-dot" aria-hidden="true" /> Runs on this computer · experimental demo
        </div>
      </header>

      <section className="hero" aria-labelledby="main">
        <p className="eyebrow">ASL TO ENGLISH · CONTINUOUS SIGNING · ONE DOMAIN</p>
        <h1 id="main">
          Sign a sentence. <em>Read it in English.</em>
        </h1>
        <p className="intro">
          The camera or a video file is translated sentence by sentence by a model running locally. It covers one small
          conversation domain with one signer; see the list below.
        </p>
      </section>

      <StatusBar health={health} stream={stream} />

      {stream.error && (
        <p className="alert page-alert" role="alert" data-testid="stream-error">
          {stream.error}
        </p>
      )}

      <section className="workspace" aria-label="Translation workspace">
        <section className="panel capture-panel" aria-labelledby="capture-heading">
          <div className="section-heading">
            <div>
              <p className="step">01 · INPUT</p>
              <h2 id="capture-heading">Your signing</h2>
            </div>
            <div className="mode-switch" role="tablist" aria-label="Input source">
              <button role="tab" aria-selected={mode === "camera"} className={mode === "camera" ? "selected" : ""} onClick={() => setMode("camera")} disabled={busy} data-testid="mode-camera">
                Camera
              </button>
              <button role="tab" aria-selected={mode === "file"} className={mode === "file" ? "selected" : ""} onClick={() => setMode("file")} disabled={busy} data-testid="mode-file">
                Video file
              </button>
            </div>
          </div>
          {mode === "camera" ? (
            <CameraView stream={stream} onUserStop={speech.silence} />
          ) : (
            <VideoInputPanel
              stream={stream}
              onUserStop={speech.silence}
              onSpeak={speech.availability === "ready" ? speech.speakOne : undefined}
            />
          )}
        </section>

        <Captions
          committed={stream.committed}
          partial={stream.partial}
          active={busy}
          onClear={stream.clearCaptions}
          spokenAloud={speech.active}
          controls={<SpeechOutput speech={speech} />}
        />
      </section>

      <DomainPanel />

      <footer>
        <span>
          Built on the Uni-Sign pose-only translation model, RTMPose whole-body keypoints through rtmlib, and the
          mT5-Base text model.
        </span>
        <span>Personal, non-commercial demo. Tuned to one signer and one domain.</span>
      </footer>
    </main>
  );
}
