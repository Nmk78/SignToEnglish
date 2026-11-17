"use client";

import { RATE_MAX, RATE_MIN, RATE_STEP, type SpeechApi } from "@/hooks/useSpeech";

export const SPEECH_MESSAGES = {
  noLocalVoice:
    "No speech voice installed on this computer can be used offline, so sentences will not be read aloud. Captions still appear. Add a voice in the system's spoken-content settings, then reload this page.",
  unsupported: "This browser has no built-in speech, so sentences will not be read aloud. Captions still appear.",
  nonEnglish: "No English voice is installed on this computer, so another language's voice is reading the English text.",
} as const;

/** Controls for reading committed sentences aloud: on/off, local voice, rate, and what it is doing. */
export function SpeechOutput({ speech }: { speech: SpeechApi }) {
  const { availability, settings, queue } = speech;
  const usable = availability === "ready";
  const skipped = queue.skipped;

  let status = "";
  if (usable && settings.enabled) {
    if (queue.speaking !== null) status = queue.waiting > 0 ? `Speaking · ${queue.waiting} waiting` : "Speaking";
    else status = "Each committed sentence is read aloud. Text still in progress is never spoken.";
  } else if (usable) {
    status = "Read-aloud is off.";
  } else if (availability === "checking") {
    status = "Looking for a voice on this computer…";
  }

  return (
    <div className="speech-output" data-testid="speech-output">
      <div className="speech-row">
        <label className="speech-toggle">
          <input
            type="checkbox"
            checked={settings.enabled}
            onChange={(event) => speech.setEnabled(event.target.checked)}
            data-testid="speech-toggle"
          />
          Read sentences aloud
        </label>
        <label className="speech-field">
          <span>Voice</span>
          <select
            value={speech.voice?.voiceURI ?? ""}
            onChange={(event) => speech.setVoiceUri(event.target.value)}
            disabled={!usable}
            data-testid="speech-voice"
          >
            {speech.voices.length === 0 && <option value="">No local voice</option>}
            {speech.voices.map((voice) => (
              <option key={voice.voiceURI} value={voice.voiceURI}>
                {voice.name} ({voice.lang})
              </option>
            ))}
          </select>
        </label>
        <label className="speech-field">
          <span>Speed {settings.rate.toFixed(1)}×</span>
          <input
            type="range"
            min={RATE_MIN}
            max={RATE_MAX}
            step={RATE_STEP}
            value={settings.rate}
            onChange={(event) => speech.setRate(Number(event.target.value))}
            disabled={!usable}
            data-testid="speech-rate"
          />
        </label>
      </div>

      {status !== "" && (
        <p className="speech-status" data-testid="speech-status">
          {status}
        </p>
      )}
      {skipped > 0 && (
        <p className="warning" role="status" data-testid="speech-skipped">
          {skipped === 1 ? "1 sentence was" : `${skipped} sentences were`} not read aloud, to keep up with the signing.
          All of them are in the captions.
        </p>
      )}
      {availability === "no-local-voice" && (
        <p className="warning" role="status" data-testid="speech-unavailable">
          {SPEECH_MESSAGES.noLocalVoice}
        </p>
      )}
      {availability === "unsupported" && (
        <p className="warning" role="status" data-testid="speech-unavailable">
          {SPEECH_MESSAGES.unsupported}
        </p>
      )}
      {usable && speech.nonEnglishOnly && <p className="warning">{SPEECH_MESSAGES.nonEnglish}</p>}
    </div>
  );
}
