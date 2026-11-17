"use client";

// Read-aloud of committed sentences (ARCHITECTURE.md 6.4, SpeechOutput). Holds the settings,
// the list of local voices, and one SpeechQueue. Only final sentences are ever handed to it.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { IDLE_SPEECH, SpeechQueue, type SpeechQueueState, isEnglish, offeredVoices, pickVoice } from "@/lib/speechQueue";

export const RATE_MIN = 0.6;
export const RATE_MAX = 1.6;
export const RATE_STEP = 0.1;

export interface SpeechSettings {
  /** Default is on: the point of the feature is that a listener hears the sentence. */
  enabled: boolean;
  voiceUri: string | null;
  rate: number;
}

const DEFAULT_SETTINGS: SpeechSettings = { enabled: true, voiceUri: null, rate: 1 };
/** Remembered for the browser session (sessionStorage), not across sessions. */
export const SETTINGS_KEY = "s2e.speech";
/** Chromium fills the voice list a moment after page load; wait this long before saying "none". */
const VOICE_WAIT_MS = 1500;

function loadSettings(): SpeechSettings {
  try {
    const raw = window.sessionStorage.getItem(SETTINGS_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const value = JSON.parse(raw) as Partial<SpeechSettings>;
    const rate = typeof value.rate === "number" && value.rate >= RATE_MIN && value.rate <= RATE_MAX ? value.rate : 1;
    return {
      enabled: typeof value.enabled === "boolean" ? value.enabled : DEFAULT_SETTINGS.enabled,
      voiceUri: typeof value.voiceUri === "string" ? value.voiceUri : null,
      rate,
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function saveSettings(settings: SpeechSettings): void {
  try {
    window.sessionStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Storage can be unavailable; the choices then last until the page is reloaded.
  }
}

export type SpeechAvailability = "checking" | "ready" | "no-local-voice" | "unsupported";

export interface SpeechApi {
  availability: SpeechAvailability;
  /** Local voices offered in the picker. */
  voices: SpeechSynthesisVoice[];
  voice: SpeechSynthesisVoice | null;
  /** True when the only local voices are not English. */
  nonEnglishOnly: boolean;
  settings: SpeechSettings;
  queue: SpeechQueueState;
  /** True when committed sentences are being read aloud (on, and a local voice exists). */
  active: boolean;
  setEnabled: (enabled: boolean) => void;
  setVoiceUri: (uri: string) => void;
  setRate: (rate: number) => void;
  /** A stream session started: accept sentences again and reset the skipped count. */
  sessionStarted: () => void;
  /** A committed sentence arrived from the stream. */
  sentenceCommitted: (text: string) => void;
  /** Stop speaking, empty the queue, and stay silent until the next session starts. */
  silence: () => void;
  /** Stop speaking and empty the queue; sentences committed after this are still read. */
  interrupt: () => void;
  /** Speak one sentence now because the user pressed its button (works with read-aloud off). */
  speakOne: (text: string) => void;
}

export function useSpeech(): SpeechApi {
  const [settings, setSettings] = useState<SpeechSettings>(DEFAULT_SETTINGS);
  const [allVoices, setAllVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [checked, setChecked] = useState(false);
  const [supported, setSupported] = useState(true);
  const [queue, setQueue] = useState<SpeechQueueState>(IDLE_SPEECH);
  const queueRef = useRef<SpeechQueue | null>(null);
  const enabledRef = useRef(DEFAULT_SETTINGS.enabled);
  const acceptingRef = useRef(false);

  useEffect(() => {
    if (typeof window.speechSynthesis === "undefined" || typeof window.SpeechSynthesisUtterance === "undefined") {
      const t = window.setTimeout(() => {
        setSupported(false);
        setChecked(true);
      }, 0);
      return () => window.clearTimeout(t);
    }
    const synth = window.speechSynthesis;
    queueRef.current = new SpeechQueue(synth, setQueue);
    const readVoices = () => {
      const voices = synth.getVoices();
      setAllVoices(voices);
      if (voices.length > 0) setChecked(true);
    };
    synth.addEventListener("voiceschanged", readVoices);
    const first = window.setTimeout(() => {
      const loaded = loadSettings();
      enabledRef.current = loaded.enabled;
      setSettings(loaded);
      readVoices();
    }, 0);
    const giveUp = window.setTimeout(() => setChecked(true), VOICE_WAIT_MS);
    return () => {
      synth.removeEventListener("voiceschanged", readVoices);
      window.clearTimeout(first);
      window.clearTimeout(giveUp);
      queueRef.current?.cancel();
      queueRef.current = null;
    };
  }, []);

  const voices = useMemo(() => offeredVoices(allVoices), [allVoices]);
  const voice = useMemo(() => pickVoice(voices, settings.voiceUri), [voices, settings.voiceUri]);

  useEffect(() => {
    const q = queueRef.current;
    if (!q) return;
    q.voice = voice;
    q.rate = settings.rate;
  }, [voice, settings.rate]);

  const update = useCallback((change: Partial<SpeechSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...change };
      saveSettings(next);
      return next;
    });
  }, []);

  const setEnabled = useCallback(
    (enabled: boolean) => {
      enabledRef.current = enabled;
      if (!enabled) queueRef.current?.cancel();
      update({ enabled });
    },
    [update],
  );
  const setVoiceUri = useCallback((voiceUri: string) => update({ voiceUri }), [update]);
  const setRate = useCallback((rate: number) => update({ rate }), [update]);

  const sessionStarted = useCallback(() => {
    acceptingRef.current = true;
    queueRef.current?.resetSkipped();
  }, []);

  const sentenceCommitted = useCallback((text: string) => {
    if (!enabledRef.current || !acceptingRef.current) return;
    queueRef.current?.enqueue(text);
  }, []);

  const silence = useCallback(() => {
    acceptingRef.current = false;
    queueRef.current?.cancel();
  }, []);

  const interrupt = useCallback(() => {
    queueRef.current?.cancel();
  }, []);

  const speakOne = useCallback((text: string) => {
    queueRef.current?.speakNow(text);
  }, []);

  let availability: SpeechAvailability = "checking";
  if (!supported) availability = "unsupported";
  else if (voice) availability = "ready";
  else if (checked) availability = "no-local-voice";

  return {
    availability,
    voices,
    voice,
    nonEnglishOnly: voices.length > 0 && !voices.some(isEnglish),
    settings,
    queue,
    active: settings.enabled && availability === "ready",
    setEnabled,
    setVoiceUri,
    setRate,
    sessionStarted,
    sentenceCommitted,
    silence,
    interrupt,
    speakOne,
  };
}
