// TEST-SIDE INSTRUMENTATION of the browser's real speech API, shared by the fake-server and the
// real-service browser tests. Nothing in the page is replaced: `speechSynthesis.speak` and
// `speechSynthesis.cancel` are wrapped so that each call is logged and then passed on to the real
// function, `start` / `end` / `error` listeners are added to each utterance, and the arrival time
// of each "final" WebSocket message is noted.
import type { Page } from "@playwright/test";

export interface SpokenEntry {
  text: string;
  voiceUri: string | null;
  voiceName: string | null;
  localService: boolean | null;
  lang: string;
  rate: number;
  calledAt: number;
  events: { type: string; at: number; error?: string }[];
}
export interface SpeechLog {
  spoken: SpokenEntry[];
  cancels: number;
  finalsReceived: { text: string; at: number }[];
}

/** Test-side instrumentation; see the note at the top of this file. */
export async function instrumentSpeech(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const log: SpeechLog = { spoken: [], cancels: 0, finalsReceived: [] };
    (window as unknown as { __speechLog: SpeechLog }).__speechLog = log;
    const synth = window.speechSynthesis;
    const realSpeak = synth.speak.bind(synth);
    const realCancel = synth.cancel.bind(synth);
    synth.speak = (utterance: SpeechSynthesisUtterance) => {
      const entry: SpokenEntry = {
        text: utterance.text,
        voiceUri: utterance.voice?.voiceURI ?? null,
        voiceName: utterance.voice?.name ?? null,
        localService: utterance.voice?.localService ?? null,
        lang: utterance.lang,
        rate: utterance.rate,
        calledAt: performance.now(),
        events: [],
      };
      log.spoken.push(entry);
      for (const type of ["start", "end", "error"] as const) {
        utterance.addEventListener(type, (event) => {
          const error = (event as SpeechSynthesisErrorEvent).error;
          entry.events.push({ type, at: performance.now(), ...(error ? { error } : {}) });
        });
      }
      realSpeak(utterance);
    };
    synth.cancel = () => {
      log.cancels += 1;
      realCancel();
    };
    // Also note when each "final" message reaches the page, to measure the delay to speech.
    const RealWebSocket = window.WebSocket;
    window.WebSocket = class extends RealWebSocket {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols);
        this.addEventListener("message", (event: MessageEvent) => {
          if (typeof event.data !== "string") return;
          try {
            const message = JSON.parse(event.data) as { type?: string; text?: string };
            if (message.type === "final") log.finalsReceived.push({ text: message.text ?? "", at: performance.now() });
          } catch {
            // not JSON; nothing to note
          }
        });
      }
    };
  });
}

export async function speechLog(page: Page): Promise<SpeechLog> {
  return page.evaluate(() => (window as unknown as { __speechLog: SpeechLog }).__speechLog);
}

export async function isSpeaking(page: Page): Promise<boolean> {
  return page.evaluate(() => window.speechSynthesis.speaking);
}
