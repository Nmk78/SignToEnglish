// Queue for reading committed sentences aloud with the browser's built-in speech synthesis
// (Web Speech API). One sentence is spoken at a time. Nothing here loads a model or uses the
// network: the caller only ever hands it a voice whose `localService` is true.

/**
 * How many sentences may wait behind the one being spoken. When one more arrives, every waiting
 * sentence except the newest is dropped and counted as skipped, so speech never drifts far behind
 * the signer. The sentence being spoken is allowed to finish.
 */
export const MAX_WAITING = 2;

/** If the browser never reports the end of an utterance, move on after this long. */
const WATCHDOG_MS = 20_000;

export interface SpeechQueueState {
  /** Text being spoken now, or null. */
  speaking: string | null;
  waiting: number;
  /** Sentences dropped by the skip-ahead rule since the last resetSkipped(). */
  skipped: number;
}

export const IDLE_SPEECH: SpeechQueueState = { speaking: null, waiting: 0, skipped: 0 };

export class SpeechQueue {
  voice: SpeechSynthesisVoice | null = null;
  rate = 1;

  private waiting: string[] = [];
  private current: SpeechSynthesisUtterance | null = null;
  private currentText: string | null = null;
  private generation = 0;
  private skipped = 0;
  private watchdog: number | null = null;

  constructor(
    private readonly synth: SpeechSynthesis,
    private readonly onChange: (state: SpeechQueueState) => void,
  ) {}

  /** Adds one sentence. Silent when no local voice has been set. */
  enqueue(text: string): void {
    const sentence = text.trim();
    if (!this.voice || sentence === "") return;
    this.waiting.push(sentence);
    if (this.waiting.length > MAX_WAITING) {
      this.skipped += this.waiting.length - 1;
      this.waiting = [sentence];
    }
    this.pump();
    this.emit();
  }

  /** Drops everything and speaks this one sentence (an explicit button press). */
  speakNow(text: string): void {
    this.cancel();
    this.enqueue(text);
  }

  /** Stops the current sentence and empties the queue. */
  cancel(): void {
    this.generation += 1;
    this.waiting = [];
    this.current = null;
    this.currentText = null;
    this.clearWatchdog();
    this.synth.cancel();
    this.emit();
  }

  resetSkipped(): void {
    this.skipped = 0;
    this.emit();
  }

  private pump(): void {
    if (this.current || !this.voice) return;
    const text = this.waiting.shift();
    if (text === undefined) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = this.voice;
    utterance.lang = this.voice.lang;
    utterance.rate = this.rate;
    const generation = this.generation;
    const finished = () => {
      // Ignore events from an utterance that cancel() already replaced.
      if (generation !== this.generation || this.current !== utterance) return;
      this.current = null;
      this.currentText = null;
      this.clearWatchdog();
      this.pump();
      this.emit();
    };
    utterance.onend = finished;
    utterance.onerror = finished;
    // Keeping the utterance referenced until it ends also keeps its events from being lost.
    this.current = utterance;
    this.currentText = text;
    this.watchdog = window.setTimeout(() => {
      if (this.current !== utterance) return;
      this.synth.cancel();
      finished();
    }, WATCHDOG_MS);
    this.synth.speak(utterance);
  }

  private clearWatchdog(): void {
    if (this.watchdog !== null) window.clearTimeout(this.watchdog);
    this.watchdog = null;
  }

  private emit(): void {
    this.onChange({ speaking: this.currentText, waiting: this.waiting.length, skipped: this.skipped });
  }
}

/** Voices that run on this computer. Network voices are never offered or used. */
export function localVoices(all: SpeechSynthesisVoice[]): SpeechSynthesisVoice[] {
  return all.filter((voice) => voice.localService === true);
}

export function isEnglish(voice: SpeechSynthesisVoice): boolean {
  return voice.lang.toLowerCase().replace("_", "-").startsWith("en");
}

/**
 * The voices to offer: local English voices (the captions are English); if this computer has no
 * local English voice, every local voice, so that there is still speech.
 */
export function offeredVoices(all: SpeechSynthesisVoice[]): SpeechSynthesisVoice[] {
  const local = localVoices(all);
  const english = local.filter(isEnglish);
  return english.length > 0 ? english : local;
}

/** The remembered voice if it is still offered, else the system default, else the first offered. */
export function pickVoice(offered: SpeechSynthesisVoice[], preferredUri: string | null): SpeechSynthesisVoice | null {
  return (
    offered.find((voice) => voice.voiceURI === preferredUri) ??
    offered.find((voice) => voice.default) ??
    offered.find((voice) => voice.lang.toLowerCase().replace("_", "-") === "en-us") ??
    offered[0] ??
    null
  );
}
