// Read-aloud of committed sentences.
//
// WHAT THESE TESTS OBSERVE. The page's own code is not replaced. Before the page loads, the test
// installs INSTRUMENTATION around the browser's real speech API: `speechSynthesis.speak` and
// `speechSynthesis.cancel` are wrapped so that each call is logged and then passed on to the real
// function, and `start` / `end` / `error` listeners are added to each utterance. So the tests see
// (a) exactly what the page handed to speech synthesis, and (b) the real events and the real
// `speechSynthesis.speaking` flag that this browser reports for it. On this Mac, headless Chromium
// exposes the macOS system voices and fires real start and end events, and the Mac does talk
// while these tests run. The tests cannot hear: that sound really reaches a listener through the
// speakers can only be confirmed by a person.
//
// One test (no local voice) goes further and REPLACES `speechSynthesis.getVoices` with a list in
// which every voice is marked as a network voice, because that situation cannot be produced on
// this machine otherwise. That test says so again where it does it.
//
// The fake server is in manual caption mode here: the test decides which partial and final
// messages arrive, and when.
import { type APIRequestContext, type Page, expect, test } from "@playwright/test";
import { CLIP_FILE, FAKE, resetFake } from "./helpers";
import { type SpokenEntry, instrumentSpeech, isSpeaking, speechLog } from "./speech-instrument";

async function send(request: APIRequestContext, messages: object[]): Promise<void> {
  const response = await request.post(`${FAKE}/__dev/send`, { data: { messages } });
  expect(((await response.json()) as { sockets: number }).sockets).toBe(1);
}

function caption(type: "partial" | "final", segmentId: number, text: string) {
  return { type, segment_id: segmentId, text, start_ms: 0, end_ms: 1000, latency_ms: 0 };
}

function eventTypes(entry: SpokenEntry): string[] {
  return entry.events.map((e) => e.type);
}

async function startCamera(page: Page): Promise<void> {
  await page.getByTestId("camera-start").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");
}

test.beforeEach(async ({ page, request }) => {
  await resetFake(request);
  await request.post(`${FAKE}/__dev/captions`, { data: { mode: "manual" } });
  await instrumentSpeech(page);
});

test("a committed sentence is spoken exactly once with the chosen local voice and rate; partial text is never spoken", async ({ page, request }) => {
  await page.goto("/");
  // On by default.
  await expect(page.getByTestId("speech-toggle")).toBeChecked();

  // Every voice offered is a local one, and they are English.
  const picker = page.getByTestId("speech-voice");
  await expect(picker).toBeEnabled();
  const offered = await picker.locator("option").evaluateAll((options) => options.map((o) => (o as HTMLOptionElement).value));
  const system = await page.evaluate(() =>
    window.speechSynthesis.getVoices().map((v) => ({ uri: v.voiceURI, local: v.localService, lang: v.lang })),
  );
  expect(offered.length).toBeGreaterThan(1);
  for (const uri of offered) {
    const voice = system.find((v) => v.uri === uri);
    expect(voice?.local).toBe(true);
    expect(voice?.lang.toLowerCase().startsWith("en")).toBe(true);
  }
  console.log(`[measured] system voices: ${system.length}, local: ${system.filter((v) => v.local).length}, offered (local English): ${offered.length}`);

  // Choose a voice other than the preselected one, and a rate other than the default.
  const preselected = await picker.inputValue();
  const chosen = offered.find((uri) => uri !== preselected)!;
  await picker.selectOption(chosen);
  await page.getByTestId("speech-rate").fill("1.3");

  await startCamera(page);

  // Partial text: shown, never spoken.
  await send(request, [caption("partial", 0, "Hello how")]);
  await expect(page.getByTestId("caption-partial")).toContainText("Hello how");
  await send(request, [caption("partial", 0, "Hello how are")]);
  await expect(page.getByTestId("caption-partial")).toContainText("Hello how are");
  await page.waitForTimeout(700);
  expect((await speechLog(page)).spoken).toHaveLength(0);
  expect(await isSpeaking(page)).toBe(false);

  // Final text: spoken.
  await send(request, [caption("final", 0, "Hello, how are you?")]);
  await expect(page.getByTestId("caption-final")).toHaveText(["Hello, how are you?"]);
  await expect.poll(async () => eventTypes((await speechLog(page)).spoken[0] ?? ({ events: [] } as unknown as SpokenEntry))).toContain("start");
  expect(await isSpeaking(page)).toBe(true); // the browser's own flag
  await expect(page.getByTestId("speech-status")).toContainText("Speaking");
  // While the page speaks, the caption list is not a live region (no second reading by a screen reader).
  await expect(page.getByTestId("captions-final")).toHaveAttribute("aria-live", "off");

  await expect.poll(async () => eventTypes((await speechLog(page)).spoken[0]!), { timeout: 20_000 }).toContain("end");
  await page.waitForTimeout(1000);
  const log = await speechLog(page);
  expect(log.spoken).toHaveLength(1); // exactly once
  const entry = log.spoken[0]!;
  expect(entry.text).toBe("Hello, how are you?");
  expect(entry.voiceUri).toBe(chosen);
  expect(entry.localService).toBe(true);
  expect(entry.rate).toBeCloseTo(1.3, 5);
  expect(eventTypes(entry)).toEqual(["start", "end"]);
  expect(await isSpeaking(page)).toBe(false);

  const received = log.finalsReceived[0]!.at;
  const started = entry.events[0]!.at;
  const ended = entry.events[1]!.at;
  console.log(
    `[measured] voice "${entry.voiceName}" rate ${entry.rate}: final message -> speak() call ${(entry.calledAt - received).toFixed(1)} ms, ` +
      `final message -> start event ${(started - received).toFixed(1)} ms, speaking lasted ${(ended - started).toFixed(0)} ms`,
  );

  // The choices are remembered for the session.
  await page.getByTestId("camera-stop").click();
  await page.reload();
  await expect(page.getByTestId("speech-voice")).toHaveValue(chosen);
  await expect(page.getByTestId("speech-rate")).toHaveValue("1.3");
});

test("with read-aloud switched off nothing is spoken, and switching it off mid-sentence stops speech", async ({ page, request }) => {
  await page.goto("/");
  await page.getByTestId("speech-toggle").uncheck();
  await startCamera(page);
  await send(request, [caption("final", 0, "Nice to meet you.")]);
  await expect(page.getByTestId("caption-final")).toHaveText(["Nice to meet you."]);
  await page.waitForTimeout(1000);
  let log = await speechLog(page);
  expect(log.spoken).toHaveLength(0);
  expect(await isSpeaking(page)).toBe(false);
  // Off: the caption list is an ordinary polite live region again.
  await expect(page.getByTestId("captions-final")).toHaveAttribute("aria-live", "polite");

  // Switch on, start a sentence and queue another, then switch off while the first is being spoken.
  await page.getByTestId("speech-toggle").check();
  await send(request, [
    caption("final", 1, "Where do you live? I am learning to sign, and I want to meet your family today."),
    caption("final", 2, "This sentence is still waiting."),
  ]);
  await expect.poll(async () => eventTypes((await speechLog(page)).spoken[0] ?? ({ events: [] } as unknown as SpokenEntry))).toContain("start");
  await page.getByTestId("speech-toggle").uncheck();
  await expect.poll(() => isSpeaking(page)).toBe(false);
  await page.waitForTimeout(1000);
  log = await speechLog(page);
  expect(log.spoken).toHaveLength(1); // the waiting sentence was dropped, never spoken
  expect(log.cancels).toBeGreaterThanOrEqual(1);
  expect(eventTypes(log.spoken[0]!)).not.toContain("end"); // cut off, not finished
  console.log(`[measured] switch off mid-sentence: utterance events ${JSON.stringify(log.spoken[0]!.events.map((e) => e.error ?? e.type))}`);
  // Sentences that arrive while it is off stay silent.
  await send(request, [caption("final", 3, "Still silent.")]);
  await expect(page.getByTestId("caption-final")).toHaveCount(4);
  await page.waitForTimeout(700);
  expect((await speechLog(page)).spoken).toHaveLength(1);
});

test("pressing Stop cancels speech and empties the queue", async ({ page, request }) => {
  await page.goto("/");
  await startCamera(page);
  await send(request, [
    caption("final", 0, "Where do you live? I am learning to sign, and I want to meet your family today."),
    caption("final", 1, "This sentence is still waiting."),
  ]);
  await expect.poll(async () => eventTypes((await speechLog(page)).spoken[0] ?? ({ events: [] } as unknown as SpokenEntry))).toContain("start");
  expect(await isSpeaking(page)).toBe(true);

  await page.getByTestId("camera-stop").click();
  await expect.poll(() => isSpeaking(page)).toBe(false);
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  await page.waitForTimeout(1000);
  const log = await speechLog(page);
  expect(log.spoken).toHaveLength(1);
  expect(eventTypes(log.spoken[0]!)).not.toContain("end");
  // Both sentences are still in the captions.
  await expect(page.getByTestId("caption-final")).toHaveCount(2);
});

test("the sentence the service commits after Stop is still read aloud", async ({ page, request }) => {
  // Automatic captions: on "stop" the fake server, like the real one, sends the open sentence as
  // a last final, then a summary, then closes.
  await request.post(`${FAKE}/__dev/captions`, { data: { mode: "auto" } });
  await page.goto("/");
  await startCamera(page);
  await expect(page.getByTestId("caption-partial")).toBeVisible();
  expect((await speechLog(page)).spoken).toHaveLength(0);
  await page.getByTestId("camera-stop").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  await expect(page.getByTestId("caption-final")).toHaveText(["[canned] Hello, how are you?"]);
  await expect.poll(async () => eventTypes((await speechLog(page)).spoken[0] ?? ({ events: [] } as unknown as SpokenEntry)), { timeout: 20_000 }).toEqual(["start", "end"]);
  const log = await speechLog(page);
  expect(log.spoken.map((s) => s.text)).toEqual(["[canned] Hello, how are you?"]);
});

test("a dropped connection cancels speech and empties the queue", async ({ page, request }) => {
  await page.goto("/");
  await startCamera(page);
  await send(request, [
    caption("final", 0, "Where do you live? I am learning to sign, and I want to meet your family today."),
    caption("final", 1, "This sentence is still waiting."),
  ]);
  await expect.poll(async () => eventTypes((await speechLog(page)).spoken[0] ?? ({ events: [] } as unknown as SpokenEntry))).toContain("start");

  await request.post(`${FAKE}/__dev/drop`);
  await expect(page.getByTestId("stream-error")).toContainText("connection to the translation service was lost");
  await expect.poll(() => isSpeaking(page)).toBe(false);
  await page.waitForTimeout(1000);
  const log = await speechLog(page);
  expect(log.spoken).toHaveLength(1);
  expect(eventTypes(log.spoken[0]!)).not.toContain("end");
});

test("when speech falls behind it skips ahead to the newest sentence and says so", async ({ page, request }) => {
  await page.goto("/");
  await startCamera(page);
  // Six sentences arrive at once. The first starts; at most two may wait behind it.
  const texts = ["One.", "Two.", "Three.", "Four.", "Five.", "Six."];
  await send(request, texts.map((text, i) => caption("final", i, text)));
  await expect(page.getByTestId("caption-final")).toHaveText(texts); // captions keep every sentence

  // 2 wait (Two, Three); Four makes 3, so Two and Three are dropped and Four waits; Five waits;
  // Six makes 3, so Four and Five are dropped. Spoken: One, then Six. Skipped: four sentences.
  await expect(page.getByTestId("speech-skipped")).toContainText("4 sentences were not read aloud");
  await expect.poll(async () => (await speechLog(page)).spoken.filter((s) => eventTypes(s).includes("end")).length, { timeout: 20_000 }).toBe(2);
  await page.waitForTimeout(1000);
  const log = await speechLog(page);
  expect(log.spoken.map((s) => s.text)).toEqual(["One.", "Six."]);
  // Not overlapping: the second sentence starts after the first has ended.
  const firstEnd = log.spoken[0]!.events.find((e) => e.type === "end")!.at;
  const secondStart = log.spoken[1]!.events.find((e) => e.type === "start")!.at;
  expect(secondStart).toBeGreaterThanOrEqual(firstEnd);
  await page.getByTestId("camera-stop").click();
});

test("no local voice: a clear message, and nothing is handed to speech synthesis", async ({ page, request }) => {
  // REPLACEMENT of a browser API for this test only (not of any page code): the voice list is the
  // real one with every voice marked as a network voice. The page must refuse to use them.
  await page.addInitScript(() => {
    const synth = window.speechSynthesis;
    const realGetVoices = synth.getVoices.bind(synth);
    synth.getVoices = () =>
      realGetVoices().map((v) => ({ name: v.name, lang: v.lang, voiceURI: v.voiceURI, default: v.default, localService: false }));
  });
  await page.goto("/");
  const message = page.getByTestId("speech-unavailable");
  await expect(message).toBeVisible();
  await expect(message).toContainText("will not be read aloud");
  await expect(message).toContainText("Captions still appear");
  await expect(page.getByTestId("speech-voice")).toBeDisabled();
  expect(await page.evaluate(() => window.speechSynthesis.getVoices().length)).toBeGreaterThan(0); // voices exist, none local

  await startCamera(page);
  await send(request, [caption("final", 0, "Hello, how are you?")]);
  await expect(page.getByTestId("caption-final")).toHaveText(["Hello, how are you?"]);
  await page.waitForTimeout(1000);
  expect((await speechLog(page)).spoken).toHaveLength(0);
  await expect(page.getByTestId("captions-final")).toHaveAttribute("aria-live", "polite");
  await page.getByTestId("camera-stop").click();
});

test("play as live is read aloud, including the last sentence when the file ends", async ({ page, request }) => {
  await page.goto("/");
  await page.getByTestId("mode-file").click();
  await page.getByTestId("file-input").setInputFiles(CLIP_FILE);
  await page.getByTestId("play-as-live").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");
  await send(request, [caption("final", 0, "I am learning to sign, and this sentence is still being spoken when the file ends.")]);
  // The 6 s file runs out and the session ends by itself; that is not a Stop press.
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle", { timeout: 20_000 });
  await expect.poll(async () => eventTypes((await speechLog(page)).spoken[0]!), { timeout: 20_000 }).toEqual(["start", "end"]);
  expect((await speechLog(page)).spoken).toHaveLength(1);
});

test("translate file is not read automatically; each sentence has its own Speak button", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("mode-file").click();
  await page.getByTestId("file-input").setInputFiles(CLIP_FILE);
  await page.getByTestId("translate-file").click();
  await expect(page.getByTestId("translate-segment")).toHaveCount(2);
  await page.waitForTimeout(1000);
  expect((await speechLog(page)).spoken).toHaveLength(0);

  await page.getByTestId("segment-speak").nth(1).click();
  await expect.poll(async () => eventTypes((await speechLog(page)).spoken[0] ?? ({ events: [] } as unknown as SpokenEntry)), { timeout: 20_000 }).toEqual(["start", "end"]);
  const log = await speechLog(page);
  expect(log.spoken).toHaveLength(1);
  // The text is passed as it is; the page does not treat the fake server's "[canned]" prefix specially.
  expect(log.spoken[0]!.text).toBe("[canned] Nice to meet you.");
});
