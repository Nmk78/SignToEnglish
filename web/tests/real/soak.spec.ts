// PLAN.md Step 4.1 metric target: frames sent at the configured rate within 10% for 5 minutes
// without memory growth in the tab. Run with `npm run test:real:soak`.
//
// How memory is measured, every 15 seconds:
//   - JS heap in use after a forced garbage collection (DevTools protocol:
//     HeapProfiler.collectGarbage, then Performance.getMetrics JSHeapUsedSize), plus the number of
//     DOM nodes and event listeners from the same call;
//   - resident memory of the browser's renderer processes, from `ps`.
// Growth is the last minute's average minus the second minute's average (the first minute is
// warm-up). The numbers are printed and written to test-results-real/soak.json; the only
// assertions are that the stream ran cleanly. Read-aloud is on, as it will be at the show, so
// this Mac talks for five minutes.
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { instrumentSpeech, speechLog } from "../e2e/speech-instrument";
import { median, ofType, recordServerMessages } from "./helpers";
import { RESULTS_DIR, ensureService } from "./service";

const RUN_MS = 5 * 60 * 1000;
const SAMPLE_MS = 15_000;
const SEND_FPS = 10;

function rendererRssMb(): number {
  const out = execFileSync("ps", ["-axo", "rss=,command="], { encoding: "utf8" });
  let kb = 0;
  for (const line of out.split("\n")) {
    if (line.includes("ms-playwright") && line.includes("--type=renderer")) kb += Number(line.trim().split(/\s+/)[0]);
  }
  return kb / 1024;
}

const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1);

test.beforeEach(async () => ensureService());

test("real service: five minutes of camera streaming at the configured rate, with memory sampled", async ({ page }) => {
  await instrumentSpeech(page);
  const messages = recordServerMessages(page);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Performance.enable");
  await cdp.send("HeapProfiler.enable");

  await page.goto("/");
  await expect(page.getByTestId("service-status")).toContainText("Service ready");
  await page.getByTestId("camera-start").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");
  const started = Date.now();

  const samples: { t_s: number; heap_mb: number; nodes: number; listeners: number; renderer_rss_mb: number; frames_sent: number; captions: number }[] = [];
  while (Date.now() - started < RUN_MS) {
    await page.waitForTimeout(SAMPLE_MS);
    await cdp.send("HeapProfiler.collectGarbage");
    const { metrics } = await cdp.send("Performance.getMetrics");
    const metric = (name: string) => metrics.find((m) => m.name === name)?.value ?? NaN;
    samples.push({
      t_s: Math.round((Date.now() - started) / 1000),
      heap_mb: metric("JSHeapUsedSize") / 1e6,
      nodes: metric("Nodes"),
      listeners: metric("JSEventListeners"),
      renderer_rss_mb: rendererRssMb(),
      frames_sent: Number.parseInt((await page.getByTestId("frames-sent").textContent()) ?? "0", 10),
      captions: await page.getByTestId("caption-final").count(),
    });
    await expect(page.getByTestId("stream-phase")).toHaveText("Streaming");
  }
  await page.getByTestId("camera-stop").click();
  await expect(page.getByTestId("stream-phase")).toHaveText("Idle");
  await expect(page.getByTestId("stream-error")).toHaveCount(0);

  const summary = ofType(messages, "summary")[0]!;
  const frames = summary.frames as { received: number; dropped_behind: number; processed: number; rejected_bad: number; rejected_non_increasing: number };
  const streamS = Number(summary.stream_ms) / 1000;
  const overallFps = (frames.received - 1) / streamS;
  // Send rate in each sampling interval, from the page's own count of frames sent.
  const intervalFps = samples.slice(1).map((s, i) => (s.frames_sent - samples[i]!.frames_sent) / (s.t_s - samples[i]!.t_s));
  const second = samples.filter((s) => s.t_s > 60 && s.t_s <= 120);
  const last = samples.filter((s) => s.t_s > 240);
  const finals = ofType(messages, "final");
  const statuses = ofType(messages, "status");
  const gaps = [...ofType(messages, "partial"), ...finals].sort((a, b) => a.at - b.at).map((m, i, all) => (i ? m.at - all[i - 1]!.at : 0)).slice(1);
  const speech = await speechLog(page);
  const result = {
    configured_fps: SEND_FPS,
    stream_s: streamS,
    frames_received_by_service: frames.received,
    overall_fps: Number(overallFps.toFixed(3)),
    interval_fps_min: Number(Math.min(...intervalFps).toFixed(2)),
    interval_fps_max: Number(Math.max(...intervalFps).toFixed(2)),
    within_10_percent: overallFps >= SEND_FPS * 0.9 && overallFps <= SEND_FPS * 1.1 && Math.min(...intervalFps) >= SEND_FPS * 0.9 && Math.max(...intervalFps) <= SEND_FPS * 1.1,
    frames_dropped_by_service: frames.dropped_behind,
    frames_rejected: frames.rejected_bad + frames.rejected_non_increasing,
    rtf_median: median(statuses.map((s) => Number(s.rtf)).filter((v) => Number.isFinite(v))),
    rtf_max: Math.max(...statuses.map((s) => Number(s.rtf)).filter((v) => Number.isFinite(v))),
    finals: finals.length,
    commit_delay_ms_median: median(finals.map((f) => Number(f.latency_ms))),
    commit_delay_ms_max: Math.max(...finals.map((f) => Number(f.latency_ms))),
    text_update_gap_ms_median: median(gaps),
    text_update_gap_ms_max: Math.max(...gaps),
    stretches_not_translated: ofType(messages, "dropped").length,
    errors: ofType(messages, "error").length,
    heap_mb_minute_2: Number(mean(second.map((s) => s.heap_mb)).toFixed(2)),
    heap_mb_last_minute: Number(mean(last.map((s) => s.heap_mb)).toFixed(2)),
    renderer_rss_mb_minute_2: Number(mean(second.map((s) => s.renderer_rss_mb)).toFixed(1)),
    renderer_rss_mb_last_minute: Number(mean(last.map((s) => s.renderer_rss_mb)).toFixed(1)),
    dom_nodes_minute_2: Math.round(mean(second.map((s) => s.nodes))),
    dom_nodes_last_minute: Math.round(mean(last.map((s) => s.nodes))),
    sentences_spoken: speech.spoken.length,
    sentences_spoken_to_the_end: speech.spoken.filter((s) => s.events.some((e) => e.type === "end")).length,
    speech_skip_notice: (await page.getByTestId("speech-skipped").count()) > 0 ? await page.getByTestId("speech-skipped").textContent() : null,
    samples,
  };
  mkdirSync(RESULTS_DIR, { recursive: true });
  writeFileSync(resolve(RESULTS_DIR, "soak.json"), `${JSON.stringify(result, null, 2)}\n`);
  const { samples: _samples, ...headline } = result;
  void _samples;
  console.log(`[measured] soak: ${JSON.stringify(headline, null, 1)}`);
  console.log(`[measured] soak samples (t_s, heap_mb, renderer_rss_mb, nodes, frames_sent, captions): ${samples.map((s) => `${s.t_s}:${s.heap_mb.toFixed(1)}:${s.renderer_rss_mb.toFixed(0)}:${s.nodes}:${s.frames_sent}:${s.captions}`).join(" ")}`);

  expect(summary.complete).toBe(true);
  expect(frames.rejected_bad + frames.rejected_non_increasing).toBe(0);
  expect(streamS).toBeGreaterThan(295);
});
