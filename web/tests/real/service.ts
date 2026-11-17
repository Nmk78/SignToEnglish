// Starts and stops the real translation service for the browser tests. Only a service started
// here (recorded in a pid file) is ever stopped here.
import { spawn } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

export const SERVICE_URL = "http://127.0.0.1:8000";
const WEB_ROOT = resolve(__dirname, "..", "..");
const REPO_ROOT = resolve(WEB_ROOT, "..");
export const RESULTS_DIR = resolve(WEB_ROOT, "test-results-real");
const PID_FILE = resolve(RESULTS_DIR, "service.pid");
const LOG_FILE = resolve(RESULTS_DIR, "service.log");

export interface Health {
  status: "ready" | "not_ready";
  detail: string | null;
  checkpoint: string | null;
  checkpoint_id: string | null;
  preprocessing_hash: string | null;
  device: string | null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** The service's health report, or null when nothing answers. */
export async function health(): Promise<Health | null> {
  try {
    const response = await fetch(`${SERVICE_URL}/health`, { signal: AbortSignal.timeout(2000) });
    return (await response.json()) as Health;
  } catch {
    return null;
  }
}

export function startedByTests(): boolean {
  return existsSync(PID_FILE);
}

/** Starts the service and waits until the model is loaded. Returns the seconds it took. */
export async function startService(timeoutMs = 240_000): Promise<number> {
  mkdirSync(RESULTS_DIR, { recursive: true });
  const log = openSync(LOG_FILE, "a");
  const child = spawn("uv", ["run", "--frozen", "s2e", "serve"], {
    cwd: REPO_ROOT,
    detached: true, // its own process group, so the whole group can be stopped
    stdio: ["ignore", log, log],
    env: { ...process.env, HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" },
  });
  closeSync(log);
  child.unref();
  if (child.pid === undefined) throw new Error("could not start `uv run --frozen s2e serve`");
  writeFileSync(PID_FILE, String(child.pid));
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const h = await health();
    if (h?.status === "ready") return (Date.now() - started) / 1000;
    await sleep(500);
  }
  await stopService();
  throw new Error(`the service was not ready after ${timeoutMs / 1000} s; see ${LOG_FILE}`);
}

/** Stops the service these tests started. `hard` kills it without warning, like a crash. */
export async function stopService(hard = false): Promise<void> {
  if (!existsSync(PID_FILE)) return;
  const pid = Number(readFileSync(PID_FILE, "utf8"));
  const signal = (s: NodeJS.Signals) => {
    try {
      process.kill(-pid, s);
    } catch {
      // already gone
    }
  };
  signal(hard ? "SIGKILL" : "SIGTERM");
  const started = Date.now();
  while ((await health()) !== null) {
    if (Date.now() - started > 15_000) signal("SIGKILL");
    if (Date.now() - started > 30_000) throw new Error("the service did not stop");
    await sleep(250);
  }
  rmSync(PID_FILE, { force: true });
}

/** Makes sure the service is up and ready, starting it if needed. */
export async function ensureService(): Promise<void> {
  const h = await health();
  if (h?.status === "ready") return;
  if (h === null) {
    await stopService();
    await startService();
    return;
  }
  const started = Date.now();
  while ((await health())?.status !== "ready") {
    if (Date.now() - started > 240_000) throw new Error("the service did not become ready");
    await sleep(500);
  }
}
