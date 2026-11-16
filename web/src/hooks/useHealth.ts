"use client";

// Polls GET /health so the status bar can say whether the local service is ready.

import { useEffect, useState } from "react";
import { config } from "@/lib/config";
import { type HealthResponse, parseHealth } from "@/lib/protocol";

export type HealthState =
  | { kind: "checking" }
  | { kind: "unreachable" }
  | { kind: "invalid" }
  | { kind: "ok"; health: HealthResponse };

export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({ kind: "checking" });

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;

    async function poll(): Promise<void> {
      let next: HealthState;
      try {
        const response = await fetch(`${config.serviceUrl}/health`, {
          cache: "no-store",
          signal: AbortSignal.timeout(2500),
        });
        // A not-ready service may answer with an error status; the body still says why.
        const health = parseHealth(await response.json());
        next = health ? { kind: "ok", health } : { kind: "invalid" };
      } catch {
        next = { kind: "unreachable" };
      }
      if (cancelled) return;
      setState(next);
      timer = window.setTimeout(() => void poll(), config.healthPollMs);
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, []);

  return state;
}
