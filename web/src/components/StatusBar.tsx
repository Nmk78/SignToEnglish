"use client";

import type { HealthState } from "@/hooks/useHealth";
import type { StreamApi } from "@/hooks/useStream";
import { config } from "@/lib/config";

const PHASE_TEXT: Record<StreamApi["phase"], string> = {
  idle: "Idle",
  connecting: "Connecting…",
  streaming: "Streaming",
  stopping: "Finishing…",
};

function serviceText(health: HealthState): { label: string; tone: "good" | "bad" | "wait"; detail: string | null } {
  switch (health.kind) {
    case "checking":
      return { label: "Checking service…", tone: "wait", detail: null };
    case "unreachable":
      return { label: "Service unreachable", tone: "bad", detail: `Nothing answered at ${config.serviceUrl}.` };
    case "invalid":
      return { label: "Service answered unexpectedly", tone: "bad", detail: `${config.serviceUrl}/health did not return a health report.` };
    case "ok":
      return health.health.status === "ready"
        ? { label: "Service ready", tone: "good", detail: null }
        : { label: "Service not ready", tone: "bad", detail: health.health.detail };
  }
}

function fmt(value: number | null, digits: number): string {
  return value === null ? "–" : value.toFixed(digits);
}

export function StatusBar({ health, stream }: { health: HealthState; stream: StreamApi }) {
  const service = serviceText(health);
  const info = health.kind === "ok" ? health.health : null;
  const status = stream.serverStatus;
  const stats = stream.clientStats;
  const skipped = stats.skippedBusy + stats.skippedBackpressure;

  return (
    <section className="statusbar" aria-label="Status">
      <span className={`chip ${service.tone}`} data-testid="service-status">
        <span className="dot" aria-hidden="true" />
        {service.label}
        {service.detail ? `: ${service.detail}` : ""}
      </span>
      <span className={`chip ${stream.phase === "streaming" ? "good" : "plain"}`} data-testid="stream-phase">
        {PHASE_TEXT[stream.phase]}
      </span>
      {stream.phase !== "idle" && status && (
        <span className={`chip ${status.signing ? "good" : "plain"}`} data-testid="signing-indicator">
          {status.signing ? "Signing detected" : "Hands at rest"}
        </span>
      )}
      <dl className="stats">
        <div>
          <dt>Sent</dt>
          <dd data-testid="frames-sent">{stats.framesSent} frames</dd>
        </div>
        <div>
          <dt>Send rate</dt>
          <dd>
            {stats.sendFps.toFixed(1)} / {config.sendFps} fps
          </dd>
        </div>
        <div>
          <dt>Skipped</dt>
          <dd>{skipped}</dd>
        </div>
        <div>
          <dt>Service in</dt>
          <dd>{fmt(status?.fps_in ?? null, 1)} fps</dd>
        </div>
        <div>
          <dt>Queue</dt>
          <dd>{fmt(status?.queue_depth ?? null, 0)}</dd>
        </div>
        <div>
          <dt>Real-time factor</dt>
          <dd>{fmt(status?.rtf ?? null, 2)}</dd>
        </div>
        {info?.checkpoint_id && (
          <div>
            <dt>Checkpoint</dt>
            <dd data-testid="checkpoint-id">{info.checkpoint_id}</dd>
          </div>
        )}
        {info?.device && (
          <div>
            <dt>Device</dt>
            <dd>{info.device}</dd>
          </div>
        )}
        {info?.preprocessing_hash && (
          <div>
            <dt>Preprocessing</dt>
            <dd>{info.preprocessing_hash}</dd>
          </div>
        )}
      </dl>
    </section>
  );
}
