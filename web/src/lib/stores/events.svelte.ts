/**
 * Singleton EventStream wired to /ui/api/events. Components subscribe to
 * specific event names; the activity-stream view reads `entries` directly.
 *
 * We keep a bounded ring buffer (RING_SIZE) so memory stays predictable
 * regardless of how many events arrive while the page is open.
 */
import { EventStream, type StreamHealth } from "$lib/api/stream";

const RING_SIZE = 500;

export type ActivityEntry = {
  id: string;
  type: string;          // e.g. "workspace.action" or "chat.message"
  kind: string;          // workspace entry kind ("action") or "chat"
  ts: number;
  source?: string;
  content: string;
  priority?: number;
  salience?: number;
  metadata?: Record<string, unknown>;
};

let _stream: EventStream | null = null;

let entries = $state<ActivityEntry[]>([]);
let health = $state<StreamHealth>("connecting");
let totalReceived = $state(0);

// The server replays its backlog on every (re)connect; remember what we've
// already shown so reconnects don't duplicate rows. `seq` keeps entry ids
// unique even when two events share a timestamp.
let seq = 0;
const seen = new Set<string>();

function summarise(payload: any): string {
  if (typeof payload?.content === "string") return payload.content;
  if (typeof payload?.output === "string") return payload.output;
  if (typeof payload?.input === "string") return payload.input;
  if (typeof payload?.user === "string") return payload.user;
  if (typeof payload?.agent === "string") return payload.agent;
  if (payload?.goal_id) return `goal ${payload.goal_id} ${payload.action ?? "changed"}`;
  if (payload?.project_id) return `project ${payload.project_id} → ${payload.status ?? "updated"}`;
  if (payload?.paused !== undefined) return payload.paused ? "paused" : "resumed";
  return "—";
}

function push(key: string, entry: ActivityEntry) {
  if (seen.has(key)) return;
  seen.add(key);
  if (seen.size > RING_SIZE * 2) {
    const oldest = seen.values().next().value;
    if (oldest !== undefined) seen.delete(oldest);
  }
  entries = [entry, ...entries].slice(0, RING_SIZE);
  totalReceived += 1;
}

export function startEventStream(): void {
  if (_stream) return;
  _stream = new EventStream("/ui/api/events");

  _stream.onHealth((h) => (health = h));

  // Workspace entries: 10 channels.
  const channels = [
    "observation", "intention", "plan", "action", "result",
    "reflection", "memory", "system", "conflict", "self_state",
  ];
  for (const kind of channels) {
    _stream.on(`workspace.${kind}`, (e: any) => {
      push(`${e.type}:${e.ts}`, {
        id: `ws-${e.ts}-${seq++}`,
        type: e.type,
        kind,
        ts: e.ts,
        source: e.source,
        content: e.content ?? summarise(e),
        priority: e.priority,
        salience: e.salience,
        metadata: e.metadata,
      });
    });
  }

  // Service-level events.
  for (const t of ["chat.message", "episode.created", "project.updated", "goal.changed", "control.paused"]) {
    _stream.on(t, (e: any) => {
      push(`${t}:${e.ts}`, {
        id: `${t}-${e.ts}-${seq++}`,
        type: t,
        kind: t.split(".")[0],
        ts: e.ts,
        source: t,
        content: summarise(e),
        metadata: e,
      });
    });
  }

  _stream.start();
}

export function stopEventStream(): void {
  _stream?.stop();
  _stream = null;
}

export const events = {
  get entries() { return entries; },
  get health() { return health; },
  get totalReceived() { return totalReceived; },
  clear() { entries = []; },
};
