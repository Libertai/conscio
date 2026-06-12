/**
 * Thin wrapper around EventSource that:
 *   - reconnects automatically on transient failures (browser default + our backoff cap)
 *   - tracks connection health (live / stalled / disconnected) for the UI
 *   - exposes a typed subscribe API per event name
 */

export type StreamHealth = "connecting" | "live" | "stalled" | "disconnected";

export type StreamEvent = {
  type: string;
  ts: number;
  [key: string]: unknown;
};

type Listener = (e: StreamEvent) => void;

export class EventStream {
  private es: EventSource | null = null;
  private listeners = new Map<string, Set<Listener>>();
  private wildcard = new Set<Listener>();
  private healthListeners = new Set<(h: StreamHealth) => void>();
  private health: StreamHealth = "connecting";
  private stallTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly stallMs = 30_000;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelayMs = 1_000;
  private readonly reconnectMaxMs = 30_000;
  private stopped = false;

  constructor(private readonly url: string) {}

  start(): void {
    this.stopped = false;
    if (this.es) return;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.stallTimer) clearTimeout(this.stallTimer);
    this.stallTimer = null;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    if (this.es) {
      this.es.close();
      this.es = null;
    }
    this.setHealth("disconnected");
  }

  on(eventName: string, listener: Listener): () => void {
    if (eventName === "*") {
      this.wildcard.add(listener);
      return () => this.wildcard.delete(listener);
    }
    let bucket = this.listeners.get(eventName);
    if (!bucket) {
      bucket = new Set();
      this.listeners.set(eventName, bucket);
      // One DOM handler per event name: dispatch() fans out to the bucket,
      // so attaching per-listener would double-fire everything.
      this.es?.addEventListener(eventName, this.handlerFor(eventName));
    }
    bucket.add(listener);
    return () => bucket!.delete(listener);
  }

  onHealth(listener: (h: StreamHealth) => void): () => void {
    this.healthListeners.add(listener);
    listener(this.health);
    return () => this.healthListeners.delete(listener);
  }

  private connect(): void {
    this.setHealth("connecting");
    const es = new EventSource(this.url, { withCredentials: true });
    this.es = es;

    es.addEventListener("stream.open", () => {
      this.reconnectDelayMs = 1_000;
      this.setHealth("live");
      this.bump();
    });

    // Server heartbeat: keeps the stall detector honest while the agent is quiet.
    es.addEventListener("ping", () => this.bump());

    es.addEventListener("error", () => {
      if (es.readyState === EventSource.CLOSED) {
        // The browser gives up permanently on HTTP errors (502 during a
        // deploy, 401 after session expiry) — recreate the source ourselves.
        this.setHealth("disconnected");
        es.close();
        if (this.es === es) this.es = null;
        this.scheduleReconnect();
      } else {
        this.setHealth("connecting");
      }
    });

    // (Re-)attach listeners for every previously-registered event name.
    for (const name of this.listeners.keys()) {
      es.addEventListener(name, this.handlerFor(name));
    }
    // Generic message handler for unnamed messages + wildcard.
    es.onmessage = (e: MessageEvent) => this.dispatch("message", e.data);
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.stopped && !this.es) this.connect();
    }, this.reconnectDelayMs);
    this.reconnectDelayMs = Math.min(this.reconnectDelayMs * 2, this.reconnectMaxMs);
  }

  private handlerFor(name: string) {
    return (e: MessageEvent) => this.dispatch(name, e.data);
  }

  private dispatch(name: string, raw: string) {
    this.bump();
    let payload: StreamEvent;
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = { type: name, ts: Date.now() / 1000, raw };
    }
    const bucket = this.listeners.get(name);
    if (bucket) for (const fn of bucket) fn(payload);
    for (const fn of this.wildcard) fn(payload);
  }

  private setHealth(next: StreamHealth) {
    if (this.health === next) return;
    this.health = next;
    for (const fn of this.healthListeners) fn(next);
  }

  private bump() {
    if (this.stallTimer) clearTimeout(this.stallTimer);
    this.stallTimer = setTimeout(() => this.setHealth("stalled"), this.stallMs);
    if (this.health === "stalled" || this.health === "connecting") {
      this.setHealth("live");
    }
  }
}
