<script lang="ts">
  import { onMount } from "svelte";
  import { theme } from "$lib/stores/theme.svelte";
  import { api } from "$lib/api/client";
  import { push } from "svelte-spa-router";

  type Metrics = {
    running?: boolean;
    paused?: boolean;
    agent_profile?: string;
    premises?: string;
    external_side_effects?: string;
    unsafe_autonomy?: boolean;
    queue_depth?: number;
    actions_last_hour?: number;
    tool_events_total?: number;
    episode_count?: number;
    llm_calls_recent?: number;
    tool_calls_recent?: number;
    latest_model_context_chars?: number;
    schema_version?: number;
    db_path?: string;
    working_directory?: string;
    last_error?: string;
  };

  type ToolEvent = {
    id: number;
    source: string;
    tool: string;
    capabilities: string[];
    args: Record<string, unknown>;
    result_summary: string;
    error: boolean;
    exit_code?: number | null;
    taint_origin?: string;
    created_at: number;
  };

  let metrics = $state<Metrics>({});
  let toolEvents = $state<ToolEvent[]>([]);
  let loading = $state(false);
  let error = $state<string | null>(null);

  async function refresh() {
    loading = true;
    error = null;
    try {
      const [m, events] = await Promise.all([
        api<Metrics>("/ui/api/metrics"),
        api<ToolEvent[]>("/ui/api/tools/events?limit=20"),
      ]);
      metrics = m;
      toolEvents = events;
    } catch (err) {
      error = err instanceof Error ? err.message : "failed to load operations data";
    } finally {
      loading = false;
    }
  }

  async function logout() {
    try {
      await api("/ui/logout", { method: "POST" });
    } catch (_) {}
    push("/login");
  }

  function boolText(value: boolean | undefined): string {
    if (value === undefined) return "--";
    return value ? "yes" : "no";
  }

  function intText(value: number | undefined): string {
    if (value === undefined || !Number.isFinite(value)) return "0";
    return String(Math.trunc(value));
  }

  function fmtTime(seconds: number): string {
    return new Date(seconds * 1000).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function argsText(value: Record<string, unknown>): string {
    try {
      return JSON.stringify(value ?? {});
    } catch (_) {
      return "{}";
    }
  }

  const metricCells: Array<{ label: string; value: string }> = $derived([
    { label: "profile", value: metrics.agent_profile ?? "--" },
    { label: "premises", value: metrics.premises || "--" },
    { label: "side effects", value: metrics.external_side_effects ?? "--" },
    { label: "unsafe", value: boolText(metrics.unsafe_autonomy) },
    { label: "queue", value: intText(metrics.queue_depth) },
    { label: "actions/h", value: intText(metrics.actions_last_hour) },
    { label: "tool events", value: intText(metrics.tool_events_total) },
    { label: "episodes", value: intText(metrics.episode_count) },
    { label: "llm recent", value: intText(metrics.llm_calls_recent) },
    { label: "tools recent", value: intText(metrics.tool_calls_recent) },
    { label: "context chars", value: intText(metrics.latest_model_context_chars) },
    { label: "schema", value: intText(metrics.schema_version) },
  ]);

  onMount(refresh);
</script>

<section class="max-w-6xl mx-auto px-5 md:px-8 py-8 grid gap-8 journal-section">
  <header>
    <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
      iii. instrument settings
    </p>
    <h1 class="mt-2 font-display italic text-3xl tracking-tight" style="color: var(--color-fg)">
      calibration.
    </h1>
  </header>

  <div class="grid md:grid-cols-2 gap-4">
    <div class="border rounded-md p-5" style="background: var(--color-bg-elev)">
      <div class="flex items-center justify-between gap-4">
        <div>
          <p class="font-mono text-xs smallcaps" style="color: var(--color-fg)">theme</p>
          <p class="text-xs mt-0.5" style="color: var(--color-fg-mute)">
            dark mode is the operator's default. the light variant survives daylight.
          </p>
        </div>
        <button
          onclick={() => theme.toggle()}
          class="h-9 px-4 rounded-md border font-mono text-[11px] smallcaps tabular transition-colors"
          style="border-color: var(--color-border-hot); color: var(--color-fg)"
        >
          {theme.value}
        </button>
      </div>
    </div>

    <div class="border rounded-md p-5" style="background: var(--color-bg-elev)">
      <div class="flex items-center justify-between gap-4">
        <div>
          <p class="font-mono text-xs smallcaps" style="color: var(--color-fg)">session</p>
          <p class="text-xs mt-0.5" style="color: var(--color-fg-mute)">
            end the operator session and clear the cookie.
          </p>
        </div>
        <button
          onclick={logout}
          class="h-9 px-4 rounded-md border font-mono text-[11px] smallcaps tabular transition-colors"
          style="border-color: var(--color-danger); color: var(--color-danger)"
        >
          log out
        </button>
      </div>
    </div>
  </div>

  <section class="grid gap-4">
    <div class="flex items-center justify-between gap-4">
      <div>
        <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
          operations
        </p>
        <h2 class="mt-1 font-display italic text-2xl tracking-tight" style="color: var(--color-fg)">
          service posture
        </h2>
      </div>
      <button
        onclick={refresh}
        disabled={loading}
        class="h-9 px-4 rounded-md border font-mono text-[11px] smallcaps tabular transition-opacity disabled:opacity-50"
        style="border-color: var(--color-border-hot); color: var(--color-fg)"
      >
        {loading ? "loading" : "refresh"}
      </button>
    </div>

    <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-2">
      {#each metricCells as cell (cell.label)}
        <div class="border rounded-md px-3 py-2.5" style="background: var(--color-bg-elev)">
          <p class="font-mono text-[9px] smallcaps tabular" style="color: var(--color-fg-faint)">
            {cell.label}
          </p>
          <p class="mt-1 font-mono text-sm tabular truncate" style="color: var(--color-fg)">
            {cell.value}
          </p>
        </div>
      {/each}
    </div>

    <div class="grid md:grid-cols-2 gap-2">
      <div class="border rounded-md px-3 py-2.5 min-w-0" style="background: var(--color-bg-elev)">
        <p class="font-mono text-[9px] smallcaps tabular" style="color: var(--color-fg-faint)">db</p>
        <p class="mt-1 font-mono text-xs truncate" title={metrics.db_path ?? ""} style="color: var(--color-fg)">
          {metrics.db_path ?? "--"}
        </p>
      </div>
      <div class="border rounded-md px-3 py-2.5 min-w-0" style="background: var(--color-bg-elev)">
        <p class="font-mono text-[9px] smallcaps tabular" style="color: var(--color-fg-faint)">workdir</p>
        <p class="mt-1 font-mono text-xs truncate" title={metrics.working_directory ?? ""} style="color: var(--color-fg)">
          {metrics.working_directory ?? "--"}
        </p>
      </div>
    </div>

    {#if metrics.last_error}
      <p class="font-mono text-xs rounded-md border px-3 py-2" style="border-color: var(--color-danger); color: var(--color-danger)">
        {metrics.last_error}
      </p>
    {/if}
  </section>

  <section class="grid gap-4">
    <div>
      <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
        recent tool events · {toolEvents.length}
      </p>
    </div>

    {#if toolEvents.length === 0}
      <p class="font-mono text-xs" style="color: var(--color-fg-mute)">no tool events recorded yet.</p>
    {:else}
      <ul class="grid gap-2">
        {#each toolEvents as event (event.id)}
          <li class="border rounded-md p-3 grid gap-2" style="background: var(--color-bg-elev)">
            <div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
              <div class="flex items-baseline gap-2 min-w-0">
                <span class="font-mono text-xs tabular" style="color: {event.error ? 'var(--color-danger)' : 'var(--color-accent)'}">
                  {event.tool}
                </span>
                <span class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
                  {event.source}
                </span>
              </div>
              <span class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
                {fmtTime(event.created_at)}
              </span>
            </div>
            <div class="flex flex-wrap gap-1">
              {#each event.capabilities ?? [] as capability (capability)}
                <span class="px-2 py-0.5 rounded-sm font-mono text-[9px] smallcaps tabular border"
                      style="border-color: var(--color-border-hot); color: var(--color-fg-mute)">
                  {capability}
                </span>
              {/each}
              {#if event.taint_origin}
                <span class="px-2 py-0.5 rounded-sm font-mono text-[9px] smallcaps tabular border"
                      style="border-color: var(--color-warn); color: var(--color-warn)">
                  {event.taint_origin}
                </span>
              {/if}
            </div>
            <p class="text-sm leading-relaxed" style="color: var(--color-fg)">
              {event.result_summary || "(empty result)"}
            </p>
            <pre class="font-mono text-[10px] leading-relaxed overflow-x-auto rounded-sm px-2 py-1.5"
                 style="background: var(--color-bg-sunk); color: var(--color-fg-mute)">{argsText(event.args)}</pre>
          </li>
        {/each}
      </ul>
    {/if}
  </section>

  {#if error}
    <p class="font-mono text-xs" style="color: var(--color-danger)">{error}</p>
  {/if}
</section>
