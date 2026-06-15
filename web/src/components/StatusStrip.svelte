<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { link } from "svelte-spa-router";
  import active from "svelte-spa-router/active";
  import { api } from "$lib/api/client";

  const navTabs = [
    { href: "/stream", label: "stream" },
    { href: "/chat", label: "chat" },
    { href: "/projects", label: "projects" },
    { href: "/goals", label: "goals" },
    { href: "/influences", label: "influences" },
    { href: "/memory", label: "memory" },
    { href: "/episodes", label: "episodes" },
    { href: "/trace", label: "trace" },
    { href: "/settings", label: "settings" },
  ];

  type Status = {
    running?: boolean;
    paused?: boolean;
    autonomous?: boolean;
    unsafe_autonomy?: boolean;
    session_id?: string;
    queue_depth?: number;
    actions_last_hour?: number;
    last_autonomous_action?: string;
    uptime?: number;
  };

  let status = $state<Status>({});
  let lastSyncAt = $state(0);
  let now = $state(Date.now());
  let controlBusy = $state(false);
  let timer: ReturnType<typeof setInterval> | null = null;
  let clock: ReturnType<typeof setInterval> | null = null;

  async function refresh() {
    try {
      const snap = await api<{ status: Status }>("/ui/api/snapshot");
      status = snap.status ?? {};
      lastSyncAt = Date.now();
    } catch (_) {
      /* tolerated until SSE lands */
    }
  }

  onMount(() => {
    refresh();
    timer = setInterval(refresh, 5000);
    clock = setInterval(() => (now = Date.now()), 1000);
  });

  onDestroy(() => {
    if (timer) clearInterval(timer);
    if (clock) clearInterval(clock);
  });

  function fmtUptime(seconds: number | undefined): string {
    if (!seconds || !Number.isFinite(seconds)) return "--:--:--";
    const s = Math.floor(seconds);
    const h = Math.floor(s / 3600).toString().padStart(2, "0");
    const m = Math.floor((s % 3600) / 60).toString().padStart(2, "0");
    const sec = (s % 60).toString().padStart(2, "0");
    return `${h}:${m}:${sec}`;
  }

  async function setPaused(next: boolean) {
    controlBusy = true;
    try {
      await api(`/ui/api/control/${next ? "pause" : "resume"}`, { method: "POST" });
      status = { ...status, paused: next };
      await refresh();
    } finally {
      controlBusy = false;
    }
  }

  let synced = $derived(Math.max(0, Math.floor((now - lastSyncAt) / 1000)));
  let paused = $derived(!!status.paused);

  const indicators: Array<{ label: string; on: boolean | undefined; pulse: boolean; color: string }> = $derived([
    { label: "run", on: status.running, pulse: !!status.running && !status.paused, color: "var(--color-ok)" },
    { label: "pause", on: status.paused, pulse: false, color: "var(--color-warn)" },
    { label: "auto", on: status.autonomous, pulse: false, color: "var(--color-ch-intention)" },
    { label: "unsafe", on: status.unsafe_autonomy, pulse: false, color: "var(--color-danger)" },
  ]);

  const readouts: Array<{ label: string; value: string }> = $derived([
    { label: "uptime", value: fmtUptime(status.uptime) },
    { label: "queue", value: String(status.queue_depth ?? 0) },
    { label: "acts/h", value: String(status.actions_last_hour ?? 0) },
    { label: "session", value: status.session_id?.slice(0, 8) ?? "--" },
  ]);
</script>

<div class="pause-overlay {paused ? 'on' : ''}"></div>
{#if paused}
  <div class="fixed inset-x-0 top-16 z-50 flex justify-center pointer-events-none">
    <button
      type="button"
      onclick={() => setPaused(false)}
      disabled={controlBusy}
      class="pointer-events-auto h-10 px-5 rounded-md border font-mono text-[11px] smallcaps tabular shadow-lg disabled:opacity-50"
      style="background: var(--color-bg-elev); border-color: var(--color-warn); color: var(--color-warn)"
    >
      {controlBusy ? "resuming" : "resume autonomy"}
    </button>
  </div>
{/if}

<header
  class="sticky top-0 z-20 h-14 flex items-center gap-6 px-5 border-b backdrop-blur-sm boot-reveal"
  style="background: color-mix(in oklab, var(--color-bg-elev) 92%, transparent);"
>
  <a href="#/" class="flex items-baseline gap-2 no-underline shrink-0">
    <span class="font-display italic text-xl tracking-tight" style="color: var(--color-fg)">conscio</span>
    <span class="hidden xs:inline font-mono text-[10px] tabular smallcaps" style="color: var(--color-fg-faint)">observatory</span>
  </a>

  <!-- desktop nav (mobile uses BottomTabBar). Horizontal-scrolls on narrower
       laptops so all 9 tabs stay reachable. -->
  <nav class="hidden md:flex items-center gap-0.5 ml-2 overflow-x-auto no-scrollbar min-w-0">
    {#each navTabs as tab (tab.href)}
      <a
        href={tab.href}
        use:link
        use:active={{ path: tab.href, className: "nav-active" }}
        class="nav-link px-3 py-2 rounded-md font-mono text-[12px] smallcaps tracking-wider no-underline transition-colors whitespace-nowrap"
      >
        {tab.label}
      </a>
    {/each}
  </nav>

  <div class="flex items-center gap-3">
    {#each indicators as ind (ind.label)}
      <span class="flex items-center gap-1.5 font-mono text-[10px] smallcaps"
            style="color: {ind.on ? 'var(--color-fg)' : 'var(--color-fg-faint)'}">
        <span
          class="inline-block w-1.5 h-1.5 rounded-full transition-opacity {ind.pulse ? 'breathe' : ''}"
          style="background: {ind.on ? ind.color : 'var(--color-border-hot)'}; opacity: {ind.on ? 1 : 0.45}"
        ></span>
        {ind.label}
      </span>
    {/each}
  </div>

  <button
    type="button"
    onclick={() => setPaused(!paused)}
    disabled={controlBusy}
    class="h-8 px-3 rounded-md border font-mono text-[10px] smallcaps tabular transition-opacity disabled:opacity-50"
    style="border-color: {paused ? 'var(--color-warn)' : 'var(--color-border-hot)'}; color: {paused ? 'var(--color-warn)' : 'var(--color-fg-mute)'}"
  >
    {controlBusy ? "..." : paused ? "resume" : "pause"}
  </button>

  <div class="hidden sm:flex items-baseline gap-5 ml-auto font-mono text-[11px] tabular"
       style="color: var(--color-fg-mute)">
    {#each readouts as r (r.label)}
      <span class="flex items-baseline gap-1.5">
        <span class="smallcaps text-[9px]" style="color: var(--color-fg-faint)">{r.label}</span>
        <span style="color: var(--color-fg)">{r.value}</span>
      </span>
    {/each}
  </div>

  <div class="ml-auto sm:ml-0 flex items-center gap-3 font-mono text-[10px] smallcaps"
       style="color: var(--color-fg-faint)">
    <kbd class="hidden md:inline-block px-1.5 py-0.5 rounded-sm border tabular"
         style="border-color: var(--color-border); color: var(--color-fg-mute)">⌘k</kbd>
    <span class="hidden sm:inline">synced {synced}s</span>
    <span class="inline-block w-1.5 h-1.5 rounded-full breathe"
          style="background: var(--color-accent)"></span>
  </div>
</header>

<style>
  .nav-link {
    color: var(--color-fg);
    opacity: 0.55;
    border: 1px solid transparent;
  }
  .nav-link:hover {
    opacity: 0.9;
    background: color-mix(in oklab, var(--color-fg-faint) 12%, transparent);
  }
  :global(.nav-active) {
    opacity: 1 !important;
    border-color: var(--color-border-hot);
    background: color-mix(in oklab, var(--color-accent) 14%, transparent);
    box-shadow: inset 0 -2px 0 var(--color-accent);
  }
</style>
