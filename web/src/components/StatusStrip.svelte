<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { link } from "svelte-spa-router";
  import active from "svelte-spa-router/active";
  import { api } from "$lib/api/client";

  const navTabs = [
    { href: "/stream", label: "stream" },
    { href: "/chat", label: "chat" },
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

  let synced = $derived(Math.max(0, Math.floor((now - lastSyncAt) / 1000)));

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

<header
  class="sticky top-0 z-20 h-14 flex items-center gap-6 px-5 border-b backdrop-blur-sm"
  style="background: color-mix(in oklab, var(--color-bg-elev) 92%, transparent);"
>
  <a href="#/" class="flex items-baseline gap-2 no-underline shrink-0">
    <span class="font-display italic text-xl tracking-tight" style="color: var(--color-fg)">conscio</span>
    <span class="hidden xs:inline font-mono text-[10px] tabular smallcaps" style="color: var(--color-fg-faint)">observatory</span>
  </a>

  <!-- desktop nav (mobile uses BottomTabBar) -->
  <nav class="hidden md:flex items-center gap-0.5 -mx-1">
    {#each navTabs as tab (tab.href)}
      <a
        href={tab.href}
        use:link
        use:active={{ path: tab.href, className: "nav-active" }}
        class="nav-link px-3 py-1.5 rounded-sm font-mono text-[11px] smallcaps tracking-wider no-underline transition-colors"
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

  <div class="hidden sm:flex items-baseline gap-5 ml-auto font-mono text-[11px] tabular"
       style="color: var(--color-fg-mute)">
    {#each readouts as r (r.label)}
      <span class="flex items-baseline gap-1.5">
        <span class="smallcaps text-[9px]" style="color: var(--color-fg-faint)">{r.label}</span>
        <span style="color: var(--color-fg)">{r.value}</span>
      </span>
    {/each}
  </div>

  <div class="ml-auto sm:ml-0 flex items-center gap-2 font-mono text-[10px] smallcaps"
       style="color: var(--color-fg-faint)">
    <span class="hidden sm:inline">synced {synced}s</span>
    <span class="inline-block w-1.5 h-1.5 rounded-full breathe"
          style="background: var(--color-accent)"></span>
  </div>
</header>

<style>
  .nav-link {
    color: var(--color-fg-mute);
    border: 1px solid transparent;
  }
  .nav-link:hover {
    color: var(--color-fg);
    background: color-mix(in oklab, var(--color-fg-faint) 8%, transparent);
  }
  :global(.nav-active) {
    color: var(--color-fg) !important;
    border-color: var(--color-border-hot);
    background: color-mix(in oklab, var(--color-accent) 8%, transparent);
  }
</style>
