<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { events, startEventStream, stopEventStream, type ActivityEntry } from "$lib/stores/events.svelte";

  const CHANNELS: Array<{ kind: string; label: string; abbr: string }> = [
    { kind: "observation", label: "observation", abbr: "OB" },
    { kind: "intention",   label: "intention",   abbr: "IN" },
    { kind: "plan",        label: "plan",        abbr: "PL" },
    { kind: "action",      label: "action",      abbr: "AC" },
    { kind: "result",      label: "result",      abbr: "RE" },
    { kind: "reflection",  label: "reflection",  abbr: "RF" },
    { kind: "memory",      label: "memory",      abbr: "MM" },
    { kind: "system",      label: "system",      abbr: "SY" },
    { kind: "conflict",    label: "conflict",    abbr: "CF" },
    { kind: "self_state",  label: "self state",  abbr: "SS" },
  ];

  // Service-level event abbreviations.
  const SERVICE_ABBR: Record<string, { abbr: string; color: string }> = {
    chat: { abbr: "CH", color: "var(--color-fg)" },
    episode: { abbr: "EP", color: "var(--color-ch-result)" },
    project: { abbr: "PR", color: "var(--color-ch-plan)" },
    control: { abbr: "CT", color: "var(--color-ch-system)" },
  };

  function abbrFor(entry: ActivityEntry): { abbr: string; color: string } {
    const ch = CHANNELS.find((c) => c.kind === entry.kind);
    if (ch) return { abbr: ch.abbr, color: `var(--color-ch-${ch.kind})` };
    return SERVICE_ABBR[entry.kind] ?? { abbr: "··", color: "var(--color-fg-mute)" };
  }

  function fmtTime(ts: number): string {
    const d = new Date(ts * 1000);
    const pad = (n: number, w = 2) => n.toString().padStart(w, "0");
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3).slice(0, 3)}`;
  }

  let pulseFor = $state<Record<string, number>>({});
  let lastPulseId = 0;

  onMount(() => {
    startEventStream();
  });

  onDestroy(() => {
    stopEventStream();
  });

  // Pulse the rail of a channel when a new entry of that kind arrives.
  $effect(() => {
    const e = events.entries[0];
    if (!e) return;
    const id = ++lastPulseId;
    pulseFor = { ...pulseFor, [e.kind]: id };
    const k = e.kind;
    setTimeout(() => {
      if (pulseFor[k] === id) pulseFor = { ...pulseFor, [k]: 0 };
    }, 600);
  });

  let healthLabel = $derived(
    events.health === "live" ? "live" :
    events.health === "stalled" ? "stalled" :
    events.health === "connecting" ? "connecting…" : "offline"
  );

  let healthColor = $derived(
    events.health === "live" ? "var(--color-accent)" :
    events.health === "stalled" ? "var(--color-warn)" :
    events.health === "connecting" ? "var(--color-ch-intention)" : "var(--color-danger)"
  );
</script>

<div class="relative grid md:grid-cols-[1fr_auto] min-h-[calc(100vh-3.5rem-4rem)] md:min-h-[calc(100vh-3.5rem)]">
  <!-- background dot grid + tape lines -->
  <div class="absolute inset-0 dot-grid pointer-events-none opacity-40"></div>

  <!-- main column: event log -->
  <section class="relative flex flex-col min-w-0">
    <header class="sticky top-14 z-10 flex items-baseline justify-between gap-4 px-5 md:px-8 py-4 border-b backdrop-blur"
            style="background: color-mix(in oklab, var(--color-bg) 90%, transparent)">
      <div class="flex items-baseline gap-3">
        <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">i.</p>
        <h1 class="font-display italic text-2xl md:text-3xl tracking-tight" style="color: var(--color-fg)">
          activity stream
        </h1>
      </div>
      <div class="flex items-center gap-3 font-mono text-[10px] smallcaps tabular"
           style="color: var(--color-fg-mute)">
        <span class="hidden sm:inline">{events.totalReceived} events</span>
        <span class="flex items-center gap-1.5">
          <span class="inline-block w-1.5 h-1.5 rounded-full breathe" style="background: {healthColor}"></span>
          {healthLabel}
        </span>
      </div>
    </header>

    <div class="flex-1 overflow-y-auto tape-lines">
      {#if events.entries.length === 0}
        <div class="grid place-items-center min-h-[40vh] px-6">
          <p class="font-mono text-xs smallcaps tabular text-center" style="color: var(--color-fg-faint)">
            waiting for the first signal.<br/>
            cognition will surface here as it broadcasts.
          </p>
        </div>
      {:else}
        <ul class="px-5 md:px-8 py-4 space-y-2">
          {#each events.entries as entry (entry.id)}
            {@const meta = abbrFor(entry)}
            <li class="slide-in-x grid grid-cols-[3.25rem_2.25rem_1fr_auto] items-baseline gap-3 py-2 border-b border-dashed"
                style="border-color: color-mix(in oklab, var(--color-border) 60%, transparent)">
              <span class="font-mono text-[10px] tabular" style="color: var(--color-fg-faint)">{fmtTime(entry.ts)}</span>
              <span class="font-mono text-[10px] tabular smallcaps text-center px-1.5 py-0.5 rounded-sm border"
                    style="border-color: {meta.color}; color: {meta.color}; background: color-mix(in oklab, {meta.color} 10%, transparent)">
                {meta.abbr}
              </span>
              <span class="text-sm leading-snug" style="color: var(--color-fg)">
                {entry.content}
              </span>
              {#if entry.source}
                <span class="font-mono text-[10px] tabular smallcaps justify-self-end" style="color: var(--color-fg-faint)">
                  {entry.source}
                </span>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </section>

  <!-- channel rails: 10 narrow vertical bars on the right edge -->
  <aside class="hidden md:flex relative flex-col w-[64px] border-l"
         style="background: color-mix(in oklab, var(--color-bg-elev) 60%, transparent)">
    <div class="sticky top-14 flex flex-col gap-1 py-4 px-2">
      {#each CHANNELS as ch (ch.kind)}
        <div class="flex flex-col items-center gap-1.5 py-2 relative">
          <span class="font-mono text-[9px] smallcaps tabular" style="color: var(--color-fg-faint)">
            {ch.abbr}
          </span>
          <div class="relative w-[3px] h-12 rounded-full overflow-hidden"
               style="background: color-mix(in oklab, var(--color-ch-{ch.kind}) 18%, transparent)">
            {#if pulseFor[ch.kind]}
              <div class="absolute inset-0 channel-pulse"
                   style="background: var(--color-ch-{ch.kind}); box-shadow: 0 0 8px var(--color-ch-{ch.kind})">
              </div>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  </aside>
</div>
