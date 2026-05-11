<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "$lib/api/client";

  type Episode = {
    id: string;
    source: string;
    event_type: string;
    input: string;
    output: string;
    selected_action: string;
    created_at: number;
    metrics?: Record<string, number>;
  };

  let episodes = $state<Episode[]>([]);
  let loading = $state(false);
  let exhausted = $state(false);
  let error = $state<string | null>(null);
  let expanded = $state<Set<string>>(new Set());

  async function loadInitial() {
    loading = true;
    try {
      episodes = await api<Episode[]>("/ui/api/episodes?limit=20");
      exhausted = episodes.length < 20;
    } catch (err) {
      error = err instanceof Error ? err.message : "failed to load";
    } finally {
      loading = false;
    }
  }

  async function loadMore() {
    if (loading || exhausted || episodes.length === 0) return;
    loading = true;
    const oldest = episodes[episodes.length - 1].created_at;
    try {
      const more = await api<Episode[]>(`/ui/api/episodes?limit=20&before=${oldest}`);
      episodes = [...episodes, ...more];
      if (more.length < 20) exhausted = true;
    } catch (err) {
      error = err instanceof Error ? err.message : "failed to load more";
    } finally {
      loading = false;
    }
  }

  function toggle(id: string) {
    const next = new Set(expanded);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    expanded = next;
  }

  function fmt(ts: number): string {
    const d = new Date(ts * 1000);
    return d.toLocaleString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", month: "short", day: "numeric" });
  }

  onMount(loadInitial);
</script>

<section class="max-w-4xl mx-auto px-5 md:px-8 py-8 grid gap-6">
  <header>
    <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">vi.</p>
    <h1 class="mt-1 font-display italic text-3xl tracking-tight" style="color: var(--color-fg)">
      episodes
    </h1>
    <p class="mt-1 font-mono text-xs" style="color: var(--color-fg-mute)">
      durable record of every event the agent has processed
    </p>
  </header>

  {#if error}
    <p class="font-mono text-xs" style="color: var(--color-danger)">{error}</p>
  {/if}

  <ul class="grid gap-2">
    {#each episodes as ep (ep.id)}
      {@const open = expanded.has(ep.id)}
      <li class="border rounded-md" style="background: var(--color-bg-elev)">
        <button
          type="button"
          onclick={() => toggle(ep.id)}
          class="w-full text-left px-4 py-3 grid grid-cols-[auto_1fr_auto] gap-3 items-baseline transition-colors hover:bg-[color-mix(in_oklab,var(--color-fg-faint)_5%,transparent)]"
        >
          <span class="font-mono text-[10px] tabular smallcaps" style="color: var(--color-fg-faint)">
            {fmt(ep.created_at)}
          </span>
          <span class="text-sm leading-snug truncate" style="color: var(--color-fg)">
            {ep.input || "(no input)"}
          </span>
          <span class="font-mono text-[10px] tabular smallcaps shrink-0"
                style="color: var(--color-accent)">
            {ep.selected_action ?? ep.event_type}
          </span>
        </button>
        {#if open}
          <div class="px-4 pb-4 grid gap-3 text-sm border-t" style="border-color: var(--color-border)">
            <div class="pt-3">
              <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">input</p>
              <p class="mt-1 whitespace-pre-wrap" style="color: var(--color-fg)">{ep.input || "—"}</p>
            </div>
            <div>
              <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">output</p>
              <p class="mt-1 whitespace-pre-wrap" style="color: var(--color-fg)">{ep.output || "—"}</p>
            </div>
            {#if ep.metrics}
              <div class="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] tabular smallcaps"
                   style="color: var(--color-fg-faint)">
                {#each Object.entries(ep.metrics) as [k, v]}
                  <span>{k} · {v}</span>
                {/each}
              </div>
            {/if}
          </div>
        {/if}
      </li>
    {/each}
  </ul>

  <div class="flex justify-center">
    {#if exhausted}
      <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
        no earlier episodes.
      </p>
    {:else}
      <button
        onclick={loadMore}
        disabled={loading}
        class="h-10 px-5 rounded-md border font-mono text-[11px] smallcaps tracking-wider transition-opacity disabled:opacity-50"
        style="border-color: var(--color-border-hot); color: var(--color-fg)"
      >
        {loading ? "loading…" : "load more"}
      </button>
    {/if}
  </div>
</section>
