<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "$lib/api/client";

  type Fact = { fact: string; source: string; confidence: string; created_at?: number; updated_at?: number };
  type Skill = { skill: string; description?: string; steps?: string; use_count?: number };

  let recentFacts = $state<Fact[]>([]);
  let skills = $state<Skill[]>([]);
  let q = $state("");
  let results = $state<Fact[]>([]);
  let searching = $state(false);
  let error = $state<string | null>(null);

  async function refresh() {
    try {
      const m = await api<{ facts: Fact[]; skills: Skill[] }>("/ui/api/memory/recent?limit=30");
      recentFacts = m.facts;
      skills = m.skills;
    } catch (err) {
      error = err instanceof Error ? err.message : "failed to load";
    }
  }

  let searchTimer: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    const query = q.trim();
    if (searchTimer) clearTimeout(searchTimer);
    if (!query) {
      results = [];
      return;
    }
    searchTimer = setTimeout(async () => {
      searching = true;
      try {
        results = await api<Fact[]>(`/ui/api/memory/search?q=${encodeURIComponent(query)}&limit=30`);
      } catch (err) {
        error = err instanceof Error ? err.message : "search failed";
      } finally {
        searching = false;
      }
    }, 250);
  });

  onMount(refresh);
</script>

<section class="max-w-5xl mx-auto px-5 md:px-8 py-8 grid gap-8">
  <header>
    <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">v.</p>
    <h1 class="mt-1 font-display italic text-3xl tracking-tight" style="color: var(--color-fg)">
      memory
    </h1>
    <p class="mt-1 font-mono text-xs" style="color: var(--color-fg-mute)">
      semantic facts + procedural skills · accumulated across sessions
    </p>
  </header>

  <div>
    <label class="grid gap-1.5 max-w-2xl">
      <span class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-mute)">
        search
      </span>
      <input
        type="search"
        bind:value={q}
        placeholder="probe the agent's knowledge…"
        class="h-11 px-3.5 rounded-md border bg-transparent outline-none text-sm focus:border-[color:var(--color-accent)]"
        style="background: var(--color-bg-sunk); color: var(--color-fg)"
      />
    </label>

    {#if q.trim()}
      <div class="mt-4">
        <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
          {searching ? "searching…" : `${results.length} matches`}
        </p>
        <ul class="mt-2 space-y-1.5">
          {#each results as r (r.fact)}
            <li class="px-3 py-2 rounded-md border text-sm" style="background: var(--color-bg-elev)">
              <p style="color: var(--color-fg)">{r.fact}</p>
              <p class="mt-1 font-mono text-[10px] tabular smallcaps" style="color: var(--color-fg-faint)">
                {r.source ?? "agent"} · {r.confidence ?? "MEDIUM"}
              </p>
            </li>
          {/each}
        </ul>
      </div>
    {/if}
  </div>

  <div class="grid md:grid-cols-2 gap-8">
    <div>
      <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
        recent facts · {recentFacts.length}
      </p>
      {#if recentFacts.length === 0}
        <p class="mt-3 font-mono text-xs" style="color: var(--color-fg-mute)">no facts stored yet.</p>
      {:else}
        <ul class="mt-3 space-y-1.5">
          {#each recentFacts as f (f.fact)}
            <li class="px-3 py-2 rounded-md text-sm border-l-2"
                style="background: var(--color-bg-elev); border-color: var(--color-ch-memory); color: var(--color-fg)">
              {f.fact}
            </li>
          {/each}
        </ul>
      {/if}
    </div>

    <div>
      <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
        skills · {skills.length}
      </p>
      {#if skills.length === 0}
        <p class="mt-3 font-mono text-xs" style="color: var(--color-fg-mute)">no skills yet.</p>
      {:else}
        <ul class="mt-3 space-y-1.5">
          {#each skills as s (s.skill)}
            <li class="px-3 py-2 rounded-md text-sm border-l-2"
                style="background: var(--color-bg-elev); border-color: var(--color-ch-plan); color: var(--color-fg)">
              <p>{s.skill}</p>
              {#if s.description}
                <p class="mt-1 font-mono text-[11px]" style="color: var(--color-fg-mute)">{s.description}</p>
              {/if}
              <p class="mt-1 font-mono text-[10px] tabular smallcaps" style="color: var(--color-fg-faint)">
                used · {s.use_count ?? 0}
              </p>
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </div>

  {#if error}
    <p class="font-mono text-xs" style="color: var(--color-danger)">{error}</p>
  {/if}
</section>
