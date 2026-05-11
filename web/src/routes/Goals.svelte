<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "$lib/api/client";

  type Goal = {
    id: string;
    description: string;
    status: string;
    source: string;
    priority: number;
    confidence: number;
    appraisal_weight: number;
    review_notes: string;
    created_at: number;
    updated_at: number;
  };

  let goals = $state<Goal[]>([]);
  let draftDesc = $state("");
  let draftPriority = $state(0.5);
  let creating = $state(false);
  let error = $state<string | null>(null);
  let editing = $state<string | null>(null);
  let editDraft = $state("");

  async function refresh() {
    try {
      goals = await api<Goal[]>("/ui/api/goals");
    } catch (err) {
      error = err instanceof Error ? err.message : "failed to load";
    }
  }

  async function create(e: Event) {
    e.preventDefault();
    if (!draftDesc.trim() || creating) return;
    creating = true;
    error = null;
    try {
      await api("/ui/api/goals", { json: { description: draftDesc.trim(), priority: draftPriority } });
      draftDesc = "";
      draftPriority = 0.5;
      await refresh();
    } catch (err) {
      error = err instanceof Error ? err.message : "create failed";
    } finally {
      creating = false;
    }
  }

  async function retire(id: string) {
    if (!confirm("retire this goal?")) return;
    try {
      await api(`/ui/api/goals/${encodeURIComponent(id)}`, { method: "DELETE" });
      await refresh();
    } catch (err) {
      error = err instanceof Error ? err.message : "retire failed";
    }
  }

  async function commitEdit(id: string) {
    if (!editDraft.trim()) return;
    try {
      await api(`/ui/api/goals/${encodeURIComponent(id)}`, {
        method: "PATCH",
        json: { description: editDraft.trim() },
      });
      editing = null;
      editDraft = "";
      await refresh();
    } catch (err) {
      error = err instanceof Error ? err.message : "edit failed";
    }
  }

  function startEdit(g: Goal) {
    editing = g.id;
    editDraft = g.description;
  }

  function statusColor(status: string): string {
    return status === "active" ? "var(--color-ok)"
         : status === "paused" ? "var(--color-warn)"
         : status === "retired" ? "var(--color-fg-faint)"
         : "var(--color-fg-mute)";
  }

  function pri(p: number): string {
    return p.toFixed(2);
  }

  onMount(refresh);
</script>

<section class="max-w-4xl mx-auto px-5 md:px-8 py-8 grid gap-8">
  <header>
    <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">iv.</p>
    <h1 class="mt-1 font-display italic text-3xl tracking-tight" style="color: var(--color-fg)">
      goals
    </h1>
    <p class="mt-1 font-mono text-xs" style="color: var(--color-fg-mute)">
      {goals.length} total · ordered by status, priority
    </p>
  </header>

  <form onsubmit={create} class="border rounded-md p-4" style="background: var(--color-bg-elev)">
    <p class="font-mono text-[10px] smallcaps tabular mb-3" style="color: var(--color-fg-faint)">
      add goal
    </p>
    <div class="grid gap-3">
      <textarea
        bind:value={draftDesc}
        placeholder="describe the goal in one sentence"
        rows="2"
        disabled={creating}
        class="w-full px-3 py-2 rounded-md border bg-transparent outline-none text-sm focus:border-[color:var(--color-accent)] resize-none"
        style="background: var(--color-bg-sunk); color: var(--color-fg)"
      ></textarea>
      <div class="grid grid-cols-[1fr_auto] gap-3 items-end">
        <label class="grid gap-1">
          <span class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-mute)">
            priority · {pri(draftPriority)}
          </span>
          <input type="range" min="0" max="1" step="0.05" bind:value={draftPriority} disabled={creating} class="accent-[color:var(--color-accent)]" />
        </label>
        <button
          type="submit"
          disabled={creating || !draftDesc.trim()}
          class="h-10 px-5 rounded-md font-mono text-xs smallcaps tracking-wider transition-opacity disabled:opacity-40"
          style="background: var(--color-accent); color: var(--color-accent-fg)"
        >
          {creating ? "adding…" : "add"}
        </button>
      </div>
    </div>
    {#if error}
      <p class="mt-2 font-mono text-xs" style="color: var(--color-danger)">{error}</p>
    {/if}
  </form>

  <ul class="grid gap-2">
    {#each goals as g (g.id)}
      <li class="border rounded-md p-4" style="background: var(--color-bg-elev)">
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0 flex-1">
            {#if editing === g.id}
              <textarea
                bind:value={editDraft}
                rows="2"
                class="w-full px-2 py-1.5 rounded-sm border bg-transparent text-sm focus:border-[color:var(--color-accent)] resize-none"
                style="background: var(--color-bg-sunk); color: var(--color-fg)"
              ></textarea>
              <div class="mt-2 flex gap-2">
                <button onclick={() => commitEdit(g.id)}
                  class="h-7 px-3 rounded-sm font-mono text-[10px] smallcaps tabular"
                  style="background: var(--color-accent); color: var(--color-accent-fg)">save</button>
                <button onclick={() => (editing = null)}
                  class="h-7 px-3 rounded-sm font-mono text-[10px] smallcaps tabular border"
                  style="border-color: var(--color-border-hot); color: var(--color-fg-mute)">cancel</button>
              </div>
            {:else}
              <p class="text-sm leading-relaxed" style="color: var(--color-fg)">{g.description}</p>
              <div class="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 font-mono text-[10px] tabular smallcaps"
                   style="color: var(--color-fg-faint)">
                <span style="color: {statusColor(g.status)}">{g.status}</span>
                <span>priority · {pri(g.priority)}</span>
                <span>conf · {pri(g.confidence)}</span>
                <span>source · {g.source}</span>
              </div>
            {/if}
          </div>
          {#if editing !== g.id}
            <div class="flex flex-col gap-1 shrink-0">
              <button
                onclick={() => startEdit(g)}
                class="px-2.5 py-1 font-mono text-[10px] smallcaps tabular border rounded-sm transition-colors"
                style="border-color: var(--color-border-hot); color: var(--color-fg-mute)"
              >
                edit
              </button>
              {#if g.status !== "retired"}
                <button
                  onclick={() => retire(g.id)}
                  class="px-2.5 py-1 font-mono text-[10px] smallcaps tabular border rounded-sm transition-colors"
                  style="border-color: var(--color-danger); color: var(--color-danger)"
                >
                  retire
                </button>
              {/if}
            </div>
          {/if}
        </div>
      </li>
    {/each}
  </ul>
</section>
