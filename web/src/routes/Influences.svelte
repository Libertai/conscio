<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "$lib/api/client";

  type Influence = {
    id: string;
    kind: string;
    content: string;
    source: string;
    status: string;
    appraisal: string;
    created_at: number;
    updated_at: number;
  };

  let influences = $state<Influence[]>([]);
  let kind = $state<"goal" | "constraint">("goal");
  let draft = $state("");
  let posting = $state(false);
  let error = $state<string | null>(null);

  async function refresh() {
    try {
      influences = await api<Influence[]>("/ui/api/influences");
    } catch (err) {
      error = err instanceof Error ? err.message : "failed to load";
    }
  }

  async function submit(e: Event) {
    e.preventDefault();
    if (!draft.trim() || posting) return;
    posting = true;
    error = null;
    try {
      await api(`/ui/api/influence/${kind}`, { json: { content: draft.trim() } });
      draft = "";
      await refresh();
    } catch (err) {
      error = err instanceof Error ? err.message : "submit failed";
    } finally {
      posting = false;
    }
  }

  async function retire(id: string) {
    if (!confirm("retire this influence?")) return;
    try {
      await api(`/ui/api/influences/${encodeURIComponent(id)}`, { method: "DELETE" });
      await refresh();
    } catch (err) {
      error = err instanceof Error ? err.message : "retire failed";
    }
  }

  function statusColor(s: string): string {
    return s === "adopted" || s === "active" ? "var(--color-ok)"
         : s === "rejected" ? "var(--color-danger)"
         : s === "negotiating" ? "var(--color-warn)"
         : s === "deferred" ? "var(--color-ch-intention)"
         : "var(--color-fg-faint)";
  }

  onMount(refresh);
</script>

<section class="max-w-4xl mx-auto px-5 md:px-8 py-8 grid gap-8">
  <header>
    <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">viii.</p>
    <h1 class="mt-1 font-display italic text-3xl tracking-tight" style="color: var(--color-fg)">
      influences
    </h1>
    <p class="mt-1 font-mono text-xs" style="color: var(--color-fg-mute)">
      operator-supplied goals + constraints; the agent appraises each on receipt
    </p>
  </header>

  <form onsubmit={submit} class="border rounded-md p-4" style="background: var(--color-bg-elev)">
    <div class="flex gap-1 mb-3">
      <button type="button"
        onclick={() => (kind = "goal")}
        class="px-3 py-1.5 rounded-sm font-mono text-[11px] smallcaps tabular border transition-colors"
        style="border-color: {kind === 'goal' ? 'var(--color-accent)' : 'var(--color-border)'}; color: {kind === 'goal' ? 'var(--color-fg)' : 'var(--color-fg-mute)'}; background: {kind === 'goal' ? 'color-mix(in oklab, var(--color-accent) 12%, transparent)' : 'transparent'}">
        goal
      </button>
      <button type="button"
        onclick={() => (kind = "constraint")}
        class="px-3 py-1.5 rounded-sm font-mono text-[11px] smallcaps tabular border transition-colors"
        style="border-color: {kind === 'constraint' ? 'var(--color-accent)' : 'var(--color-border)'}; color: {kind === 'constraint' ? 'var(--color-fg)' : 'var(--color-fg-mute)'}; background: {kind === 'constraint' ? 'color-mix(in oklab, var(--color-accent) 12%, transparent)' : 'transparent'}">
        constraint
      </button>
    </div>
    <div class="grid grid-cols-[1fr_auto] gap-3 items-end">
      <textarea
        bind:value={draft}
        rows="2"
        placeholder={kind === "goal" ? "what should the agent pursue?" : "what must the agent never do?"}
        disabled={posting}
        class="w-full px-3 py-2 rounded-md border bg-transparent outline-none text-sm focus:border-[color:var(--color-accent)] resize-none"
        style="background: var(--color-bg-sunk); color: var(--color-fg)"
      ></textarea>
      <button
        type="submit"
        disabled={posting || !draft.trim()}
        class="h-10 px-5 rounded-md font-mono text-xs smallcaps tracking-wider transition-opacity disabled:opacity-40"
        style="background: var(--color-accent); color: var(--color-accent-fg)"
      >
        {posting ? "submitting…" : "submit"}
      </button>
    </div>
    {#if error}
      <p class="mt-2 font-mono text-xs" style="color: var(--color-danger)">{error}</p>
    {/if}
  </form>

  <ul class="grid gap-2">
    {#each influences as inf (inf.id)}
      <li class="border rounded-md p-4 grid grid-cols-[1fr_auto] gap-3 items-start" style="background: var(--color-bg-elev)">
        <div class="min-w-0">
          <div class="flex items-baseline gap-2 mb-1">
            <span class="font-mono text-[10px] tabular smallcaps" style="color: var(--color-fg-faint)">{inf.kind}</span>
            <span class="font-mono text-[10px] tabular smallcaps" style="color: {statusColor(inf.status)}">{inf.status}</span>
          </div>
          <p class="text-sm leading-relaxed" style="color: var(--color-fg)">{inf.content}</p>
          {#if inf.appraisal}
            <p class="mt-1.5 font-mono text-[11px]" style="color: var(--color-fg-mute)">
              <span class="smallcaps tabular text-[10px]" style="color: var(--color-fg-faint)">appraisal · </span>{inf.appraisal}
            </p>
          {/if}
        </div>
        {#if inf.status !== "retired" && inf.status !== "rejected"}
          <button
            onclick={() => retire(inf.id)}
            class="px-2.5 py-1 font-mono text-[10px] smallcaps tabular border rounded-sm"
            style="border-color: var(--color-danger); color: var(--color-danger)"
          >
            retire
          </button>
        {/if}
      </li>
    {/each}
  </ul>
</section>
