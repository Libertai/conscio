<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "$lib/api/client";

  type Task = {
    id: string;
    description: string;
    status: string;
    tool_name?: string | null;
    result?: string | null;
    created_at: number;
    updated_at: number;
  };

  type Project = {
    id: string;
    title: string;
    status: string;
    goal_id: string | null;
    created_at: number;
    updated_at: number;
    tasks?: Task[];
  };

  let projects = $state<Project[]>([]);
  let selectedId = $state<string | null>(null);
  let detail = $state<Project | null>(null);
  let loadError = $state<string | null>(null);
  let acting = $state<string | null>(null);

  async function refresh() {
    try {
      projects = await api<Project[]>("/ui/api/projects");
      if (selectedId) await loadDetail(selectedId);
    } catch (err) {
      loadError = err instanceof Error ? err.message : "failed to load projects";
    }
  }

  async function loadDetail(id: string) {
    try {
      detail = await api<Project>(`/ui/api/projects/${encodeURIComponent(id)}`);
      selectedId = id;
    } catch (err) {
      loadError = err instanceof Error ? err.message : "failed to load project";
    }
  }

  async function setStatus(id: string, next: "paused" | "active") {
    acting = id;
    try {
      await api(`/ui/api/projects/${encodeURIComponent(id)}/${next === "active" ? "resume" : "pause"}`, { method: "POST" });
      await refresh();
    } finally {
      acting = null;
    }
  }

  function shortId(id: string): string {
    return id.length > 12 ? id.slice(0, 12) : id;
  }

  onMount(refresh);
</script>

<section class="min-h-[calc(100vh-3.5rem-4rem)] md:min-h-[calc(100vh-3.5rem)] grid md:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
  <!-- list pane -->
  <aside class="border-b md:border-b-0 md:border-r overflow-y-auto">
    <header class="px-5 md:px-6 py-4 border-b">
      <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">iii.</p>
      <h1 class="mt-1 font-display italic text-2xl tracking-tight" style="color: var(--color-fg)">
        projects
      </h1>
      <p class="mt-1 font-mono text-xs" style="color: var(--color-fg-mute)">
        {projects.length} on file
      </p>
    </header>

    {#if loadError}
      <p class="px-5 py-3 font-mono text-xs" style="color: var(--color-danger)">{loadError}</p>
    {/if}

    {#if projects.length === 0 && !loadError}
      <p class="px-5 py-8 font-mono text-xs smallcaps tabular text-center" style="color: var(--color-fg-faint)">
        no projects yet.
      </p>
    {:else}
      <ul class="divide-y">
        {#each projects as p (p.id)}
          <li>
            <button
              type="button"
              onclick={() => loadDetail(p.id)}
              class="w-full text-left px-5 md:px-6 py-4 transition-colors hover:bg-[color-mix(in_oklab,var(--color-fg-faint)_8%,transparent)]"
              style="background: {selectedId === p.id ? 'color-mix(in oklab, var(--color-accent) 10%, transparent)' : 'transparent'}; border-left: 2px solid {selectedId === p.id ? 'var(--color-accent)' : 'transparent'}"
            >
              <div class="flex items-baseline justify-between gap-3">
                <span class="text-sm leading-snug" style="color: var(--color-fg)">{p.title}</span>
                <span class="font-mono text-[10px] tabular smallcaps shrink-0"
                      style="color: {p.status === 'active' ? 'var(--color-ok)' : p.status === 'paused' ? 'var(--color-warn)' : 'var(--color-fg-faint)'}">
                  {p.status}
                </span>
              </div>
              <p class="mt-1 font-mono text-[10px] tabular" style="color: var(--color-fg-faint)">
                {shortId(p.id)}
              </p>
            </button>
          </li>
        {/each}
      </ul>
    {/if}
  </aside>

  <!-- detail pane -->
  <section class="overflow-y-auto">
    {#if detail}
      <header class="px-5 md:px-8 py-5 border-b">
        <div class="flex items-baseline justify-between gap-4 flex-wrap">
          <div>
            <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
              project · {shortId(detail.id)}
            </p>
            <h2 class="mt-2 font-display italic text-2xl leading-tight" style="color: var(--color-fg)">
              {detail.title}
            </h2>
          </div>
          <div class="flex items-center gap-2">
            {#if detail.status === "active"}
              <button
                disabled={acting === detail.id}
                onclick={() => setStatus(detail!.id, "paused")}
                class="h-9 px-4 rounded-md border font-mono text-[11px] smallcaps tabular transition-opacity disabled:opacity-50"
                style="border-color: var(--color-warn); color: var(--color-warn)"
              >
                pause
              </button>
            {:else if detail.status === "paused"}
              <button
                disabled={acting === detail.id}
                onclick={() => setStatus(detail!.id, "active")}
                class="h-9 px-4 rounded-md border font-mono text-[11px] smallcaps tabular transition-opacity disabled:opacity-50"
                style="border-color: var(--color-ok); color: var(--color-ok)"
              >
                resume
              </button>
            {/if}
          </div>
        </div>
      </header>

      <div class="px-5 md:px-8 py-6">
        <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
          tasks · {detail.tasks?.length ?? 0}
        </p>

        {#if !detail.tasks || detail.tasks.length === 0}
          <p class="mt-6 font-mono text-xs" style="color: var(--color-fg-mute)">
            no tasks yet.
          </p>
        {:else}
          <ul class="mt-4 space-y-2.5">
            {#each detail.tasks as task (task.id)}
              <li class="border rounded-md p-3.5" style="background: var(--color-bg-elev)">
                <div class="flex items-baseline justify-between gap-3">
                  <span class="text-sm leading-snug" style="color: var(--color-fg)">
                    {task.description}
                  </span>
                  <span class="font-mono text-[10px] tabular smallcaps shrink-0"
                        style="color: {task.status === 'done' ? 'var(--color-ok)' : task.status === 'active' ? 'var(--color-accent)' : task.status === 'blocked' ? 'var(--color-danger)' : 'var(--color-fg-mute)'}">
                    {task.status}
                  </span>
                </div>
                {#if task.tool_name}
                  <p class="mt-1.5 font-mono text-[10px] tabular smallcaps" style="color: var(--color-fg-faint)">
                    tool · {task.tool_name}
                  </p>
                {/if}
                {#if task.result}
                  <p class="mt-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap" style="color: var(--color-fg-mute)">
                    {task.result}
                  </p>
                {/if}
              </li>
            {/each}
          </ul>
        {/if}
      </div>
    {:else}
      <div class="grid place-items-center min-h-[40vh] px-6">
        <p class="font-mono text-xs smallcaps tabular text-center" style="color: var(--color-fg-faint)">
          select a project to inspect.
        </p>
      </div>
    {/if}
  </section>
</section>
