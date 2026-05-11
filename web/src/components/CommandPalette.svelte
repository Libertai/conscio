<script lang="ts">
  import { push } from "svelte-spa-router";

  type Item = { href: string; label: string; hint: string };
  const items: Item[] = [
    { href: "/stream", label: "stream", hint: "live activity feed" },
    { href: "/chat", label: "chat", hint: "address the agent" },
    { href: "/projects", label: "projects", hint: "active project plans + tasks" },
    { href: "/goals", label: "goals", hint: "long-term goal CRUD" },
    { href: "/influences", label: "influences", hint: "goal + constraint queue" },
    { href: "/memory", label: "memory", hint: "search facts + skills" },
    { href: "/episodes", label: "episodes", hint: "every event processed" },
    { href: "/trace", label: "trace", hint: "cognitive trace + model context" },
    { href: "/settings", label: "settings", hint: "theme + session" },
  ];

  let open = $state(false);
  let query = $state("");
  let activeIndex = $state(0);

  let filtered = $derived(() => {
    const q = query.toLowerCase().trim();
    if (!q) return items;
    return items.filter((it) => it.label.includes(q) || it.hint.includes(q));
  });

  function close() {
    open = false;
    query = "";
    activeIndex = 0;
  }

  function go(item: Item) {
    close();
    push(item.href);
  }

  function onGlobalKey(e: KeyboardEvent) {
    const inField = ["INPUT", "TEXTAREA"].includes((e.target as HTMLElement)?.tagName);
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      open = !open;
      return;
    }
    if (!inField && e.key === "?" && !open) {
      e.preventDefault();
      open = true;
      return;
    }
    if (!open) return;
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      activeIndex = Math.min(activeIndex + 1, filtered().length - 1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const target = filtered()[activeIndex];
      if (target) go(target);
    }
  }

  $effect(() => {
    if (query !== undefined) activeIndex = 0;
  });
</script>

<svelte:window onkeydown={onGlobalKey} />

{#if open}
  <div
    class="fixed inset-0 z-50 grid place-items-start pt-[16vh] px-4 backdrop-blur-sm"
    style="background: color-mix(in oklab, var(--color-bg) 60%, transparent)"
    onclick={(e) => { if (e.target === e.currentTarget) close(); }}
    role="dialog"
    aria-modal="true"
  >
    <div class="w-full max-w-md rounded-lg border overflow-hidden shadow-2xl"
         style="background: var(--color-bg-elev); border-color: var(--color-border-hot)">
      <div class="border-b" style="border-color: var(--color-border)">
        <input
          type="text"
          bind:value={query}
          autofocus
          placeholder="navigate or search… (↑↓ / ⏎)"
          class="w-full h-12 px-4 bg-transparent outline-none text-sm"
          style="color: var(--color-fg)"
        />
      </div>
      <ul class="max-h-72 overflow-y-auto">
        {#each filtered() as item, i (item.href)}
          <li>
            <button
              type="button"
              onclick={() => go(item)}
              onmouseenter={() => (activeIndex = i)}
              class="w-full text-left px-4 py-2.5 flex items-baseline justify-between gap-4 transition-colors"
              style="background: {i === activeIndex ? 'color-mix(in oklab, var(--color-accent) 14%, transparent)' : 'transparent'}"
            >
              <span class="font-mono text-sm smallcaps tabular tracking-wider" style="color: var(--color-fg)">{item.label}</span>
              <span class="font-mono text-[11px]" style="color: var(--color-fg-mute)">{item.hint}</span>
            </button>
          </li>
        {:else}
          <li class="px-4 py-6 font-mono text-xs text-center" style="color: var(--color-fg-faint)">
            no matches.
          </li>
        {/each}
      </ul>
      <div class="px-4 py-2 border-t font-mono text-[10px] smallcaps tabular flex items-center justify-between"
           style="border-color: var(--color-border); color: var(--color-fg-faint)">
        <span>↑ ↓ navigate · ⏎ open · esc close</span>
        <span>⌘k / ?</span>
      </div>
    </div>
  </div>
{/if}
