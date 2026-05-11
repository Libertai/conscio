<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { api } from "$lib/api/client";

  let trace = $state("");
  let modelCtx = $state("");
  let copied = $state<"trace" | "ctx" | null>(null);
  let timer: ReturnType<typeof setInterval> | null = null;

  async function refresh() {
    try {
      const [t, m] = await Promise.all([
        api<{ trace: string }>("/ui/api/trace"),
        api<{ model_context: string }>("/ui/api/model_context"),
      ]);
      trace = t.trace;
      modelCtx = m.model_context;
    } catch (_) {
      /* keep stale text */
    }
  }

  async function copy(target: "trace" | "ctx") {
    const text = target === "trace" ? trace : modelCtx;
    try {
      await navigator.clipboard.writeText(text);
      copied = target;
      setTimeout(() => (copied = null), 1400);
    } catch (_) {
      copied = null;
    }
  }

  onMount(() => {
    refresh();
    timer = setInterval(refresh, 8000);
  });
  onDestroy(() => {
    if (timer) clearInterval(timer);
  });
</script>

<section class="max-w-5xl mx-auto px-5 md:px-8 py-8 grid gap-8">
  <header>
    <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">vii.</p>
    <h1 class="mt-1 font-display italic text-3xl tracking-tight" style="color: var(--color-fg)">
      trace
    </h1>
    <p class="mt-1 font-mono text-xs" style="color: var(--color-fg-mute)">
      raw cognitive trace + the assembled model context · refreshing every 8s
    </p>
  </header>

  <article class="border rounded-md overflow-hidden" style="background: var(--color-bg-elev)">
    <div class="flex items-baseline justify-between px-4 py-2.5 border-b">
      <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
        cognitive trace · {trace.length} chars
      </p>
      <button
        onclick={() => copy("trace")}
        class="font-mono text-[10px] smallcaps tabular px-2 py-1 rounded-sm border transition-colors"
        style="border-color: {copied === 'trace' ? 'var(--color-ok)' : 'var(--color-border-hot)'}; color: {copied === 'trace' ? 'var(--color-ok)' : 'var(--color-fg-mute)'}"
      >
        {copied === "trace" ? "copied" : "copy"}
      </button>
    </div>
    <pre class="px-4 py-3 font-mono text-[11px] leading-relaxed overflow-x-auto max-h-[60vh] whitespace-pre-wrap"
         style="color: var(--color-fg)">{trace || "(empty)"}</pre>
  </article>

  <article class="border rounded-md overflow-hidden" style="background: var(--color-bg-elev)">
    <div class="flex items-baseline justify-between px-4 py-2.5 border-b">
      <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
        model context · {modelCtx.length} chars
      </p>
      <button
        onclick={() => copy("ctx")}
        class="font-mono text-[10px] smallcaps tabular px-2 py-1 rounded-sm border transition-colors"
        style="border-color: {copied === 'ctx' ? 'var(--color-ok)' : 'var(--color-border-hot)'}; color: {copied === 'ctx' ? 'var(--color-ok)' : 'var(--color-fg-mute)'}"
      >
        {copied === "ctx" ? "copied" : "copy"}
      </button>
    </div>
    <pre class="px-4 py-3 font-mono text-[11px] leading-relaxed overflow-x-auto max-h-[60vh] whitespace-pre-wrap"
         style="color: var(--color-fg)">{modelCtx || "(empty)"}</pre>
  </article>
</section>
