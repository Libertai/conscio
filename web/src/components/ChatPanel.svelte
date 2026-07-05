<script lang="ts">
  import { onMount, tick } from "svelte";
  import { chat, loadHistory, sendMessage } from "$lib/stores/chat.svelte";
  import { chatStream } from "$lib/stores/events.svelte";

  let composer = $state("");
  let textarea: HTMLTextAreaElement | undefined;
  let scroller: HTMLDivElement | undefined;

  function fmtTime(ts: number): string {
    const d = new Date(ts * 1000);
    const pad = (n: number) => n.toString().padStart(2, "0");
    return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  async function submit(e: Event) {
    e.preventDefault();
    const text = composer.trim();
    if (!text) return;
    composer = "";
    await sendMessage(text);
    chatStream.reset();
    await tick();
    scroller?.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
  }

  function onKeydown(e: KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      submit(e);
    }
  }

  onMount(async () => {
    await loadHistory();
    await tick();
    scroller?.scrollTo({ top: scroller?.scrollHeight ?? 0 });
  });

  $effect(() => {
    // Re-scroll on each new message or streamed token.
    if ((chat.messages.length || chatStream.text) && scroller) {
      const isAtBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
      if (isAtBottom) scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
    }
  });
</script>

<div class="flex flex-col h-[calc(100vh-3.5rem-4rem)] md:h-[calc(100vh-3.5rem)]">
  <header class="flex items-baseline justify-between gap-4 px-5 md:px-8 py-4 border-b">
    <div class="flex items-baseline gap-3">
      <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">ii.</p>
      <h1 class="font-display italic text-2xl md:text-3xl tracking-tight" style="color: var(--color-fg)">
        operator console
      </h1>
    </div>
    <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
      session · {chat.sessionId}
    </p>
  </header>

  <div bind:this={scroller} class="flex-1 overflow-y-auto px-5 md:px-8 py-6">
    {#if chat.loadError}
      <p class="font-mono text-xs mb-4" style="color: var(--color-danger)">
        {chat.loadError}
      </p>
    {/if}

    {#if chat.messages.length === 0}
      <div class="grid place-items-center h-full">
        <p class="font-mono text-xs smallcaps tabular text-center" style="color: var(--color-fg-faint)">
          the line is open. address the agent directly.
        </p>
      </div>
    {:else}
      <ul class="space-y-5 max-w-3xl mx-auto">
        {#each chat.messages as msg (msg.id)}
          <li class="grid grid-cols-[4rem_1fr] gap-4 items-baseline">
            <span class="font-mono text-[10px] tabular smallcaps text-right pt-1"
                  style="color: var(--color-fg-faint)">
              {msg.role === "user" ? "operator" : "agent"}
              <br/>
              <span style="color: color-mix(in oklab, var(--color-fg-faint) 70%, transparent)">{fmtTime(msg.created_at)}</span>
            </span>
            <div class="prose-tight">
              <p class="text-base leading-relaxed whitespace-pre-wrap"
                 style="color: {msg.role === 'user' ? 'var(--color-fg)' : 'var(--color-fg)'}; opacity: {msg.role === 'user' ? 0.95 : 1}; border-left: 2px solid {msg.role === 'user' ? 'var(--color-border-hot)' : 'var(--color-accent)'}; padding-left: 0.875rem">
                {msg.content}
              </p>
              {#if msg.selected_action && msg.role === "agent"}
                <p class="mt-1 ml-3.5 font-mono text-[10px] tabular smallcaps"
                   style="color: var(--color-fg-faint)">
                  selected action · {msg.selected_action}
                </p>
              {/if}
            </div>
          </li>
        {/each}
        {#if chat.sending && chatStream.text}
          <li class="grid grid-cols-[4rem_1fr] gap-4 items-baseline">
            <span class="font-mono text-[10px] tabular smallcaps text-right pt-1"
                  style="color: var(--color-fg-faint)">
              agent
              <br/>
              <span style="color: color-mix(in oklab, var(--color-fg-faint) 70%, transparent)">live</span>
            </span>
            <div class="prose-tight">
              <p class="text-base leading-relaxed whitespace-pre-wrap"
                 style="color: var(--color-fg); opacity: 0.75; border-left: 2px solid var(--color-accent); padding-left: 0.875rem">
                {chatStream.text}<span class="animate-pulse">▍</span>
              </p>
            </div>
          </li>
        {/if}
      </ul>
    {/if}
  </div>

  <form onsubmit={submit} class="border-t px-5 md:px-8 py-4">
    <div class="max-w-3xl mx-auto grid grid-cols-[1fr_auto] gap-3 items-end">
      <textarea
        bind:this={textarea}
        bind:value={composer}
        onkeydown={onKeydown}
        rows="2"
        placeholder="address the agent · ⌘⏎ to send"
        disabled={chat.sending}
        class="w-full px-3 py-2.5 rounded-md border bg-transparent outline-none text-sm leading-relaxed
               focus:border-[color:var(--color-accent)] resize-none min-h-[44px] max-h-40"
        style="background: var(--color-bg-sunk); color: var(--color-fg)"
      ></textarea>
      <button
        type="submit"
        disabled={chat.sending || !composer.trim()}
        class="h-11 px-5 rounded-md font-mono text-xs smallcaps tracking-wider transition-opacity disabled:opacity-40"
        style="background: var(--color-accent); color: var(--color-accent-fg)"
      >
        {chat.sending ? "sending" : "send"}
      </button>
    </div>
  </form>
</div>
