<script lang="ts">
  import { onMount, tick } from "svelte";
  import {
    chat,
    loadSessions,
    selectSession,
    createSession,
    deleteSession,
    sendMessage,
  } from "$lib/stores/chat.svelte";
  import { chatStream } from "$lib/stores/events.svelte";

  // The default session ships with the backend and cannot be deleted (the
  // server rejects it with 400), so we never offer a delete control for it.
  const DEFAULT_SESSION = "main";

  let composer = $state("");
  let newTitle = $state("");
  let creating = $state(false);
  let confirmingDelete = $state<string | null>(null);
  let textarea: HTMLTextAreaElement | undefined;
  let scroller: HTMLDivElement | undefined;

  function fmtTime(ts: number): string {
    const d = new Date(ts * 1000);
    const pad = (n: number) => n.toString().padStart(2, "0");
    return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  function sessionLabel(id: string, title: string | null): string {
    if (title && title.trim()) return title;
    return id === DEFAULT_SESSION ? "operator console" : "untitled";
  }

  const activeTitle = $derived(
    sessionLabel(
      chat.activeId,
      chat.sessions.find((s) => s.id === chat.activeId)?.title ?? null,
    ),
  );

  async function scrollToBottom(smooth = false) {
    await tick();
    scroller?.scrollTo({ top: scroller.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }

  async function submit(e: Event) {
    e.preventDefault();
    const text = composer.trim();
    if (!text) return;
    composer = "";
    await sendMessage(text);
    chatStream.reset();
    await scrollToBottom(true);
  }

  function onKeydown(e: KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      submit(e);
    }
  }

  async function pick(id: string) {
    if (id === chat.activeId) return;
    confirmingDelete = null;
    await selectSession(id);
    await scrollToBottom();
  }

  async function createNew(e: Event) {
    e.preventDefault();
    if (creating) return;
    creating = true;
    try {
      await createSession(newTitle);
      newTitle = "";
      await scrollToBottom();
      textarea?.focus();
    } finally {
      creating = false;
    }
  }

  async function confirmDelete(id: string) {
    await deleteSession(id);
    confirmingDelete = null;
  }

  onMount(async () => {
    await loadSessions();
    await scrollToBottom();
  });

  $effect(() => {
    // Re-scroll on each new message or streamed token, but only if the operator
    // is already near the bottom (don't yank them away from scrollback).
    if ((chat.messages.length || chatStream.text) && scroller) {
      const isAtBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
      if (isAtBottom) scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
    }
  });

  // The live token bubble belongs to the session that initiated the send; never
  // render it while viewing a different thread.
  const showStreaming = $derived(
    chat.sendingSession === chat.activeId && !!chatStream.text,
  );
</script>

<section
  class="flex flex-col md:grid md:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]
         h-[calc(100vh-3.5rem-4rem)] md:h-[calc(100vh-3.5rem)]"
>
  <!-- session rail -->
  <aside class="flex flex-col min-h-0 max-h-[40%] md:max-h-none border-b md:border-b-0 md:border-r">
    <header class="shrink-0 px-5 md:px-6 py-4 border-b">
      <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">sessions</p>
      <p class="mt-1 font-mono text-xs" style="color: var(--color-fg-mute)">
        {chat.sessions.length} open
      </p>
    </header>

    <form onsubmit={createNew} class="shrink-0 px-5 md:px-6 py-3 border-b">
      <div class="grid grid-cols-[1fr_auto] gap-2 items-center">
        <input
          bind:value={newTitle}
          placeholder="title (optional)"
          disabled={creating}
          class="w-full px-2.5 py-1.5 rounded-md border bg-transparent outline-none text-sm
                 focus:border-[color:var(--color-accent)]"
          style="background: var(--color-bg-sunk); color: var(--color-fg)"
        />
        <button
          type="submit"
          disabled={creating}
          class="h-8 px-3 rounded-md font-mono text-[10px] smallcaps tracking-wider transition-opacity disabled:opacity-40"
          style="background: var(--color-accent); color: var(--color-accent-fg)"
        >
          {creating ? "…" : "new"}
        </button>
      </div>
      {#if chat.sessionsError}
        <p class="mt-2 font-mono text-[11px]" style="color: var(--color-danger)">{chat.sessionsError}</p>
      {/if}
    </form>

    <ul class="flex-1 min-h-0 overflow-y-auto divide-y">
      {#each chat.sessions as s (s.id)}
        {@const active = s.id === chat.activeId}
        <li>
          <div
            class="flex items-stretch"
            style="background: {active ? 'color-mix(in oklab, var(--color-accent) 10%, transparent)' : 'transparent'};
                   border-left: 2px solid {active ? 'var(--color-accent)' : 'transparent'}"
          >
            <button
              type="button"
              onclick={() => pick(s.id)}
              class="min-w-0 flex-1 text-left px-5 md:px-6 py-3.5 transition-colors hover:bg-[color-mix(in_oklab,var(--color-fg-faint)_8%,transparent)]"
            >
              <div class="flex items-baseline justify-between gap-3">
                <span class="text-sm leading-snug truncate" style="color: var(--color-fg)">
                  {sessionLabel(s.id, s.title)}
                </span>
                {#if chat.sendingSession === s.id}
                  <span class="font-mono text-[10px] tabular smallcaps shrink-0 breathe"
                        style="color: var(--color-accent)">live</span>
                {/if}
              </div>
              <p class="mt-1 font-mono text-[10px] tabular" style="color: var(--color-fg-faint)">
                {s.id === DEFAULT_SESSION ? "default" : s.id} · {fmtTime(s.updated_at)}
              </p>
            </button>

            {#if s.id !== DEFAULT_SESSION}
              <div class="shrink-0 flex items-center pr-3 md:pr-4">
                {#if confirmingDelete === s.id}
                  <div class="flex items-center gap-1.5">
                    <button
                      type="button"
                      onclick={() => confirmDelete(s.id)}
                      class="px-2 py-1 font-mono text-[10px] smallcaps tabular border rounded-sm"
                      style="border-color: var(--color-danger); color: var(--color-danger)"
                    >
                      delete
                    </button>
                    <button
                      type="button"
                      onclick={() => (confirmingDelete = null)}
                      class="px-2 py-1 font-mono text-[10px] smallcaps tabular border rounded-sm"
                      style="border-color: var(--color-border-hot); color: var(--color-fg-mute)"
                    >
                      keep
                    </button>
                  </div>
                {:else}
                  <button
                    type="button"
                    onclick={() => (confirmingDelete = s.id)}
                    aria-label="delete session"
                    class="px-2 py-1 font-mono text-xs leading-none rounded-sm transition-colors hover:text-[color:var(--color-danger)]"
                    style="color: var(--color-fg-faint)"
                  >
                    ✕
                  </button>
                {/if}
              </div>
            {/if}
          </div>
        </li>
      {/each}
    </ul>
  </aside>

  <!-- conversation -->
  <div class="flex flex-col min-h-0">
    <header class="shrink-0 flex items-baseline justify-between gap-4 px-5 md:px-8 py-4 border-b">
      <div class="flex items-baseline gap-3">
        <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">ii.</p>
        <h1 class="font-display italic text-2xl md:text-3xl tracking-tight" style="color: var(--color-fg)">
          operator console
        </h1>
      </div>
      <p class="font-mono text-[10px] smallcaps tabular truncate max-w-[45%] text-right" style="color: var(--color-fg-faint)">
        {activeTitle}
      </p>
    </header>

    <div bind:this={scroller} class="flex-1 min-h-0 overflow-y-auto px-5 md:px-8 py-6">
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
          {#if showStreaming}
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

    <form onsubmit={submit} class="shrink-0 border-t px-5 md:px-8 py-4">
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
</section>
