<script lang="ts">
  import { push } from "svelte-spa-router";
  import { api, ApiError } from "$lib/api/client";

  let password = $state("");
  let error = $state("");
  let pending = $state(false);

  async function submit(e: Event) {
    e.preventDefault();
    error = "";
    pending = true;
    try {
      await api("/ui/login", { json: { password } });
      push("/stream");
    } catch (err) {
      if (err instanceof ApiError) {
        error = err.status === 429 ? "too many attempts. wait 5 min." : "invalid password.";
      } else {
        error = "network error.";
      }
    } finally {
      pending = false;
    }
  }
</script>

<div class="min-h-screen grid place-items-center px-6">
  <div class="w-full max-w-sm">
    <div class="mb-10">
      <h1 class="font-display italic text-5xl leading-none tracking-tight"
          style="color: var(--color-fg)">conscio</h1>
      <p class="mt-2 font-mono text-[11px] smallcaps tabular" style="color: var(--color-fg-faint)">
        observatory · single operator
      </p>
    </div>

    <form onsubmit={submit} class="grid gap-4">
      <label class="grid gap-1.5">
        <span class="font-mono text-[10px] smallcaps" style="color: var(--color-fg-mute)">passphrase</span>
        <input
          type="password"
          autocomplete="current-password"
          bind:value={password}
          disabled={pending}
          class="h-11 px-3 rounded-md border bg-transparent outline-none transition-colors
                 focus:border-[color:var(--color-accent)]"
          style="background: var(--color-bg-sunk); color: var(--color-fg)"
        />
      </label>
      <button
        type="submit"
        disabled={pending || !password}
        class="h-11 rounded-md font-mono text-xs smallcaps tracking-wider transition-opacity disabled:opacity-50"
        style="background: var(--color-accent); color: var(--color-accent-fg)"
      >
        {pending ? "verifying…" : "enter"}
      </button>
      {#if error}
        <p class="font-mono text-xs" style="color: var(--color-danger)">{error}</p>
      {/if}
    </form>

    <p class="mt-12 font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
      you are about to observe an autonomous mind in motion.
    </p>
  </div>
</div>
