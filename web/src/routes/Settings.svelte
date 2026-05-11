<script lang="ts">
  import { theme } from "$lib/stores/theme";
  import { api } from "$lib/api/client";
  import { push } from "svelte-spa-router";

  async function logout() {
    try {
      await api("/ui/logout", { method: "POST" });
    } catch (_) {}
    push("/login");
  }
</script>

<section class="max-w-xl mx-auto px-6 py-10 grid gap-8 journal-section">
  <header>
    <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
      iii. instrument settings
    </p>
    <h1 class="mt-2 font-display italic text-3xl tracking-tight" style="color: var(--color-fg)">
      calibration.
    </h1>
  </header>

  <div class="border rounded-md p-5" style="background: var(--color-bg-elev)">
    <div class="flex items-center justify-between gap-4">
      <div>
        <p class="font-mono text-xs smallcaps" style="color: var(--color-fg)">theme</p>
        <p class="text-xs mt-0.5" style="color: var(--color-fg-mute)">
          dark mode is the operator's default. the light variant survives daylight.
        </p>
      </div>
      <button
        onclick={() => theme.toggle()}
        class="h-9 px-4 rounded-md border font-mono text-[11px] smallcaps tabular transition-colors"
        style="border-color: var(--color-border-hot); color: var(--color-fg)"
      >
        {theme.value}
      </button>
    </div>
  </div>

  <div class="border rounded-md p-5" style="background: var(--color-bg-elev)">
    <div class="flex items-center justify-between gap-4">
      <div>
        <p class="font-mono text-xs smallcaps" style="color: var(--color-fg)">session</p>
        <p class="text-xs mt-0.5" style="color: var(--color-fg-mute)">
          end the operator session and clear the cookie.
        </p>
      </div>
      <button
        onclick={logout}
        class="h-9 px-4 rounded-md border font-mono text-[11px] smallcaps tabular transition-colors"
        style="border-color: var(--color-danger); color: var(--color-danger)"
      >
        log out
      </button>
    </div>
  </div>

  <p class="font-mono text-[10px] smallcaps tabular" style="color: var(--color-fg-faint)">
    phase 0 · scaffold complete · further instruments arrive in subsequent phases.
  </p>
</section>
