<script lang="ts">
  import { link, router } from "svelte-spa-router";
  import active from "svelte-spa-router/active";

  const tabs = [
    { href: "/stream", label: "stream", glyph: "≋" },
    { href: "/chat", label: "chat", glyph: "❯" },
    { href: "/settings", label: "settings", glyph: "◫" },
  ];

  let isLogin = $derived(router.location === "/login");
</script>

{#if !isLogin}
  <nav
    class="md:hidden fixed bottom-0 inset-x-0 z-30 h-16 grid grid-cols-3 border-t"
    style="background: var(--color-bg-elev); padding-bottom: env(safe-area-inset-bottom);"
  >
    {#each tabs as tab (tab.href)}
      <a
        href={tab.href}
        use:link
        use:active={{ path: tab.href, className: "tab-active" }}
        class="flex flex-col items-center justify-center gap-0.5 no-underline tab-link min-h-[44px]"
      >
        <span class="font-mono text-base leading-none">{tab.glyph}</span>
        <span class="font-mono text-[10px] smallcaps">{tab.label}</span>
      </a>
    {/each}
  </nav>
{/if}

<style>
  .tab-link {
    color: var(--color-fg-faint);
    transition: color 0.15s ease;
  }
  .tab-link:hover {
    color: var(--color-fg-mute);
  }
  :global(.tab-active) {
    color: var(--color-fg) !important;
  }
  :global(.tab-active) span:first-child {
    color: var(--color-accent);
  }
</style>
