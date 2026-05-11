<script lang="ts">
  import StatusStrip from "./StatusStrip.svelte";
  import BottomTabBar from "./BottomTabBar.svelte";
  import { router } from "svelte-spa-router";

  let { children } = $props<{ children: import("svelte").Snippet }>();

  // Login is full-bleed, no shell chrome.
  let isLogin = $derived(router.location === "/login");
</script>

{#if isLogin}
  <main class="min-h-screen">
    {@render children()}
  </main>
{:else}
  <div class="min-h-screen flex flex-col">
    <StatusStrip />
    <main class="flex-1 min-h-0 pb-[64px] md:pb-0">
      {@render children()}
    </main>
    <BottomTabBar />
  </div>
{/if}
