<script lang="ts">
  import Router from "svelte-spa-router";
  import AppShell from "./components/AppShell.svelte";
  import { routes } from "./router";
  import { onMount } from "svelte";
  import { theme } from "./lib/stores/theme.svelte";

  // Reapply theme on mount in case the FOUC guard didn't run (e.g. SSR/dev).
  onMount(() => {
    theme.set(theme.value);
  });
</script>

<AppShell>
  <Router {routes} />
</AppShell>
