import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";

export default defineConfig({
  plugins: [svelte(), tailwindcss()],
  resolve: {
    alias: {
      $lib: fileURLToPath(new URL("./src/lib", import.meta.url)),
    },
  },
  base: "/ui/",
  build: {
    outDir: fileURLToPath(new URL("../src/conscio/static", import.meta.url)),
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
    cssCodeSplit: true,
  },
  server: {
    port: 5174,
    proxy: {
      "/ui/api": "http://127.0.0.1:8765",
      "/ui/login": "http://127.0.0.1:8765",
      "/ui/logout": "http://127.0.0.1:8765",
    },
  },
});
