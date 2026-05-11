/**
 * Tiny query helper for typed GET endpoints. Each callsite owns its own
 * reactive `{ data, error, loading }` state via the runed accessor pattern.
 * Refetch on demand; no caching layer — the SSE store invalidates by
 * calling `.refetch()` on the stores that should react to a typed event.
 */
import { api } from "./client";

export type QueryHandle<T> = {
  readonly data: T | undefined;
  readonly error: string | null;
  readonly loading: boolean;
  refetch(): Promise<void>;
};

export function makeQuery<T>(path: string | (() => string)): QueryHandle<T> {
  let data = $state<T | undefined>(undefined);
  let error = $state<string | null>(null);
  let loading = $state(false);

  async function refetch() {
    loading = true;
    error = null;
    try {
      const url = typeof path === "function" ? path() : path;
      data = await api<T>(url);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  return {
    get data() { return data; },
    get error() { return error; },
    get loading() { return loading; },
    refetch,
  };
}
