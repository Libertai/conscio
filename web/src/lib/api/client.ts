/**
 * Cookie-auth fetch wrapper for /ui/api/*. On 401 we redirect to the SPA login route.
 */
export type ApiOptions = RequestInit & { json?: unknown };

export class ApiError extends Error {
  constructor(public status: number, public body: string) {
    super(`HTTP ${status}: ${body.slice(0, 200)}`);
  }
}

export async function api<T = unknown>(path: string, opts: ApiOptions = {}): Promise<T> {
  const init: RequestInit = { credentials: "same-origin", ...opts };
  if (opts.json !== undefined) {
    init.method ||= "POST";
    init.headers = { "Content-Type": "application/json", ...(opts.headers ?? {}) };
    init.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, init);
  if (res.status === 401) {
    if (!location.hash.startsWith("#/login")) location.hash = "#/login";
    throw new ApiError(401, "unauthenticated");
  }
  if (!res.ok) throw new ApiError(res.status, await res.text());
  if (res.status === 204) return undefined as T;
  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) return (await res.json()) as T;
  return (await res.text()) as unknown as T;
}
