/**
 * Server-side chat persistence client. Loads + appends messages against the
 * default 'main' session for v1; multi-session UI in Phase 3.
 */
import { api } from "$lib/api/client";

export type ChatMessage = {
  id: number;
  session_id: string;
  role: "user" | "agent";
  content: string;
  selected_action: string | null;
  episode_id: string | null;
  created_at: number;
};

export type ChatSession = {
  id: string;
  title: string | null;
  created_at: number;
  updated_at: number;
};

const DEFAULT_SESSION = "main";

let sessionId = $state<string>(DEFAULT_SESSION);
let messages = $state<ChatMessage[]>([]);
let sending = $state(false);
let loadError = $state<string | null>(null);

export async function loadHistory(target: string = sessionId): Promise<void> {
  loadError = null;
  try {
    const rows = await api<ChatMessage[]>(`/ui/api/chat/sessions/${encodeURIComponent(target)}/messages?limit=200`);
    sessionId = target;
    messages = rows;
  } catch (err) {
    loadError = err instanceof Error ? err.message : "failed to load history";
  }
}

export async function sendMessage(content: string): Promise<void> {
  const text = content.trim();
  if (!text || sending) return;
  sending = true;
  // optimistic user bubble
  const optimistic: ChatMessage = {
    id: -Date.now(),
    session_id: sessionId,
    role: "user",
    content: text,
    selected_action: null,
    episode_id: null,
    created_at: Date.now() / 1000,
  };
  messages = [...messages, optimistic];
  try {
    const reply = await api<{ user: string; agent: string; selected_action: string }>(
      `/ui/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`,
      { json: { content: text } },
    );
    // Replace optimistic with confirmed pair from server.
    await loadHistory(sessionId);
    void reply;
  } catch (err) {
    const msg = err instanceof Error ? err.message : "send failed";
    // The server persists the user message before invoking the agent, so on
    // failure it may already be stored — reconcile with server state instead
    // of rolling back a message that will reappear on the next load.
    await loadHistory(sessionId);
    loadError = msg;
  } finally {
    sending = false;
  }
}

export const chat = {
  get sessionId() { return sessionId; },
  get messages() { return messages; },
  get sending() { return sending; },
  get loadError() { return loadError; },
  setSession(id: string) { sessionId = id; messages = []; },
};
