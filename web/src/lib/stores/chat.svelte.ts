/**
 * Server-side chat persistence client with multi-session support. Sessions and
 * their messages come from the cookie-authed /ui/api/chat surface; the active
 * session drives what ChatPanel renders. Messages are cached per session so
 * switching is instant, with a background reload reconciling against the server.
 */
import { api } from "$lib/api/client";
import { chatStream } from "$lib/stores/events.svelte";

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

let sessions = $state<ChatSession[]>([]);
let activeId = $state<string>(DEFAULT_SESSION);
let messagesBySession = $state<Record<string, ChatMessage[]>>({});
// The session id of an in-flight send, or null when idle. Only one send runs at
// a time (the backend processes one episode at a time and the POST blocks).
let sending = $state<string | null>(null);
let loadError = $state<string | null>(null);
let sessionsError = $state<string | null>(null);

export async function loadSessions(): Promise<void> {
  sessionsError = null;
  try {
    const rows = await api<ChatSession[]>("/ui/api/chat/sessions");
    sessions = rows;
    // Keep the active session valid; the server guarantees the default 'main'
    // session exists, so rows is never empty.
    if (!rows.some((s) => s.id === activeId)) {
      activeId = rows[0]?.id ?? DEFAULT_SESSION;
    }
    await loadHistory(activeId);
  } catch (err) {
    sessionsError = err instanceof Error ? err.message : "failed to load sessions";
  }
}

async function refreshSessions(): Promise<void> {
  try {
    sessions = await api<ChatSession[]>("/ui/api/chat/sessions");
  } catch {
    // Non-fatal: ordering/titles refresh lazily; the send itself succeeded.
  }
}

export async function loadHistory(target: string = activeId): Promise<void> {
  loadError = null;
  try {
    const rows = await api<ChatMessage[]>(
      `/ui/api/chat/sessions/${encodeURIComponent(target)}/messages?limit=200`,
    );
    messagesBySession = { ...messagesBySession, [target]: rows };
  } catch (err) {
    loadError = err instanceof Error ? err.message : "failed to load history";
  }
}

export async function selectSession(id: string): Promise<void> {
  if (id === activeId) return;
  activeId = id;
  loadError = null;
  // Cached messages (if any) show instantly via the getter; refresh regardless.
  await loadHistory(id);
}

export async function createSession(title?: string): Promise<void> {
  sessionsError = null;
  try {
    const created = await api<ChatSession>("/ui/api/chat/sessions", {
      json: { title: title?.trim() || null },
    });
    sessions = [created, ...sessions.filter((s) => s.id !== created.id)];
    messagesBySession = { ...messagesBySession, [created.id]: [] };
    activeId = created.id;
  } catch (err) {
    sessionsError = err instanceof Error ? err.message : "failed to create session";
  }
}

export async function deleteSession(id: string): Promise<void> {
  sessionsError = null;
  try {
    await api(`/ui/api/chat/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
    sessions = sessions.filter((s) => s.id !== id);
    const { [id]: _dropped, ...rest } = messagesBySession;
    messagesBySession = rest;
    if (activeId === id) {
      activeId = sessions[0]?.id ?? DEFAULT_SESSION;
      await loadHistory(activeId);
    }
  } catch (err) {
    sessionsError = err instanceof Error ? err.message : "failed to delete session";
  }
}

export async function sendMessage(content: string): Promise<void> {
  const text = content.trim();
  if (!text || sending) return;
  const target = activeId;
  sending = target;
  // Route the token stream to this session; ChatPanel gates the live bubble on
  // it so a mid-flight session switch never shows tokens in the wrong thread.
  chatStream.begin(target);
  // optimistic user bubble
  const optimistic: ChatMessage = {
    id: -Date.now(),
    session_id: target,
    role: "user",
    content: text,
    selected_action: null,
    episode_id: null,
    created_at: Date.now() / 1000,
  };
  messagesBySession = {
    ...messagesBySession,
    [target]: [...(messagesBySession[target] ?? []), optimistic],
  };
  try {
    await api<{ user: string; agent: string; selected_action: string }>(
      `/ui/api/chat/sessions/${encodeURIComponent(target)}/messages`,
      { json: { content: text } },
    );
    // Replace optimistic with the confirmed pair from the server.
    await loadHistory(target);
    // The send bumps updated_at server-side; refresh ordering/titles.
    await refreshSessions();
  } catch (err) {
    const msg = err instanceof Error ? err.message : "send failed";
    // The server persists the user message before invoking the agent, so on
    // failure it may already be stored — reconcile with server state instead
    // of rolling back a message that will reappear on the next load.
    await loadHistory(target);
    loadError = msg;
  } finally {
    sending = null;
  }
}

export const chat = {
  get sessions() { return sessions; },
  get activeId() { return activeId; },
  get messages() { return messagesBySession[activeId] ?? []; },
  get sending() { return sending !== null; },
  get sendingSession() { return sending; },
  get loadError() { return loadError; },
  get sessionsError() { return sessionsError; },
};
