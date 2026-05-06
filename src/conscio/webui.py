from __future__ import annotations

import hmac
import secrets
import time
from typing import Any

from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from conscio.service import ConscioService


SESSION_COOKIE = "conscio_web_session"
MAX_SESSIONS = 10000
MAX_LOGIN_FAILURE_TRACKERS = 10000


class LoginRequest(BaseModel):
    password: str


class TextRequest(BaseModel):
    content: str


def _session_token(service: ConscioService) -> str:
    return secrets.token_urlsafe(32)


def _sweep_sessions(sessions: dict[str, float], now: float, max_size: int = MAX_SESSIONS) -> None:
    """Drop expired entries; if still oversized, drop earliest-expiring keys to fit cap."""
    expired = [token for token, expires in sessions.items() if expires < now]
    for token in expired:
        sessions.pop(token, None)
    if len(sessions) > max_size:
        # Sort by expiry ascending; trim from the front.
        ordered = sorted(sessions.items(), key=lambda item: item[1])
        for token, _ in ordered[: len(sessions) - max_size]:
            sessions.pop(token, None)


def _sweep_login_failures(
    failures: dict[str, list[float]], now: float, window: float = 300.0,
    max_size: int = MAX_LOGIN_FAILURE_TRACKERS,
) -> None:
    """Drop tracker buckets that no longer hold any in-window failures; cap total size."""
    cutoff = now - window
    empty: list[str] = []
    for client, times in failures.items():
        in_window = [t for t in times if t >= cutoff]
        if in_window:
            failures[client] = in_window
        else:
            empty.append(client)
    for client in empty:
        failures.pop(client, None)
    if len(failures) > max_size:
        ordered = sorted(failures.items(), key=lambda item: max(item[1]) if item[1] else 0.0)
        for client, _ in ordered[: len(failures) - max_size]:
            failures.pop(client, None)


def _require_web_auth(service: ConscioService, sessions: dict[str, float], cookie: str | None) -> None:
    if not service.config.web_password:
        raise HTTPException(status_code=500, detail="web_password is not configured")
    now = time.time()
    _sweep_sessions(sessions, now)
    expires_at = sessions.get(cookie or "")
    if not expires_at or expires_at < now:
        if cookie:
            sessions.pop(cookie, None)
        raise HTTPException(status_code=401, detail="not authenticated")


def create_web_router(service: ConscioService) -> APIRouter:
    router = APIRouter()
    sessions: dict[str, float] = {}
    login_failures: dict[str, list[float]] = {}

    @router.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui")

    @router.get("/ui/login", response_class=HTMLResponse, include_in_schema=False)
    async def login_page() -> str:
        return LOGIN_HTML

    @router.post("/ui/login", include_in_schema=False)
    async def login(req: LoginRequest, request: Request, response: Response) -> dict[str, bool]:
        if not service.config.web_password:
            raise HTTPException(status_code=500, detail="web_password is not configured")
        now = time.time()
        _sweep_login_failures(login_failures, now)
        _sweep_sessions(sessions, now)
        client = request.client.host if request.client else "unknown"
        recent = [t for t in login_failures.get(client, []) if t >= now - 300]
        if len(recent) >= 8:
            login_failures[client] = recent
            raise HTTPException(status_code=429, detail="too many login attempts")
        if not hmac.compare_digest(req.password, service.config.web_password):
            recent.append(now)
            login_failures[client] = recent
            raise HTTPException(status_code=401, detail="invalid password")
        login_failures.pop(client, None)
        token = _session_token(service)
        sessions[token] = now + (60 * 60 * 24 * 14)
        response.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            secure=service.config.web_secure_cookies,
            samesite="lax",
            max_age=60 * 60 * 24 * 14,
        )
        return {"ok": True}

    @router.post("/ui/logout", include_in_schema=False)
    async def logout(response: Response, conscio_web_session: str | None = Cookie(default=None)) -> dict[str, bool]:
        if conscio_web_session:
            sessions.pop(conscio_web_session, None)
        response.delete_cookie(SESSION_COOKIE)
        return {"ok": True}

    @router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard(conscio_web_session: str | None = Cookie(default=None)) -> HTMLResponse:
        try:
            _require_web_auth(service, sessions, conscio_web_session)
        except HTTPException:
            return HTMLResponse(LOGIN_HTML, status_code=401)
        return HTMLResponse(DASHBOARD_HTML)

    @router.get("/ui/api/snapshot", include_in_schema=False)
    async def snapshot(conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        return {
            "status": (await service.status()).__dict__,
            "goals": await service.goals.list_goals(),
            "projects": await service.list_projects(),
            "influences": await service.list_influences(),
            "episodes": await service.recent_episodes(10),
            "trace": await service.recent_trace(),
            "model_context": service.latest_model_context,
            "facts": await service.recent_facts(10),
            "skills": await service.list_skills(),
        }

    @router.post("/ui/api/message", include_in_schema=False)
    async def ui_message(req: TextRequest, conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        result = await service.submit_message(req.content)
        return {"output": result.output, "selected_action": result.selected_action}

    @router.post("/ui/api/influence/goal", include_in_schema=False)
    async def ui_goal(req: TextRequest, conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        return await service.submit_influence(req.content, kind="goal")

    @router.post("/ui/api/influence/constraint", include_in_schema=False)
    async def ui_constraint(req: TextRequest, conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        return await service.submit_influence(req.content, kind="constraint")

    @router.post("/ui/api/control/{action}", include_in_schema=False)
    async def ui_control(action: str, conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        if action == "pause":
            service.pause()
            return {"paused": True}
        if action == "resume":
            service.resume()
            return {"paused": False}
        raise HTTPException(status_code=404, detail="unknown control action")

    @router.post("/ui/api/tick", include_in_schema=False)
    async def ui_tick(conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        result = await service.run_autonomous_tick()
        if result is None:
            return {"output": "", "selected_action": "wait"}
        return {"output": result.output, "selected_action": result.selected_action}

    return router


LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Conscio Login</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #f4f2ee; color: #202124; }
    main { width: min(360px, calc(100vw - 32px)); }
    h1 { margin: 0 0 18px; font-size: 28px; font-weight: 700; letter-spacing: 0; }
    form { display: grid; gap: 12px; }
    input, button { box-sizing: border-box; width: 100%; height: 44px; border-radius: 6px; font: inherit; }
    input { border: 1px solid #b9b7b1; padding: 0 12px; background: #fff; }
    button { border: 0; background: #1f6f78; color: white; font-weight: 650; cursor: pointer; }
    .error { min-height: 20px; color: #9d2f2f; font-size: 14px; }
  </style>
</head>
<body>
  <main>
    <h1>Conscio</h1>
    <form id="login">
      <input id="password" type="password" autocomplete="current-password" placeholder="Password" autofocus>
      <button type="submit">Sign in</button>
      <div class="error" id="error"></div>
    </form>
  </main>
  <script>
    document.getElementById('login').addEventListener('submit', async (event) => {
      event.preventDefault();
      const error = document.getElementById('error');
      error.textContent = '';
      const res = await fetch('/ui/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({password: document.getElementById('password').value})
      });
      if (res.ok) location.href = '/ui';
      else error.textContent = 'Invalid password.';
    });
  </script>
</body>
</html>"""


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Conscio</title>
  <style>
    :root {
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #202124; background: #f7f6f2; letter-spacing: 0;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f7f6f2; }
    header { height: 56px; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; border-bottom: 1px solid #d9d6ce; background: #ffffff; position: sticky; top: 0; z-index: 2; }
    h1 { font-size: 18px; margin: 0; letter-spacing: 0; }
    button { border: 1px solid #8fa9ad; background: #fff; color: #153f45; border-radius: 6px; height: 34px; padding: 0 12px; font: inherit; font-weight: 650; cursor: pointer; }
    button.primary { background: #1f6f78; color: white; border-color: #1f6f78; }
    button.warn { border-color: #b66a53; color: #893f2c; }
    main { display: grid; grid-template-columns: 320px minmax(0, 1fr) 360px; gap: 18px; padding: 18px; max-width: 1500px; margin: 0 auto; }
    section { min-width: 0; }
    h2 { font-size: 13px; text-transform: uppercase; color: #5f6368; margin: 0 0 10px; letter-spacing: .06em; }
    .panel { background: #fff; border: 1px solid #d9d6ce; border-radius: 8px; padding: 14px; margin-bottom: 14px; }
    .metric { display: grid; grid-template-columns: 120px 1fr; gap: 8px; font-size: 14px; padding: 4px 0; }
    .label { color: #5f6368; }
    .value { overflow-wrap: anywhere; }
    .list { display: grid; gap: 8px; }
    .item { border: 1px solid #e1ded7; border-radius: 7px; padding: 10px; background: #fbfaf7; }
    .item-title { font-weight: 700; font-size: 14px; margin-bottom: 5px; overflow-wrap: anywhere; }
    .meta { color: #6f6b63; font-size: 12px; }
    .chat { display: grid; grid-template-rows: minmax(360px, 55vh) auto; gap: 12px; }
    .messages { overflow: auto; padding: 12px; background: #fff; border: 1px solid #d9d6ce; border-radius: 8px; }
    .message { margin-bottom: 12px; max-width: 850px; line-height: 1.45; white-space: pre-wrap; overflow-wrap: anywhere; }
    .user { color: #244d52; }
    .agent { color: #202124; }
    textarea, input { width: 100%; border: 1px solid #b9b7b1; border-radius: 6px; padding: 10px; font: inherit; background: #fff; resize: vertical; min-height: 44px; }
    .compose { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: end; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .episode-list { display: grid; gap: 10px; max-height: 360px; overflow: auto; }
    .episode { border: 1px solid #e1ded7; border-radius: 7px; padding: 10px; background: #fbfaf7; }
    .episode-head { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; margin-bottom: 6px; }
    .episode-kind { font-weight: 700; font-size: 13px; overflow-wrap: anywhere; }
    .episode-action { color: #1f6f78; font-size: 12px; font-weight: 700; }
    .episode-text { font-size: 12px; line-height: 1.4; color: #3b3d40; white-space: pre-wrap; overflow-wrap: anywhere; margin-top: 6px; }
    .episode-metrics { color: #6f6b63; font-size: 12px; margin-top: 6px; }
    pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; color: #3b3d40; max-height: 360px; overflow: auto; }
    @media (max-width: 1050px) { main { grid-template-columns: 1fr; } .chat { grid-template-rows: 420px auto; } }
  </style>
</head>
<body>
  <header>
    <h1>Conscio</h1>
    <div class="actions">
      <button onclick="control('pause')" class="warn">Pause</button>
      <button onclick="control('resume')">Resume</button>
      <button onclick="tick()" class="primary">Tick</button>
      <button onclick="logout()">Logout</button>
    </div>
  </header>
  <main>
    <section>
      <div class="panel">
        <h2>Status</h2>
        <div id="status"></div>
      </div>
      <div class="panel">
        <h2>Current Goal</h2>
        <div id="goal"></div>
      </div>
      <div class="panel">
        <h2>Projects</h2>
        <div id="projects" class="list"></div>
      </div>
    </section>
    <section class="chat">
      <div id="messages" class="messages"></div>
      <form id="chat" class="compose">
        <textarea id="chatText" rows="3" placeholder="Discuss with Conscio"></textarea>
        <button class="primary" type="submit">Send</button>
      </form>
    </section>
    <section>
      <div class="panel">
        <h2>Add Goal</h2>
        <form id="goalForm" class="compose">
          <textarea id="goalText" rows="3" placeholder="Add a goal influence"></textarea>
          <button class="primary" type="submit">Add</button>
        </form>
      </div>
      <div class="panel">
        <h2>Add Constraint</h2>
        <form id="constraintForm" class="compose">
          <textarea id="constraintText" rows="2" placeholder="Add a constraint"></textarea>
          <button type="submit">Add</button>
        </form>
      </div>
      <div class="panel">
        <h2>Influences</h2>
        <div id="influences" class="list"></div>
      </div>
      <div class="panel">
        <h2>Memory</h2>
        <div id="memory" class="list"></div>
      </div>
      <div class="panel">
        <h2>Internal Reflection</h2>
        <div id="episodes" class="episode-list"></div>
      </div>
      <div class="panel">
        <h2>Model Context</h2>
        <pre id="modelContext"></pre>
      </div>
      <div class="panel">
        <h2>Cognitive Trace</h2>
        <pre id="trace"></pre>
      </div>
    </section>
  </main>
  <script>
    const messages = document.getElementById('messages');
    const fmt = (v) => v === null || v === undefined || v === '' ? 'none' : String(v);
    async function api(path, options = {}) {
      const res = await fetch(path, Object.assign({headers: {'Content-Type': 'application/json'}}, options));
      if (res.status === 401) location.href = '/ui/login';
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }
    function metric(label, value) {
      return `<div class="metric"><div class="label">${label}</div><div class="value">${fmt(value)}</div></div>`;
    }
    function item(title, meta) {
      return `<div class="item"><div class="item-title">${escapeHtml(title)}</div><div class="meta">${escapeHtml(meta)}</div></div>`;
    }
    function episodeItem(ep) {
      const metrics = ep.metrics || {};
      const meta = `${escapeHtml(ep.source || '')} | ${escapeHtml(ep.event_type || '')} | ticks ${fmt(metrics.ticks)} | attention ${fmt(metrics.attention_selections)} | errors ${fmt(metrics.prediction_errors)}`;
      return `<div class="episode">
        <div class="episode-head">
          <div class="episode-kind">${escapeHtml(ep.input || '')}</div>
          <div class="episode-action">${escapeHtml(ep.selected_action || '')}</div>
        </div>
        <div class="episode-text">${escapeHtml(ep.output || '')}</div>
        <div class="episode-metrics">${meta}</div>
      </div>`;
    }
    function escapeHtml(text) {
      return fmt(text).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function addMessage(text, who) {
      const div = document.createElement('div');
      div.className = `message ${who}`;
      div.textContent = text;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
    }
    async function refresh() {
      const data = await api('/ui/api/snapshot');
      const s = data.status || {};
      document.getElementById('status').innerHTML =
        metric('Running', s.running) + metric('Paused', s.paused) + metric('Session', s.session_id) +
        metric('Queue', s.queue_depth) + metric('Action', s.last_autonomous_action) +
        metric('Tool actions', s.actions_last_hour) + metric('Unsafe', s.unsafe_autonomy);
      const goal = s.active_goal || {};
      document.getElementById('goal').innerHTML = item(goal.description || 'No active goal', `${goal.source || ''} ${goal.status || ''}`);
      document.getElementById('projects').innerHTML = (data.projects || []).map((p) => item(p.title, `${p.status} | ${p.id.slice(0, 12)}`)).join('') || '<div class="meta">No projects yet.</div>';
      document.getElementById('influences').innerHTML = (data.influences || []).slice(0, 8).map((i) => item(i.content, `${i.kind} | ${i.status} | ${i.appraisal}`)).join('') || '<div class="meta">No influences yet.</div>';
      const facts = (data.facts || []).slice(0, 5).map((f) => item(f.fact, `fact | ${f.confidence || ''}`));
      const skills = (data.skills || []).slice(0, 5).map((s) => item(s.skill, `skill | used ${s.use_count || 0}`));
      document.getElementById('memory').innerHTML = facts.concat(skills).join('') || '<div class="meta">No memory yet.</div>';
      document.getElementById('episodes').innerHTML = (data.episodes || []).slice(0, 8).map(episodeItem).join('') || '<div class="meta">No episodes yet.</div>';
      document.getElementById('modelContext').textContent = data.model_context || '';
      document.getElementById('trace').textContent = data.trace || '';
    }
    document.getElementById('chat').addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = document.getElementById('chatText').value.trim();
      if (!text) return;
      document.getElementById('chatText').value = '';
      addMessage(text, 'user');
      const data = await api('/ui/api/message', {method: 'POST', body: JSON.stringify({content: text})});
      addMessage(data.output || '', 'agent');
      refresh();
    });
    document.getElementById('goalForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = document.getElementById('goalText').value.trim();
      if (!text) return;
      document.getElementById('goalText').value = '';
      await api('/ui/api/influence/goal', {method: 'POST', body: JSON.stringify({content: text})});
      refresh();
    });
    document.getElementById('constraintForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = document.getElementById('constraintText').value.trim();
      if (!text) return;
      document.getElementById('constraintText').value = '';
      await api('/ui/api/influence/constraint', {method: 'POST', body: JSON.stringify({content: text})});
      refresh();
    });
    async function control(action) { await api(`/ui/api/control/${action}`, {method: 'POST'}); refresh(); }
    async function tick() { const data = await api('/ui/api/tick', {method: 'POST'}); addMessage(data.output || '(wait)', 'agent'); refresh(); }
    async function logout() { await api('/ui/logout', {method: 'POST'}); location.href = '/ui/login'; }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>"""
