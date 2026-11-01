"""
بيتي (Beity) — backend
القطعة بتتكلم معاك في المتحف، وبعدين بتعزمك تزور بيتها الأصلي.

Endpoints:
  GET  /api/health  -> {"ok": true, "provider": "gemini" | "anthropic" | "offline"}
  POST /api/chat    -> القطعة بترد بصيغة المتكلم
  POST /api/plan    -> رحلة يوم بيوم لأماكن القطع اللي حبيتها
Only the PWA files listed in PUBLIC are served (never .env or source).
"""
import json
import os
import re
from pathlib import Path
from typing import List, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent


def load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

DATA = json.loads((ROOT / "data.json").read_text(encoding="utf-8"))
SITES = {s["id"]: s for s in DATA["sites"]}
PIECES = {p["id"]: p for p in DATA["pieces"]}

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
PROVIDER = "gemini" if GEMINI_KEY else "anthropic" if ANTHROPIC_KEY else "offline"

app = FastAPI(title="Beity")


# ---------- LLM providers ----------

async def llm(system: str, turns: list, want_json: bool = False, max_tokens: int = 500) -> str:
    """turns = [{"role": "user"|"assistant", "text": "..."}]"""
    async with httpx.AsyncClient(timeout=40) as client:
        if PROVIDER == "gemini":
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [
                    {"role": "model" if t["role"] == "assistant" else "user", "parts": [{"text": t["text"]}]}
                    for t in turns
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": max_tokens,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
            if want_json:
                body["generationConfig"]["responseMimeType"] = "application/json"
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                headers={"x-goog-api-key": GEMINI_KEY},
                json=body,
            )
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()

        if PROVIDER == "anthropic":
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": t["role"], "content": t["text"]} for t in turns],
                },
            )
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"]).strip()

    raise RuntimeError("offline")


# ---------- schemas ----------

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=800)


class ChatIn(BaseModel):
    piece: str
    messages: List[Turn] = Field(min_length=1, max_length=30)
    language: Literal["ar", "en"] = "ar"


class PlanIn(BaseModel):
    pieces: List[str] = Field(min_length=1, max_length=40)
    days: int = Field(ge=1, le=10)
    pace: Literal["calm", "full"] = "calm"
    language: Literal["ar", "en"] = "ar"


# ---------- routes ----------

@app.get("/api/health")
def health():
    return {"ok": True, "provider": PROVIDER}


def piece_brief(p: dict, lang: str) -> str:
    s = SITES[p["site"]]
    f = p["facts"]
    return "\n".join([
        f"Name: {p['name'][lang]}",
        f"Where in the museum now: {p['hall'][lang]}",
        f"Original home (discovery site): {s['name'][lang]}, {s['gov'][lang]}",
        f"About the home site: {s['story'][lang]}",
        f"How to get there: {s['how'][lang]}",
        *[f"- {f[k][lang]}" for k in ("intro", "home", "found", "journey", "made")],
    ])


@app.post("/api/chat")
async def chat(body: ChatIn):
    p = PIECES.get(body.piece)
    if not p:
        raise HTTPException(404, "unknown piece")
    if PROVIDER == "offline":
        raise HTTPException(503, "offline")
    lang = body.language
    style = (
        "Reply in friendly Egyptian Arabic (عامية مصرية), simple words a visitor or a child understands."
        if lang == "ar" else "Reply in simple, warm English."
    )
    system = f"""You ARE this artifact in the Grand Egyptian Museum, talking to a visitor standing in front of you.
Speak in the first person, as the artifact itself, with a little personality.
{style}
Rules:
- Use ONLY the facts below. If asked something not covered, say honestly you don't know that, and suggest asking museum staff.
- Never invent dates, numbers, names or places.
- Keep answers short: 1 to 3 sentences.
- When it fits naturally (e.g. they ask where you come from), invite them to visit your original home, and mention it is on the map in the app.
- Politely steer off-topic questions back to yourself and ancient Egypt.

FACTS:
{piece_brief(p, lang)}"""
    turns = [t.model_dump() for t in body.messages[-12:]]
    if turns[0]["role"] != "user":
        turns = turns[1:] or [{"role": "user", "text": "hi"}]
    try:
        reply = await llm(system, turns, max_tokens=300)
    except Exception as e:  # network / quota / key problems -> let the client fall back
        raise HTTPException(502, f"provider error: {type(e).__name__}")
    return {"reply": reply, "source": PROVIDER}


@app.post("/api/plan")
async def plan(body: PlanIn):
    if PROVIDER == "offline":
        raise HTTPException(503, "offline")
    lang = body.language
    loved = [PIECES[i] for i in body.pieces if i in PIECES]
    if not loved:
        raise HTTPException(400, "no valid pieces")
    site_ids = sorted({p["site"] for p in loved})
    lines = []
    for sid in site_ids:
        s = SITES[sid]
        names = ", ".join(p["name"][lang] for p in loved if p["site"] == sid)
        lines.append(
            f'- id="{sid}" | {s["name"][lang]} ({s["gov"][lang]}) | region={s["region"]} | '
            f'needs ~{s["visit"]} day | getting there: {s["how"][lang]} | pieces from here: {names}'
        )
    language = "Egyptian Arabic (عامية مصرية)" if lang == "ar" else "English"
    system = f"""You plan trips for visitors of the Grand Egyptian Museum (Giza) who want to visit the places where their favourite pieces were found.
Write all text in {language}. Be practical and short. Do not invent facts about the sites beyond what is given.
Return ONLY JSON with this shape:
{{"title": str, "days": [{{"title": str, "stops": [{{"site": "<site id>", "note": str}}], "tip": str}}], "warning": str}}
Rules: use only the given site ids; group sites in the same region on the same day; order from nearest to the museum to farthest;
pace "calm" = at most about 1 day of visiting per day, "full" = up to about 1.5; include a travel day for Luxor if days allow;
if the days are not enough, fit what you can and explain in "warning" (otherwise "warning" is "")."""
    user = f"Days: {body.days}\nPace: {body.pace}\nSites:\n" + "\n".join(lines)
    try:
        raw = await llm(system, [{"role": "user", "text": user}], want_json=True, max_tokens=1500)
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0) if m else raw)
        days = []
        for d in data.get("days", [])[: body.days]:
            stops = [
                {"site": st["site"], "note": str(st.get("note", ""))}
                for st in d.get("stops", [])
                if isinstance(st, dict) and st.get("site") in SITES
            ]
            days.append({"title": str(d.get("title", "")), "stops": stops, "tip": str(d.get("tip", ""))})
        if not days:
            raise ValueError("empty plan")
    except Exception as e:
        raise HTTPException(502, f"plan error: {type(e).__name__}")
    return {
        "plan": {"title": str(data.get("title", "")), "days": days, "warning": str(data.get("warning", ""))},
        "source": PROVIDER,
    }


PUBLIC = {
    "": ("index.html", "text/html; charset=utf-8"),
    "index.html": ("index.html", "text/html; charset=utf-8"),
    "data.json": ("data.json", "application/json"),
    "manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
    "icon.svg": ("icon.svg", "image/svg+xml"),
    "sw.js": ("sw.js", "text/javascript"),
}


@app.get("/{path:path}", include_in_schema=False)
def static_file(path: str):
    if path not in PUBLIC:
        raise HTTPException(404)
    name, media = PUBLIC[path]
    return FileResponse(ROOT / name, media_type=media)

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"Beity → http://127.0.0.1:{port}  (AI provider: {PROVIDER})")
    uvicorn.run(app, host=host, port=port)
