"""
بيتي (Beity) — backend
القطعة بتتكلم معاك في المتحف، وبعدين بتعزمك تزور بيتها الأصلي.

Public API
  GET  /api/health            provider, storage, counts
  GET  /api/data              catalogue (sites, pieces, regions)
  POST /api/chat              the piece answers in the first person
  POST /api/plan              day-by-day trip to the pieces' homes
  POST /api/identify          photo of a piece -> which piece it is (vision AI)
  GET  /api/qr/{piece}.svg    QR code that opens the piece's chat

Admin API (Bearer token from /api/admin/login, password = ADMIN_PASSWORD)
  PUT/DELETE /api/admin/pieces/{id}
  PUT/DELETE /api/admin/sites/{id}
  PUT/DELETE /api/admin/regions/{id}

Persistence: data.json on disk. If GITHUB_TOKEN + GITHUB_REPO are set, every admin
save is also committed to the repo (with [skip render]) and loaded from there at
startup, so edits survive restarts on hosts with an ephemeral disk (e.g. Render).
"""
import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Literal, Optional

import httpx
import segno
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data.json"


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

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
PROVIDER = "gemini" if GEMINI_KEY else "anthropic" if ANTHROPIC_KEY else "offline"

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
SECRET = hashlib.sha256((ADMIN_PASSWORD + os.getenv("SECRET_KEY", "beity")).encode()).digest()
GH_TOKEN = os.getenv("GITHUB_TOKEN", "")
GH_REPO = os.getenv("GITHUB_REPO", "")
GH_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GH_ON = bool(GH_TOKEN and GH_REPO)

app = FastAPI(title="Beity")

SLUG = r"^[a-z0-9][a-z0-9-]{0,39}$"


# ---------------------------------------------------------------- models

class Txt(BaseModel):
    ar: str = Field(min_length=1, max_length=2000)
    en: str = Field(default="", max_length=2000)


class Facts(BaseModel):
    intro: Txt
    home: Txt
    found: Txt
    journey: Txt
    made: Txt


class Region(BaseModel):
    ar: str = Field(min_length=1, max_length=80)
    en: str = Field(default="", max_length=80)


class Site(BaseModel):
    id: str = Field(pattern=SLUG)
    lat: float = Field(ge=20, le=33)
    lon: float = Field(ge=23, le=38)
    region: str = Field(pattern=SLUG)
    visit: float = Field(ge=0.25, le=3)
    name: Txt
    gov: Txt
    story: Txt
    how: Txt


class Piece(BaseModel):
    id: str = Field(pattern=SLUG)
    code: str = Field(pattern=r"^[A-Za-z0-9-]{1,12}$")
    site: str = Field(pattern=SLUG)
    name: Txt
    hall: Txt
    facts: Facts
    sources: List[str] = Field(default_factory=list, max_length=6)

    @field_validator("sources")
    @classmethod
    def urls(cls, v):
        for u in v:
            if not re.match(r"^https?://\S{3,500}$", u):
                raise ValueError(f"bad url: {u}")
        return v


class Catalogue(BaseModel):
    gem: dict
    regions: Dict[str, Region]
    sites: List[Site]
    pieces: List[Piece]


def check_refs(c: Catalogue):
    site_ids = [s.id for s in c.sites]
    if len(set(site_ids)) != len(site_ids):
        raise HTTPException(400, "duplicate site id")
    ids = [p.id for p in c.pieces]
    codes = [p.code for p in c.pieces]
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "duplicate piece id")
    if len(set(codes)) != len(codes):
        raise HTTPException(400, "رقم القطعة مستخدم قبل كده / code already used")
    for s in c.sites:
        if s.region not in c.regions:
            raise HTTPException(400, f"unknown region {s.region}")
    for p in c.pieces:
        if p.site not in site_ids:
            raise HTTPException(400, f"unknown site {p.site}")


# ---------------------------------------------------------------- storage

class Store:
    def __init__(self):
        self.data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        self.sha: Optional[str] = None
        self.lock = asyncio.Lock()
        self.last_error = ""

    def gh_headers(self):
        return {"Authorization": f"Bearer {GH_TOKEN}", "Accept": "application/vnd.github+json"}

    async def init(self):
        if not GH_ON:
            return
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.get(
                    f"https://api.github.com/repos/{GH_REPO}/contents/data.json",
                    params={"ref": GH_BRANCH}, headers=self.gh_headers(),
                )
                r.raise_for_status()
                j = r.json()
                remote = json.loads(base64.b64decode(j["content"]).decode("utf-8"))
                Catalogue.model_validate(remote)
                self.data, self.sha = remote, j["sha"]
                DATA_FILE.write_text(json.dumps(remote, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.last_error = f"github load failed: {type(e).__name__}"

    async def save(self, new: dict, message: str):
        cat = Catalogue.model_validate(new)
        check_refs(cat)
        clean = cat.model_dump()
        async with self.lock:
            DATA_FILE.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
            self.data = clean
            if GH_ON:
                body = json.dumps(clean, ensure_ascii=False, indent=2).encode("utf-8")
                for attempt in range(2):
                    async with httpx.AsyncClient(timeout=20) as c:
                        if not self.sha or attempt:
                            g = await c.get(f"https://api.github.com/repos/{GH_REPO}/contents/data.json",
                                            params={"ref": GH_BRANCH}, headers=self.gh_headers())
                            if g.status_code == 200:
                                self.sha = g.json()["sha"]
                        r = await c.put(
                            f"https://api.github.com/repos/{GH_REPO}/contents/data.json",
                            headers=self.gh_headers(),
                            json={
                                "message": f"{message} [skip render]",
                                "content": base64.b64encode(body).decode(),
                                "sha": self.sha, "branch": GH_BRANCH,
                            },
                        )
                    if r.status_code in (200, 201):
                        self.sha = r.json()["content"]["sha"]
                        self.last_error = ""
                        return
                    if r.status_code != 409:
                        break
                self.last_error = f"github save failed ({r.status_code})"
                raise HTTPException(502, "اتحفظ على السيرفر بس ما اتحفظش على GitHub. راجع GITHUB_TOKEN.")


STORE = Store()


@app.on_event("startup")
async def _startup():
    await STORE.init()


def D():
    return STORE.data


def site_of(sid):
    return next((s for s in D()["sites"] if s["id"] == sid), None)


def piece_of(pid):
    return next((p for p in D()["pieces"] if p["id"] == pid), None)


def L(o, lang):
    return (o or {}).get(lang) or (o or {}).get("ar") or ""


# ---------------------------------------------------------------- rate limit (protects the API key)

HITS: Dict[str, deque] = defaultdict(deque)


def limit(request: Request, per_min: int = 20):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    q, now = HITS[ip + request.url.path], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= per_min:
        raise HTTPException(429, "requests are too fast, wait a minute")
    q.append(now)


# ---------------------------------------------------------------- LLM providers

async def llm(system: str, turns: list, want_json=False, max_tokens=500, image: Optional[tuple] = None) -> str:
    """turns=[{"role":"user"|"assistant","text":...}]; image=(mime, bytes) attached to the last user turn."""
    async with httpx.AsyncClient(timeout=45) as client:
        if PROVIDER == "gemini":
            contents = []
            for i, t in enumerate(turns):
                parts = [{"text": t["text"]}]
                if image and i == len(turns) - 1:
                    parts.insert(0, {"inline_data": {"mime_type": image[0], "data": base64.b64encode(image[1]).decode()}})
                contents.append({"role": "model" if t["role"] == "assistant" else "user", "parts": parts})
            cfg = {"temperature": 0.6, "maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingBudget": 0}}
            if want_json:
                cfg["responseMimeType"] = "application/json"
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                headers={"x-goog-api-key": GEMINI_KEY},
                json={"systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": cfg},
            )
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        if PROVIDER == "anthropic":
            msgs = []
            for i, t in enumerate(turns):
                content = [{"type": "text", "text": t["text"]}]
                if image and i == len(turns) - 1:
                    content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": image[0],
                                                                   "data": base64.b64encode(image[1]).decode()}})
                msgs.append({"role": t["role"], "content": content})
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
                json={"model": ANTHROPIC_MODEL, "max_tokens": max_tokens, "system": system, "messages": msgs},
            )
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"]).strip()
    raise RuntimeError("offline")


def parse_json(raw: str):
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0) if m else raw)


# ---------------------------------------------------------------- public routes

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "provider": PROVIDER,
        "vision": PROVIDER != "offline",
        "storage": "github" if GH_ON else "local",
        "storage_error": STORE.last_error,
        "admin": bool(ADMIN_PASSWORD),
        "pieces": len(D()["pieces"]),
        "sites": len(D()["sites"]),
    }


@app.get("/api/data")
def data():
    return JSONResponse(D(), headers={"Cache-Control": "no-cache"})


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=800)


class ChatIn(BaseModel):
    piece: str
    messages: List[Turn] = Field(min_length=1, max_length=30)
    language: Literal["ar", "en"] = "ar"


class PlanIn(BaseModel):
    pieces: List[str] = Field(min_length=1, max_length=60)
    days: int = Field(ge=1, le=10)
    pace: Literal["calm", "full"] = "calm"
    language: Literal["ar", "en"] = "ar"


def piece_brief(p: dict, lang: str) -> str:
    s = site_of(p["site"])
    f = p["facts"]
    return "\n".join([
        f"Name: {L(p['name'], lang)}",
        f"Where in the museum now: {L(p['hall'], lang)}",
        f"Original home (discovery site): {L(s['name'], lang)}, {L(s['gov'], lang)}",
        f"About the home site: {L(s['story'], lang)}",
        f"How to get there: {L(s['how'], lang)}",
        *[f"- {L(f[k], lang)}" for k in ("intro", "home", "found", "journey", "made")],
    ])


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    p = piece_of(body.piece)
    if not p:
        raise HTTPException(404, "unknown piece")
    if PROVIDER == "offline":
        raise HTTPException(503, "offline")
    limit(request, 25)
    lang = body.language
    style = ("Reply in friendly Egyptian Arabic (عامية مصرية), simple words a visitor or a child understands."
             if lang == "ar" else "Reply in simple, warm English.")
    system = f"""You ARE this artifact in the Grand Egyptian Museum, talking to a visitor standing in front of you.
Speak in the first person, as the artifact itself, with a little personality. {style}
Rules:
- Use ONLY the facts below. If asked something not covered, say honestly you don't know that, and suggest asking museum staff.
- Never invent dates, numbers, names or places.
- Keep answers short: 1 to 3 sentences. Plain text, no markdown (it may be read aloud).
- When it fits naturally (e.g. they ask where you come from), invite them to visit your original home; it is on the app's map.
- Politely steer off-topic questions back to yourself and ancient Egypt.

FACTS:
{piece_brief(p, lang)}"""
    turns = [t.model_dump() for t in body.messages[-12:]]
    while turns and turns[0]["role"] != "user":
        turns = turns[1:]
    if not turns:
        raise HTTPException(400, "no user message")
    try:
        reply = await llm(system, turns, max_tokens=300)
    except Exception as e:
        raise HTTPException(502, f"provider error: {type(e).__name__}")
    return {"reply": reply, "source": PROVIDER}


@app.post("/api/plan")
async def plan(body: PlanIn, request: Request):
    if PROVIDER == "offline":
        raise HTTPException(503, "offline")
    limit(request, 10)
    lang = body.language
    loved = [p for p in (piece_of(i) for i in body.pieces) if p]
    if not loved:
        raise HTTPException(400, "no valid pieces")
    lines = []
    for sid in sorted({p["site"] for p in loved}):
        s = site_of(sid)
        names = ", ".join(L(p["name"], lang) for p in loved if p["site"] == sid)
        lines.append(f'- id="{sid}" | {L(s["name"], lang)} ({L(s["gov"], lang)}) | region={s["region"]} | '
                     f'needs ~{s["visit"]} day | getting there: {L(s["how"], lang)} | pieces from here: {names}')
    language = "Egyptian Arabic (عامية مصرية)" if lang == "ar" else "English"
    system = f"""You plan trips for visitors of the Grand Egyptian Museum (Giza) who want to visit the places where their favourite pieces were found.
Write all text in {language}. Be practical and short. Do not invent facts about the sites beyond what is given.
Return ONLY JSON: {{"title": str, "days": [{{"title": str, "stops": [{{"site": "<site id>", "note": str}}], "tip": str}}], "warning": str}}
Rules: only the given site ids; same-region sites on the same day; nearest to the museum first;
pace "calm" = up to ~1 day of visiting per day, "full" = up to ~1.5; include a travel day for far regions (e.g. Luxor, Aswan) if days allow;
if days are not enough, fit what you can and explain in "warning" (otherwise "")."""
    user = f"Days: {body.days}\nPace: {body.pace}\nSites:\n" + "\n".join(lines)
    try:
        d = parse_json(await llm(system, [{"role": "user", "text": user}], want_json=True, max_tokens=1500))
        days = []
        for day in d.get("days", [])[: body.days]:
            stops = [{"site": st["site"], "note": str(st.get("note", ""))}
                     for st in day.get("stops", []) if isinstance(st, dict) and site_of(st.get("site"))]
            days.append({"title": str(day.get("title", "")), "stops": stops, "tip": str(day.get("tip", ""))})
        if not days:
            raise ValueError("empty")
    except Exception as e:
        raise HTTPException(502, f"plan error: {type(e).__name__}")
    return {"plan": {"title": str(d.get("title", "")), "days": days, "warning": str(d.get("warning", ""))}, "source": PROVIDER}


@app.post("/api/identify")
async def identify(request: Request, image: UploadFile = File(...), language: str = Form("ar")):
    if PROVIDER == "offline":
        raise HTTPException(503, "offline")
    limit(request, 8)
    mime = image.content_type or ""
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "jpeg/png/webp only")
    raw = await image.read(6 * 1024 * 1024 + 1)
    if len(raw) > 6 * 1024 * 1024:
        raise HTTPException(413, "image too large")
    catalogue = "\n".join(
        f'- id="{p["id"]}" | {p["name"]["en"] or p["name"]["ar"]} / {p["name"]["ar"]} | shown in: {p["hall"]["en"] or p["hall"]["ar"]}'
        for p in D()["pieces"])
    system = """You identify ancient Egyptian artifacts in visitor photos taken inside the Grand Egyptian Museum.
Only choose from the catalogue. If the photo does not clearly show one of them, answer "none".
Return ONLY JSON: {"id": "<catalogue id or none>", "confidence": <0..1>, "candidates": ["<id>", ...up to 3]}"""
    try:
        d = parse_json(await llm(system, [{"role": "user", "text": "CATALOGUE:\n" + catalogue + "\n\nWhich piece is in this photo?"}],
                                 want_json=True, max_tokens=200, image=(mime, raw)))
    except Exception as e:
        raise HTTPException(502, f"vision error: {type(e).__name__}")
    pid = d.get("id") if piece_of(d.get("id")) else None
    cands = [c for c in d.get("candidates", []) if piece_of(c) and c != pid][:3]
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"id": pid, "confidence": conf, "candidates": cands}


def base_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{proto}://{host}"


@app.get("/api/qr/{pid}.svg")
def qr(pid: str, request: Request):
    if not piece_of(pid):
        raise HTTPException(404)
    q = segno.make(f"{base_url(request)}/#{pid}", error="m")
    buf = io.BytesIO()
    q.save(buf, kind="svg", scale=8, border=2, dark="#14110e", light="#ffffff", xmldecl=False)
    return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------- admin

def make_token(hours=12):
    exp = str(int(time.time()) + hours * 3600)
    return exp + "." + hmac.new(SECRET, exp.encode(), hashlib.sha256).hexdigest()


def need_admin(authorization: str = Header(default="")):
    if not ADMIN_PASSWORD:
        raise HTTPException(403, "admin disabled: set ADMIN_PASSWORD")
    tok = authorization.removeprefix("Bearer ").strip()
    exp, _, sig = tok.partition(".")
    good = hmac.new(SECRET, exp.encode(), hashlib.sha256).hexdigest()
    if not (exp.isdigit() and hmac.compare_digest(sig, good) and int(exp) > time.time()):
        raise HTTPException(401, "login required")
    return True


class LoginIn(BaseModel):
    password: str = Field(max_length=200)


@app.post("/api/admin/login")
def login(body: LoginIn, request: Request):
    if not ADMIN_PASSWORD:
        raise HTTPException(403, "admin disabled: set ADMIN_PASSWORD")
    limit(request, 6)
    if not hmac.compare_digest(body.password.encode(), ADMIN_PASSWORD.encode()):
        raise HTTPException(401, "wrong password")
    return {"token": make_token()}


@app.get("/api/admin/check")
def check(_=Depends(need_admin)):
    return {"ok": True, "storage": "github" if GH_ON else "local", "storage_error": STORE.last_error}


def copy_data():
    return json.loads(json.dumps(D()))


@app.put("/api/admin/pieces/{pid}")
async def put_piece(pid: str, piece: Piece, _=Depends(need_admin)):
    if piece.id != pid:
        raise HTTPException(400, "id mismatch")
    d = copy_data()
    d["pieces"] = [p for p in d["pieces"] if p["id"] != pid] + [piece.model_dump()]
    d["pieces"].sort(key=lambda p: p["code"])
    await STORE.save(d, f"admin: save piece {pid}")
    return piece


@app.delete("/api/admin/pieces/{pid}")
async def del_piece(pid: str, _=Depends(need_admin)):
    d = copy_data()
    if not any(p["id"] == pid for p in d["pieces"]):
        raise HTTPException(404)
    d["pieces"] = [p for p in d["pieces"] if p["id"] != pid]
    await STORE.save(d, f"admin: delete piece {pid}")
    return {"ok": True}


@app.put("/api/admin/sites/{sid}")
async def put_site(sid: str, site: Site, _=Depends(need_admin)):
    if site.id != sid:
        raise HTTPException(400, "id mismatch")
    d = copy_data()
    d["sites"] = [s for s in d["sites"] if s["id"] != sid] + [site.model_dump()]
    await STORE.save(d, f"admin: save site {sid}")
    return site


@app.delete("/api/admin/sites/{sid}")
async def del_site(sid: str, _=Depends(need_admin)):
    d = copy_data()
    if any(p["site"] == sid for p in d["pieces"]):
        raise HTTPException(400, "الموقع ده عليه قطع. انقل القطع أو امسحها الأول.")
    d["sites"] = [s for s in d["sites"] if s["id"] != sid]
    await STORE.save(d, f"admin: delete site {sid}")
    return {"ok": True}


@app.put("/api/admin/regions/{rid}")
async def put_region(rid: str, region: Region, _=Depends(need_admin)):
    if not re.match(SLUG, rid):
        raise HTTPException(400, "bad id")
    d = copy_data()
    d["regions"][rid] = region.model_dump()
    await STORE.save(d, f"admin: save region {rid}")
    return region


@app.delete("/api/admin/regions/{rid}")
async def del_region(rid: str, _=Depends(need_admin)):
    d = copy_data()
    if any(s["region"] == rid for s in d["sites"]):
        raise HTTPException(400, "المنطقة دي عليها مواقع.")
    d["regions"].pop(rid, None)
    await STORE.save(d, f"admin: delete region {rid}")
    return {"ok": True}


# ---------------------------------------------------------------- static files (whitelist only)

PUBLIC = {
    "": ("index.html", "text/html; charset=utf-8"),
    "index.html": ("index.html", "text/html; charset=utf-8"),
    "admin": ("admin.html", "text/html; charset=utf-8"),
    "admin.html": ("admin.html", "text/html; charset=utf-8"),
    "manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
    "icon.svg": ("icon.svg", "image/svg+xml"),
    "sw.js": ("sw.js", "text/javascript"),
}


@app.get("/data.json", include_in_schema=False)
def data_alias():
    return data()


@app.get("/{path:path}", include_in_schema=False)
def static_file(path: str):
    if path not in PUBLIC:
        raise HTTPException(404)
    name, media = PUBLIC[path]
    return FileResponse(ROOT / name, media_type=media, headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"Beity → http://127.0.0.1:{port}  (AI: {PROVIDER} · storage: {'github' if GH_ON else 'local'} · admin: {'on' if ADMIN_PASSWORD else 'off'})")
    uvicorn.run(app, host=host, port=port)
