#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis LangCache Demo - Gradio UI
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

try:
    from langcache import LangCache
    from langcache.models import SearchStrategy
except Exception:
    LangCache = None
    SearchStrategy = None

# ============== Env & Clients ==============
load_dotenv(override=True)

LANGCACHE_API_KEY = os.getenv("LANGCACHE_API_KEY") or os.getenv("LANGCACHE_SERVICE_KEY")
LANGCACHE_CACHE_ID = os.getenv("LANGCACHE_CACHE_ID")
LANGCACHE_BASE_URL = os.getenv("LANGCACHE_BASE_URL", "https://aws-us-east-1.langcache.redis.io")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY in environment.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Password protection
APP_PASSWORD = os.getenv("APP_PASSWORD", "secret42")

lang_cache: Optional[LangCache] = None
if LANGCACHE_API_KEY and LANGCACHE_CACHE_ID and LangCache is not None:
    lang_cache = LangCache(
        server_url=LANGCACHE_BASE_URL,
        cache_id=LANGCACHE_CACHE_ID,
        api_key=LANGCACHE_API_KEY,
    )
else:
    print("[WARNING] LangCache not configured; UI will run, but without real cache.")

# ============== Intent / Identity ==============

def is_name_prompt(p: str) -> bool:
    p = (p or "").strip().lower()
    triggers = [
        "what is my name", "what's my name", "tell me my name",
        "who am i", "my name is", "what am i called",
    ]
    return any(t in p for t in triggers)

def is_role_prompt(p: str) -> bool:
    p = (p or "").strip().lower()
    triggers = [
        "what is my role", "what's my role", "what is my function",
        "what is my position", "what is my job", "what do i do",
        "my role in the company", "tell me my role",
    ]
    return any(t in p for t in triggers)

ROLE_SET_PATTERNS = [
    r"\bmy role is\s+(?P<role>.+)$",
    r"\bmy function is\s+(?P<role>.+)$",
    r"\bi am\s+(?P<role>.+)$",
    r"\bi work as\s+(?P<role>.+)$",
]
def try_extract_role_set(p: str) -> Optional[str]:
    txt = (p or "").strip()
    txt = re.sub(r"[.!?]\s*$", "", txt)
    for pat in ROLE_SET_PATTERNS:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            role = m.group("role").strip()
            role = re.sub(r"[.!?]\s*$", "", role)
            return role
    return None

def is_identity_prompt(p: str) -> bool:
    return is_name_prompt(p) or is_role_prompt(p)

def should_personalize_name(p: str) -> bool:
    return is_name_prompt(p)

KEY_NAME = "[IDENTITY:NAME]"
KEY_ROLE = "[IDENTITY:ROLE]"

def normalize_prompt_for_cache(prompt: str) -> Tuple[str, str]:
    if is_name_prompt(prompt):
        return KEY_NAME, "identity:name"
    if is_role_prompt(prompt):
        return KEY_ROLE, "identity:role"
    return f"[FACT]\n{prompt.strip()}", "fact"

def depersonalize_safe(text: str, person: Optional[str]) -> str:
    if not text or not person:
        return text
    original = text.strip()
    t = original
    patterns = [
        rf"^hello,?\s*{re.escape(person)}\s*!?\s*",
        rf"^hi,?\s*{re.escape(person)}\s*!?\s*",
        rf"^your\s+name\s+is\s*{re.escape(person)}[.!]?\s*",
        rf"^you\s+are\s+called\s*{re.escape(person)}[.!]?\s*",
    ]
    for p in patterns:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    t = t.strip()
    return t if t else original

# ============== Context / Disambiguation ==============

AMBIGUOUS_TERMS = [
    r"\bcell\b",
    r"\bbank\b",
    r"\bnetwork\b",
    r"\bmodel\b",
    r"\bpipeline\b",
]

def infer_domain(company: str, bu: str, role: Optional[str]) -> str:
    text = f"{company} {bu} {role or ''}".lower()
    if any(k in text for k in ["health", "clinic", "medical", "hospital", "healthcare"]):
        return "healthcare"
    if any(k in text for k in ["engineering", "software", "dev", "product", "it", "technology", "tech"]):
        return "software engineering"
    if any(k in text for k in ["data", "bi", "analytics"]):
        return "data"
    if any(k in text for k in ["finance", "bank", "invest", "asset", "insurance"]):
        return "finance"
    if any(k in text for k in ["tourism", "eco", "adventure", "hotel", "travel"]):
        return "tourism"
    return "general user area"

def looks_ambiguous(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(re.search(pat, p, flags=re.IGNORECASE) for pat in AMBIGUOUS_TERMS)

def rewrite_with_domain(prompt: str, domain_label: str) -> str:
    clean = prompt.strip()
    if not clean.endswith("?"):
        clean += "?"
    return f"{clean} (in the context of {domain_label})"

# ============== LLM ==============

def call_openai(
    prompt: str,
    person: Optional[str] = None,
    company: Optional[str] = None,
    bu: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    domain = infer_domain(company or "", bu or "", role)
    system_ctx = (
        "Answer briefly and directly. "
        "Don't mention the user's name unless the question is about name/identity. "
        f"Main context: {domain}. "
        "If the question is ambiguous (e.g., 'deploy', 'pipeline', 'model'), "
        f"ANSWER ONLY in the sense of {domain} and do NOT mention other meanings."
    )

    examples = [
        # Software Engineering
        {"role": "user", "content": "What is a deploy? (in the context of software engineering)"},
        {"role": "assistant",
         "content": "Deploy is the process of making a new version of software available in production."},
        {"role": "user", "content": "What is a pipeline? (in the context of software engineering)"},
        {"role": "assistant",
         "content": "A pipeline is an automated sequence of steps to build, test, and deploy code."},

        # Finance
        {"role": "user", "content": "What is a deploy? (in the context of corporate finance)"},
        {"role": "assistant",
         "content": "In the financial context, deploy can refer to the release of a new process, system, or investment for internal use."},
        {"role": "user", "content": "What is a pipeline? (in the context of sales and finance)"},
        {"role": "assistant",
         "content": "A pipeline is the list of opportunities or revenue forecasts that are still in progress."},

        # Machine Learning examples
        {"role": "user", "content": "Explain what machine learning is."},
        {"role": "assistant",
         "content": "Machine learning is an area of AI that enables systems to learn patterns from data without explicit programming."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant",
         "content": "Machine learning is a branch of AI that enables systems to learn patterns from data and make predictions or decisions without being explicitly programmed."},
    ]
    msgs = [{"role": "system", "content": system_ctx}, *examples, {"role": "user", "content": prompt}]
    # Note: gpt-5 only supports default temperature (1.0), so we don't pass it
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
    )
    return resp.choices[0].message.content.strip()

def estimate_tokens(*texts: str) -> int:
    total_chars = sum(len(t or "") for t in texts)
    return max(1, total_chars // 4)


def calculate_savings(tokens: int, price_per_1k: float = 0.15) -> float:
    """Calculate cost savings for cached response"""
    return (tokens / 1000.0) * price_per_1k

# ============== Cache Attributes ==============

def build_attributes(company: str, bu: str, person: str, isolation: str):
    """
    Return None - cache instance doesn't have attributes configured.
    """
    return None

# ============== Core Search/Answer ==============

def search_and_answer(
    company: str,
    bu: str,
    person: str,
    prompt_original: str,
    isolation: str,
    similarity_threshold: Optional[float],
    use_exact_then_semantic: bool,
    ttl_ms: Optional[int],
) -> Tuple[str, str, str, str, int]:
    """Simplified version - basic semantic caching without attributes"""
    if not prompt_original.strip():
        return "", "—", "{}", "Idle", 0

    prompt = prompt_original.strip()

    debug = {
        "similarity_threshold": similarity_threshold,
        "strategy": "EXACT→SEMANTIC" if use_exact_then_semantic else "SEMANTIC only",
        "ttl_ms_on_set": ttl_ms,
    }

    # Try cache search
    t0 = time.perf_counter()
    cached_answer = None
    if lang_cache:
        try:
            # Search with attributes=None (no attributes configured)
            results = lang_cache.search(prompt=prompt, attributes=None)

            if results and getattr(results, "data", None):
                cached_answer = results.data[0].response
        except Exception as e:
            debug["cache_search_error"] = str(e)
    cache_latency = time.perf_counter() - t0

    # Cache hit - return cached response
    if cached_answer:
        tokens_est = estimate_tokens(prompt, cached_answer)
        latency_txt = f"<span style='color: #00C853; font-weight: bold;'>[Cache Hit]</span> search: {cache_latency:.3f}s"
        return cached_answer, "cache", json.dumps(debug, indent=2), latency_txt, tokens_est

    # Cache miss - call LLM
    t1 = time.perf_counter()
    llm_answer = call_openai(prompt)
    llm_latency = time.perf_counter() - t1

    # Store in cache
    if lang_cache:
        try:
            # Build set parameters with attributes=None
            set_params = {"prompt": prompt, "response": llm_answer, "attributes": None}
            if ttl_ms is not None:
                set_params["ttl_millis"] = ttl_ms
            lang_cache.set(**set_params)
        except Exception as e:
            debug["cache_set_error"] = str(e)

    tokens_est = estimate_tokens(prompt, llm_answer)
    latency_txt = f"<span style='color: #FF6D00; font-weight: bold;'>[Cache Miss]</span> search: {cache_latency:.3f}s, llm: {llm_latency:.3f}s"
    return llm_answer, "llm", json.dumps(debug, indent=2), latency_txt, tokens_est

# ============== FLUSH helpers ==============

def parse_deleted_count(res: Any) -> Optional[int]:
    if hasattr(res, "deleted_entries_count"):
        return getattr(res, "deleted_entries_count", None)
    if isinstance(res, dict):
        return res.get("deleted_entries_count") or res.get("deleted") or res.get("deleted_count")
    return None

def flush_entries_with_attrs(attrs: Dict[str, str]) -> Tuple[str, str]:
    if not lang_cache:
        return "⚠️ LangCache not configured; no flush executed.", json.dumps({"attributes": attrs, "ok": False}, ensure_ascii=False, indent=2)
    try:
        res = lang_cache.delete_query(attributes=attrs)
        deleted = parse_deleted_count(res)
        msg = f"✅ Flush executed. Scope={attrs}. Removed={deleted if deleted is not None else '—'}"
        debug = {"attributes": attrs, "response": getattr(res, '__dict__', res)}
        return msg, json.dumps(debug, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ Flush error: {e}", json.dumps({"attributes": attrs, "error": str(e)}, ensure_ascii=False, indent=2)

def handle_flush_scope(company: str, bu: str, person: str, isolation: str):
    attrs = build_attributes(company or "", bu or "", person or "", isolation)
    if not attrs:
        return ("⚠️ Select an isolation level other than 'none' to clear by scope.",
                json.dumps({"attributes": attrs, "error": "attributes cannot be blank"}, ensure_ascii=False, indent=2))
    return flush_entries_with_attrs(attrs)

def handle_flush_both(
    a_company: str, a_bu: str, a_person: str,
    b_company: str, b_bu: str, b_person: str,
    isolation: str,
):
    attrs_a = build_attributes(a_company or "", a_bu or "", a_person or "", isolation)
    attrs_b = build_attributes(b_company or "", b_bu or "", b_person or "", isolation)

    msgs = []
    debugs = []

    if attrs_a:
        msg_a, dbg_a = flush_entries_with_attrs(attrs_a)
        msgs.append(msg_a)
        debugs.append(json.loads(dbg_a))
    else:
        msgs.append("⚠️ Scope A: 'none' isolation cannot be cleared.")
        debugs.append({"attributes": attrs_a, "error": "attributes cannot be blank"})

    if attrs_b:
        msg_b, dbg_b = flush_entries_with_attrs(attrs_b)
        msgs.append(msg_b)
        debugs.append(json.loads(dbg_b))
    else:
        msgs.append("⚠️ Scope B: 'none' isolation cannot be cleared.")
        debugs.append({"attributes": attrs_b, "error": "attributes cannot be blank"})

    final_msg = "<br/>".join(msgs)
    return final_msg, json.dumps({"A": debugs[0], "B": debugs[1]}, ensure_ascii=False, indent=2)

# ============== UI / KPIs ==============

DESCRIPTION = """
- LangCache stores neutral LLM responses and reuses them by scope (Company/BU/Person).
- The UI below allows comparing two scenarios in parallel (A vs B).
- Ambiguous requests are automatically rewritten for the user's domain.
- You can clear cache entries by scope (A, B, or both) — the index is NEVER deleted.
"""

def format_currency(v: float, currency: str = "USD") -> str:
    return f"{currency} ${v:,.4f}"

def update_kpis(state: dict) -> Dict[str, str]:
    hits = state.get("hits", 0)
    misses = state.get("misses", 0)
    total = hits + misses
    hit_rate = (hits / total * 100) if total else 0.0
    saved_tokens = state.get("saved_tokens", 0)
    saved_usd = state.get("saved_usd", 0.0)
    return {
        "hits": f"{hits}",
        "misses": f"{misses}",
        "rate": f"{hit_rate:.1f}%",
        "tokens": f"{saved_tokens}",
        "usd": format_currency(saved_usd),
    }

def calc_savings(tokens_est: int, price_in: float, price_out: float, frac_in: float = 0.5) -> float:
    tokens_in = int(tokens_est * frac_in)
    tokens_out = max(0, tokens_est - tokens_in)
    return (tokens_in / 1000.0) * price_in + (tokens_out / 1000.0) * price_out

# ===================== CSS (VISUAL ONLY) =====================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
  --redis-red:#D82C20; --ink:#0b1220; --soft:#475569; --muted:#64748b;
  --line:#e5e7eb; --bg:#f6f7f9; --white:#ffffff; --radius:14px;
  --success:#10b981; --warning:#f59e0b;
}

* { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; }
body, #app-root { background: var(--bg); }

/* HEADER */
.app-header {
  position: sticky; top: 0; z-index: 50;
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding:14px 16px; background: var(--redis-red); color:#fff;
  box-shadow: 0 2px 8px rgba(0,0,0,.18);
}
.app-header .brand { display:flex; align-items:center; gap:14px; flex:1; }
.app-header .brand img { height:24px; display:block; }
.app-header .brand-content { display:flex; flex-direction:column; gap:4px; flex:1; }
.app-header .title {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size:20px; font-weight:700; letter-spacing:.3px;
  line-height:1.2;
}
.app-header .meta {
  display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  font-size:12px; opacity:0.95; font-weight:500;
}
.app-header .meta-item {
  display:inline-flex; align-items:center; gap:6px;
  padding:3px 8px; background:rgba(255,255,255,0.15);
  border-radius:6px; white-space:nowrap;
}
.app-header .meta-item .label { opacity:0.8; }
.app-header .meta-item .value { font-weight:600; }
.app-header .links { display:flex; gap:8px; }
.app-header .links a {
  display:inline-flex; align-items:center; gap:8px; color:#fff; text-decoration:none;
  border:1px solid rgba(255,255,255,.35); padding:7px 12px; border-radius:999px; font-weight:600; font-size:12px;
  transition: background .15s ease, transform .15s ease;
}
.app-header .links a:hover { background: rgba(255,255,255,.14); transform: translateY(-1px); }

/* Mobile responsive header */
@media (max-width: 768px) {
  .app-header { flex-direction:column; align-items:flex-start; padding:12px; }
  .app-header .brand { flex-direction:column; align-items:flex-start; gap:10px; }
  .app-header .brand img { height:20px; }
  .app-header .title { font-size:16px; }
  .app-header .meta { gap:8px; }
  .app-header .meta-item { font-size:11px; padding:2px 6px; }
  .app-header .links { width:100%; justify-content:flex-start; }
}

/* HEADINGS */
.h1 {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size:26px; font-weight:700; color:var(--ink); margin:16px 16px 6px;
}
.h2 {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size:16px; font-weight:600; color:var(--soft); margin:0 16px 14px;
}

/* Config box (clean) */
.config-card {
  margin: 10px 16px 14px; padding:12px;
  background: var(--white);
  border:1px solid var(--line); border-radius: var(--radius);
}

/* KPIs */
.kpi-row { display:flex; gap:12px; margin: 0 16px 16px; flex-wrap: wrap; }
.kpi {
  flex:1; min-width: 140px; background: var(--white); border:1px solid var(--line); border-radius:12px;
  padding:18px 20px; transition: transform .2s ease, box-shadow .2s ease;
}
.kpi:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,.08); }
.kpi .kpi-num {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size:24px; font-weight:700; color:var(--ink); line-height:1.1;
}
.kpi .kpi-value {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: 42px;
  font-weight: 800;
  color: #1a1a1a;
  line-height: 1;
  margin-bottom: 8px;
  letter-spacing: -0.02em;
}
.kpi .kpi-label {
  font-size:11px; color:var(--muted); margin-top:6px;
  text-transform:uppercase; letter-spacing:.8px; font-weight:600;
}
.kpi-accent { border-color: var(--redis-red); border-width: 2px; }
.kpi-accent .kpi-num { color: var(--redis-red); }
.kpi-accent .kpi-value { color: var(--redis-red); }

/* Cenários lado a lado */
.scenarios { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 10px 16px; }
@media (max-width: 1024px) { .scenarios { grid-template-columns: 1fr; } }

.card {
  background: var(--white); border:2px solid var(--line); border-radius: var(--radius);
  padding:16px; transition: border-color .2s ease;
}
.card:hover { border-color: var(--redis-red); }
.card .card-title {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size:18px; font-weight:700; color:var(--ink); margin-bottom:12px;
  display: flex; align-items: center; gap: 8px;
}

/* Source badges */
.source-badge {
  display: inline-block; padding: 4px 10px; border-radius: 6px;
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .5px;
}
.source-cache { background: #d1fae5; color: #065f46; }
.source-llm { background: #fef3c7; color: #92400e; }

/* History */
.dataframe { background: var(--white); border:1px solid var(--line); border-radius: var(--radius); }
.dataframe thead tr th { font-size:12px; font-weight:600; }
.dataframe tbody tr td { font-size:12px; }

/* Buttons */
button.primary, .gr-button-primary {
  background: var(--redis-red) !important; border-color: var(--redis-red) !important; color:#fff !important;
  font-weight: 600 !important; transition: all .2s ease !important;
}
button.primary:hover, .gr-button-primary:hover {
  background: #c02518 !important; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(216,44,32,.3) !important;
}

/* Secondary buttons */
.secondary-btn {
  background: var(--white) !important; border: 1px solid var(--line) !important;
  color: var(--soft) !important; font-weight: 600 !important;
}

/* --- HERO (título + subtítulo) --- */
.hero {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  margin: 16px;
  padding: 16px 18px;
}

.hero-title {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size: 26px;
  font-weight: 700;
  color: var(--ink);      /* força contraste alto */
  letter-spacing: .2px;
  margin: 0 0 8px 0;
}

.hero-sub {
  font-size: 14px;
  color: var(--soft);
  line-height: 1.6;
  margin: 0;
}

/* Se em algum tema o título estiver “preto no preto”, garante contraste: */
.h1 { color: var(--ink) !important; background: transparent !important; }
"""

# ============== APP (preserved A/B layout) ==============
custom_theme = gr.themes.Soft(primary_hue="blue")

# JavaScript to initialize and toggle theme
theme_js = """
function() {
    // Initialize theme from localStorage or default to dark
    const currentTheme = localStorage.getItem('theme') || 'dark';
    document.body.classList.toggle('dark', currentTheme === 'dark');

    // Toggle theme
    window.toggleTheme = function() {
        const isDark = document.body.classList.toggle('dark');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        return isDark ? '🌙 Dark Mode' : '☀️ Light Mode';
    };

    return currentTheme === 'dark' ? '🌙 Dark Mode' : '☀️ Light Mode';
}
"""

with gr.Blocks(title="Redis LangCache — English Demo", theme=custom_theme, css=CUSTOM_CSS, elem_id="app-root") as demo:
    st = gr.State({"hits": 0, "misses": 0, "saved_cost": 0.0})

    # Title + Subtitle
    gr.HTML("""
      <div class="hero">
        <div class="hero-title">Simple Semantic Caching with LangCache</div>
        <p class="hero-sub">
          This demo shows how LangCache caches LLM responses semantically.<br/>
          Ask the same question twice - the second time will be a cache hit!<br/>
          Try asking questions in different ways to see semantic matching in action.
        </p>
      </div>
    """)

    # Theme toggle button
    with gr.Row():
        theme_toggle_btn = gr.Button("🌙 Dark Mode", size="sm", scale=0, elem_id="theme-toggle")

    theme_toggle_btn.click(
        fn=None,
        js="""() => {
            const isDark = document.body.classList.toggle('dark');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
            return isDark ? '🌙 Dark Mode' : '☀️ Light Mode';
        }""",
        outputs=theme_toggle_btn
    )

    # Settings
    with gr.Group(elem_classes=["config-card"]):
        with gr.Row():
            threshold_global = gr.Slider(
                label="Similarity Threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.85,
                step=0.05,
                info="Higher = more strict matching"
            )
            exact_sem_global = gr.Checkbox(
                label="Use EXACT→SEMANTIC fallback",
                value=True,
                info="Try exact match first, then semantic"
            )

    # KPIs
    with gr.Row(elem_classes=["kpi-row"]):
        kpi_hits = gr.HTML("<div class='kpi'><div class='kpi-num'>0</div><div class='kpi-label'>Hits</div></div>")
        kpi_misses = gr.HTML("<div class='kpi'><div class='kpi-num'>0</div><div class='kpi-label'>Misses</div></div>")
        kpi_rate = gr.HTML("<div class='kpi'><div class='kpi-num'>0.0%</div><div class='kpi-label'>Hit Rate</div></div>")
        kpi_savings = gr.HTML("<div class='kpi kpi-accent'><div class='kpi-num'>$0.00</div><div class='kpi-label'>Cost Saved</div></div>")

    # Simple Chat Interface
    gr.Markdown("### 💬 Ask Questions")
    prompt_input = gr.Textbox(label="Question", placeholder="Ask something… (e.g., What is machine learning?)", lines=3)
    ask_btn = gr.Button("Ask", variant="primary", size="lg")
    answer_output = gr.Textbox(label="Answer", lines=6, interactive=False)
    with gr.Row():
        source_output = gr.HTML(label="Source")
        latency_output = gr.HTML(label="Latency")
    with gr.Accordion("🔍 Debug Info", open=False):
        debug_output = gr.Code(label="Debug JSON", language="json")
    with gr.Accordion("🧹 Cache Management", open=False):
        flush_btn = gr.Button("Clear All Cache", variant="secondary")
        flush_status = gr.HTML()
        with gr.Accordion("Flush Debug", open=False):
            flush_debug = gr.Code(language="json")

    # Usage Instructions
    gr.Markdown("""
    ### 📖 How to Use

    1. **Ask a question** - Type your question in the box above
    2. **First time** - Will call OpenAI (cache miss)
    3. **Ask again** - Same or similar question will hit cache!
    4. **Test semantic matching** - Try asking the question different ways

    **Test Examples:**
    - "What is machine learning?"
    - "Explain machine learning to me" ← Semantic match!
    - "Tell me about machine learning" ← Also a match!
    """)

    # ==== Events ====
    def handle_query(prompt, threshold, fallback, state_dict):
        """Handle user query"""
        answer, source, debug, latency, tokens = search_and_answer(
            company="",
            bu="",
            person="",
            prompt_original=prompt,
            isolation="none",
            similarity_threshold=threshold,
            use_exact_then_semantic=fallback,
            ttl_ms=None,
        )

        # Update metrics
        if source == "cache":
            state_dict["hits"] += 1
            savings = calculate_savings(tokens)
            state_dict["saved_cost"] += savings
        else:
            state_dict["misses"] += 1

        total = state_dict["hits"] + state_dict["misses"]
        hit_rate = (state_dict["hits"] / total * 100) if total else 0

        # Update KPI HTML with color-coded values
        kpi_h = f"<div class='kpi'><div class='kpi-value' style='color: #00C853;'>{state_dict['hits']}</div><div class='kpi-label'>Cache Hits</div></div>"
        kpi_m = f"<div class='kpi'><div class='kpi-value' style='color: #FF6D00;'>{state_dict['misses']}</div><div class='kpi-label'>Cache Misses</div></div>"
        kpi_r = f"<div class='kpi'><div class='kpi-value' style='color: #2196F3;'>{hit_rate:.1f}%</div><div class='kpi-label'>Hit Rate</div></div>"
        kpi_s = f"<div class='kpi kpi-accent'><div class='kpi-value'>${state_dict['saved_cost']:.4f}</div><div class='kpi-label'>Cost Saved</div></div>"

        # Source badge
        badge = f"<span class='source-badge source-{source}'>{'✓ CACHE HIT' if source == 'cache' else '⚡ LLM CALL'}</span>"

        return answer, badge, debug, latency, kpi_h, kpi_m, kpi_r, kpi_s, state_dict

    # Wire up events
    ask_btn.click(
        fn=handle_query,
        inputs=[prompt_input, threshold_global, exact_sem_global, st],
        outputs=[answer_output, source_output, debug_output, latency_output,
                 kpi_hits, kpi_misses, kpi_rate, kpi_savings, st],
    )

    prompt_input.submit(
        fn=handle_query,
        inputs=[prompt_input, threshold_global, exact_sem_global, st],
        outputs=[answer_output, source_output, debug_output, latency_output,
                 kpi_hits, kpi_misses, kpi_rate, kpi_savings, st],
    )

    # Clear cache handler
    def handle_clear():
        if not lang_cache:
            return "⚠️ LangCache not configured.", "{}"
        try:
            result = lang_cache.delete_query()
            msg = f"✅ Cache cleared successfully"
            return msg, json.dumps({"status": "success"}, indent=2)
        except Exception as e:
            return f"❌ Error: {e}", json.dumps({"error": str(e)}, indent=2)

    flush_btn.click(
        fn=handle_clear,
        outputs=[flush_status, flush_debug],
    )

# ============== PASSWORD PROTECTION ==============
def check_password(password):
    """Check if the provided password matches the environment variable."""
    if password == APP_PASSWORD:
        return {
            login_box: gr.update(visible=False),
            main_app: gr.update(visible=True)
        }
    else:
        return {
            login_box: gr.update(visible=True),
            main_app: gr.update(visible=False)
        }

# Wrap the demo with password protection
with gr.Blocks(title="Redis LangCache — English Demo", css=CUSTOM_CSS) as app:
    with gr.Column(visible=True, elem_id="login-container") as login_box:
        gr.HTML("""
            <div style="max-width: 400px; margin: 100px auto; padding: 40px; background: white; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <div style="text-align: center; margin-bottom: 30px;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Redis_logo.svg/2560px-Redis_logo.svg.png"
                         alt="Redis" style="height: 40px; margin-bottom: 16px;">
                    <h2 style="font-family: 'Space Grotesk', sans-serif; color: #0b1220; margin: 0;">Redis LangCache Demo</h2>
                </div>
            </div>
        """)
        with gr.Row():
            gr.HTML("<div style='flex: 1;'></div>")
            with gr.Column(scale=1, min_width=300):
                password_input = gr.Textbox(
                    label="🔒 Password",
                    type="password",
                    placeholder="Enter password...",
                    elem_id="password-input"
                )
                login_btn = gr.Button("Login", variant="primary", size="lg")
                login_status = gr.HTML("")
            gr.HTML("<div style='flex: 1;'></div>")

    with gr.Column(visible=False) as main_app:
        demo.render()

    # Handle login
    def handle_login(password):
        if password == APP_PASSWORD:
            return {
                login_box: gr.update(visible=False),
                main_app: gr.update(visible=True),
                login_status: ""
            }
        else:
            error_msg = "<div style='color: #DC2626; background: #FEE2E2; padding: 12px; border-radius: 8px; margin-top: 10px; text-align: center; font-weight: 600;'>❌ Incorrect password. Please try again.</div>"
            return {
                login_box: gr.update(visible=True),
                main_app: gr.update(visible=False),
                login_status: error_msg
            }

    def clear_error():
        """Clear error message when user starts typing"""
        return ""

    login_btn.click(
        fn=handle_login,
        inputs=[password_input],
        outputs=[login_box, main_app, login_status]
    )

    password_input.submit(
        fn=handle_login,
        inputs=[password_input],
        outputs=[login_box, main_app, login_status]
    )

    password_input.change(
        fn=clear_error,
        outputs=[login_status]
    )

    # Initialize theme on page load
    app.load(
        fn=None,
        js=theme_js,
        outputs=theme_toggle_btn
    )

if __name__ == "__main__":
    if lang_cache:
        with lang_cache:
            app.launch()
    else:
        app.launch()