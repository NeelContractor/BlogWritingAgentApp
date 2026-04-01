from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# =============================================================================
# Schemas
# =============================================================================

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should learn.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words 120-550.")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# =============================================================================
# State
# =============================================================================

class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    output_dir: str
    final: str


# =============================================================================
# LLM — strip <think> blocks for reasoning models (qwen, deepseek, etc.)
# =============================================================================

_BASE_LLM = ChatOllama(model=MODEL_NAME, temperature=0.3)

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

class _CleanLLM:
    def __init__(self, base):
        self._base = base

    def invoke(self, *args, **kwargs):
        result = self._base.invoke(*args, **kwargs)
        if hasattr(result, "content"):
            result.content = _strip_think(result.content)
        return result

    def with_structured_output(self, schema):
        return self._base.with_structured_output(schema)

llm = _CleanLLM(_BASE_LLM)


# =============================================================================
# Structured-output helper with JSON fallback
# Small models often wrap JSON in markdown fences or add preamble — we handle it.
# =============================================================================

def _invoke_structured(schema, messages, retries: int = 2):
    """
    Try with_structured_output first.
    On failure, fall back to raw-text JSON extraction.
    """
    try:
        result = llm.with_structured_output(schema).invoke(messages)
        if result is not None:
            return result
    except Exception as e1:
        print(f"[structured] with_structured_output failed ({type(e1).__name__}: {e1}), trying raw JSON…")

    import json

    # Append an explicit JSON instruction to the last HumanMessage
    from langchain_core.messages import HumanMessage as HM
    msgs = list(messages)
    last = msgs[-1]
    if hasattr(last, "content"):
        msgs[-1] = HM(content=last.content + "\n\nRespond with valid JSON ONLY. No markdown fences, no explanation.")

    for attempt in range(retries):
        try:
            raw = llm.invoke(msgs).content.strip()
            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
            # Find the first { ... } block
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)
            data = json.loads(raw)
            return schema(**data)
        except Exception as e2:
            print(f"[structured] raw JSON attempt {attempt + 1} failed: {e2}")

    raise RuntimeError(
        f"Could not get structured output for {schema.__name__} after {retries} attempts. "
        "Try a larger model: OLLAMA_MODEL=mistral"
    )


# =============================================================================
# Router
# =============================================================================

ROUTER_SYSTEM = """\
You are a routing module for a blog planner.
Decide whether web research is needed.

Modes:
- closed_book (needs_research=false): timeless, well-known topics.
- hybrid      (needs_research=true):  needs fresh examples or recent tools.
- open_book   (needs_research=true):  breaking news, "latest" topics.

Respond with valid JSON only. Example:
{
  "needs_research": false,
  "mode": "closed_book",
  "reason": "This is a well-known concept.",
  "queries": [],
  "max_results_per_query": 5
}"""

def router_node(state: State) -> dict:
    decision = _invoke_structured(RouterDecision, [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
    ])
    recency = {"open_book": 7, "hybrid": 45}.get(decision.mode, 3650)
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# =============================================================================
# Research  (Tavily — optional)
# =============================================================================

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        return [
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
            for r in (results or [])
        ]
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """\
You are a research synthesizer.
Given raw web-search results, produce an EvidencePack JSON object.
Only include items with a non-empty url.
Normalize published_at to ISO YYYY-MM-DD or null.
Deduplicate by URL.
Respond with valid JSON only. Example:
{"evidence": [{"title": "Example", "url": "https://example.com", "published_at": "2024-01-01", "snippet": "...", "source": "example.com"}]}"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))
    if not raw:
        return {"evidence": []}

    try:
        pack = _invoke_structured(EvidencePack, [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=(
                f"As-of date: {state['as_of']}\n"
                f"Recency days: {state['recency_days']}\n\n"
                f"Raw results:\n{raw}"
            )),
        ])
    except Exception as e:
        print(f"[research] evidence extraction failed: {e}")
        return {"evidence": []}

    dedup = {e.url: e for e in pack.evidence if e.url}
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [
            e for e in evidence
            if (d := _iso_to_date(e.published_at)) and d >= cutoff
        ]
    return {"evidence": evidence}


# =============================================================================
# Orchestrator
# =============================================================================

ORCH_SYSTEM = """\
You are a senior technical writer. Create a blog outline with 5-7 sections.

RULES:
- Each section MUST have a unique, descriptive title (NOT the blog title).
- Each section needs: goal (1 sentence), 3-5 bullets, target_words (150-400).
- blog_kind must be exactly one of: explainer, tutorial, news_roundup, comparison, system_design

Respond with valid JSON only. Use this exact structure:
{
  "blog_title": "...",
  "audience": "developers",
  "tone": "technical but accessible",
  "blog_kind": "explainer",
  "constraints": [],
  "tasks": [
    {
      "id": 1,
      "title": "Descriptive Section Title Here",
      "goal": "One clear sentence describing what the reader will learn.",
      "bullets": ["point 1", "point 2", "point 3"],
      "target_words": 250,
      "tags": [],
      "requires_research": false,
      "requires_citations": false,
      "requires_code": false
    }
  ]
}"""

def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    evidence_summary = ""
    if evidence:
        items = [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence[:8]]
        evidence_summary = "\n\nEvidence snippets:\n" + "\n".join(
            f"- {e.get('title', '')}: {str(e.get('snippet', ''))[:120]}" for e in items
        )

    plan = _invoke_structured(Plan, [
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n"
            f"As-of: {state['as_of']}\n"
            + (f"Set blog_kind to news_roundup.\n" if forced_kind else "")
            + evidence_summary
        )),
    ])

    if forced_kind:
        plan.blog_kind = "news_roundup"

    # Sanitize: fix any task title that equals the blog title
    for i, task in enumerate(plan.tasks):
        if task.title.strip().lower() == plan.blog_title.strip().lower():
            task.title = task.goal[:60].rstrip(".").strip() or f"Section {i + 1}"

    return {"plan": plan}


# =============================================================================
# Fanout  →  parallel workers via Send
# =============================================================================

def fanout(state: State):
    assert state["plan"] is not None
    plan_dict = state["plan"].model_dump()
    evidence_list = [
        e.model_dump() if hasattr(e, "model_dump") else e
        for e in state.get("evidence", [])
    ]
    return [
        Send("worker", {
            "topic":                state["topic"],
            "mode":                 state["mode"],
            "needs_research":       state.get("needs_research", False),
            "queries":              state.get("queries", []),
            "evidence":             evidence_list,
            "plan":                 plan_dict,
            "as_of":                state["as_of"],
            "recency_days":         state["recency_days"],
            "sections":             [],
            "merged_md":            "",
            "md_with_placeholders": "",
            "image_specs":          [],
            "output_dir":           state.get("output_dir", "."),
            "final":                "",
            "task":                 task.model_dump(),
        })
        for task in state["plan"].tasks
    ]


# =============================================================================
# Worker
# =============================================================================

WORKER_SYSTEM = """\
You are a technical writer. Write ONE blog section in Markdown.

RULES:
1. The very first line MUST be: ## <section title>
2. Cover every bullet point listed by the user.
3. Write approximately the requested number of words.
4. Output ONLY the markdown section — no preamble, no metadata.
5. Use subheadings (###), code blocks (```), or lists where they genuinely help.
6. Do NOT repeat the blog title as your section heading."""

def worker_node(payload: dict) -> dict:
    task     = Task(**payload["task"])
    plan     = Plan(**payload["plan"])
    evidence = [
        EvidenceItem(**e) if isinstance(e, dict) else e
        for e in payload.get("evidence", [])
    ]

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence and task.requires_citations:
        evidence_text = "\n\nRelevant sources (cite only these URLs if needed):\n" + "\n".join(
            f"- {e.title} | {e.url}" for e in evidence[:10]
        )

    prompt = (
        f"Blog title  : {plan.blog_title}\n"
        f"Section     : {task.title}\n"
        f"Audience    : {plan.audience}\n"
        f"Tone        : {plan.tone}\n"
        f"Goal        : {task.goal}\n"
        f"Target words: {task.target_words}\n"
        f"Include code: {task.requires_code}\n"
        f"Cover these points:{bullets_text}"
        f"{evidence_text}\n\n"
        f"Begin your response with: ## {task.title}"
    )

    raw = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=prompt),
    ]).content.strip()

    section_md = _strip_think(raw)

    # ── Enforce correct H2 heading ────────────────────────────────────────────
    lines = section_md.splitlines()
    expected = f"## {task.title}"
    blog_title_lower = plan.blog_title.strip().lower()

    if not lines:
        section_md = f"{expected}\n\n(empty section)"
    else:
        first = lines[0].strip()
        if not first.startswith("#"):
            # No heading at all — prepend
            section_md = expected + "\n\n" + section_md
        elif first.lstrip("#").strip().lower() == blog_title_lower:
            # LLM used blog title — replace with section title
            section_md = expected + "\n\n" + "\n".join(lines[1:]).strip()
        elif first.lstrip("#").strip() != task.title:
            # Wrong heading — replace
            section_md = expected + "\n\n" + "\n".join(lines[1:]).strip()

    return {"sections": [(task.id, section_md)]}


# =============================================================================
# Image planning — rule-based (no LLM needed)
# Small models reliably fail at complex nested schemas; this avoids that entirely.
# =============================================================================

def _plan_images_rule_based(
    merged_md: str, topic: str, blog_kind: str
) -> tuple[str, List[dict]]:
    """
    Insert 2 image placeholders at natural break points in the article.
    Skips news_roundup blogs.
    Returns (md_with_placeholders, image_specs).
    """
    if blog_kind == "news_roundup":
        return merged_md, []

    # Find all H2 section start positions
    h2_matches = list(re.finditer(r"^## ", merged_md, re.MULTILINE))
    if len(h2_matches) < 2:
        return merged_md, []

    # Insert after section 1 and section 3 (or last section if < 3 sections)
    target_sections = [h2_matches[0], h2_matches[min(2, len(h2_matches) - 1)]]

    def _end_of_section(md: str, section_start: int) -> int:
        """Return position of the next H2 after section_start, or end of string."""
        rest = md[section_start + 3:]
        nxt = re.search(r"^## ", rest, re.MULTILINE)
        return section_start + 3 + nxt.start() if nxt else len(md)

    insert_positions = sorted(set(
        _end_of_section(merged_md, m.start()) for m in target_sections
    ))

    # Build image specs
    section_titles = [
        merged_md[m.start():].split("\n", 1)[0].lstrip("# ").strip()
        for m in target_sections
    ]
    specs = []
    for i, title in enumerate(section_titles, start=1):
        specs.append({
            "placeholder": f"[[IMAGE_{i}]]",
            "filename": f"image_{i}.png",
            "alt": f"Diagram illustrating {title}",
            "caption": f"Figure {i}: {title}",
            "prompt": (
                f"A clean, professional technical diagram for a blog post about '{topic}'. "
                f"This diagram should illustrate the concept: '{title}'. "
                f"White background, flat vector design style, clear concise labels, "
                f"no decorative elements, suitable for a technical audience."
            ),
        })

    # Insert placeholders (iterate in reverse so character offsets stay valid)
    parts = merged_md
    for pos, spec in reversed(list(zip(insert_positions, specs))):
        parts = parts[:pos] + f"\n\n{spec['placeholder']}\n\n" + parts[pos:]

    return parts, specs


# =============================================================================
# Image generation — Gemini free tier
# =============================================================================

def _gemini_generate_image_bytes(prompt: str) -> bytes:
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY env var is not set.")

    client = genai.Client(api_key=api_key)
    image_bytes: Optional[bytes] = None

    for model_id in [
        "gemini-2.5-flash"
    ]:
        try:
            if "imagen" in model_id:
                resp = client.models.generate_images(
                    model=model_id,
                    prompt=prompt,
                    config=gtypes.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="16:9",
                        safety_filter_level="block_only_high",
                    ),
                )
                for img in (getattr(resp, "generated_images", []) or []):
                    data = getattr(getattr(img, "image", None), "image_bytes", None)
                    if data:
                        image_bytes = data
                        break
            else:
                resp = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=gtypes.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                    ),
                )
                # New SDK layout
                for candidate in (getattr(resp, "candidates", []) or []):
                    for part in (getattr(getattr(candidate, "content", None), "parts", []) or []):
                        data = getattr(getattr(part, "inline_data", None), "data", None)
                        if data:
                            image_bytes = data
                            break
                    if image_bytes:
                        break
                # Old SDK layout fallback
                if not image_bytes:
                    for part in (getattr(resp, "parts", []) or []):
                        data = getattr(getattr(part, "inline_data", None), "data", None)
                        if data:
                            image_bytes = data
                            break

            if image_bytes:
                print(f"[images] generated via {model_id}")
                break

        except Exception as e:
            print(f"[images] {model_id} failed: {e}")
            continue

    if not image_bytes:
        raise RuntimeError(
            "All Gemini image models failed. Check GOOGLE_API_KEY and API quotas."
        )

    if isinstance(image_bytes, str):
        import base64
        return base64.b64decode(image_bytes)
    return image_bytes


# =============================================================================
# Reducer
# =============================================================================

def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def reducer_node(state: State) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("reducer_node: plan is None — orchestrator may have failed.")

    blog_title = plan.blog_title if hasattr(plan, "blog_title") else plan["blog_title"]
    blog_kind  = plan.blog_kind  if hasattr(plan, "blog_kind")  else plan.get("blog_kind", "explainer")

    sections = state.get("sections") or []
    print(f"[reducer] received {len(sections)} sections")

    if not sections:
        merged_md = f"# {blog_title}\n\n*(No sections were generated — check worker logs.)*\n"
    else:
        ordered = [md for _, md in sorted(sections, key=lambda x: x[0])]
        body = "\n\n".join(ordered).strip()
        merged_md = f"# {blog_title}\n\n{body}\n"

    print(f"[reducer] merged_md length: {len(merged_md)} chars")

    # ── Image planning (rule-based — no LLM call) ─────────────────────────────
    image_specs: List[dict] = []
    md = merged_md
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")

    should_generate_images = (
        blog_kind != "news_roundup"
        and bool(sections)
        and bool(google_api_key)
    )

    if should_generate_images:
        md, image_specs = _plan_images_rule_based(merged_md, state["topic"], blog_kind)
        print(f"[reducer] image_specs planned: {len(image_specs)}")
    else:
        if not google_api_key:
            print("[reducer] GOOGLE_API_KEY not set — skipping image generation")

    # ── Generate & embed images (hard cap: 2 for free tier) ───────────────────
    output_dir = Path(state.get("output_dir") or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    active_specs = image_specs[:2]  # never exceed 2 on free tier

    if active_specs:
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for spec in active_specs:
            placeholder  = spec["placeholder"]
            img_filename = spec["filename"]
            out_path     = images_dir / img_filename

            if not out_path.exists():
                try:
                    print(f"[images] generating {img_filename} …")
                    img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                    out_path.write_bytes(img_bytes)
                    print(f"[images] saved {out_path}")
                except Exception as exc:
                    print(f"[images] FAILED {img_filename}: {exc}")
                    fallback = (
                        f"\n> **📊 {spec.get('alt', img_filename)}**  \n"
                        f"> _{spec.get('caption', '')}_  \n"
                        f"> *(Image generation unavailable: {exc})*\n"
                    )
                    md = md.replace(placeholder, fallback)
                    continue
            else:
                print(f"[images] {img_filename} already exists, reusing")

            img_md = f"![{spec['alt']}](images/{img_filename})\n*{spec['caption']}*"
            md = md.replace(placeholder, img_md)

    # Clean any leftover placeholders (e.g. if image_specs had 3 but we capped at 2)
    md = re.sub(r"\[\[IMAGE_\d+\]\]", "", md)

    # ── Write output markdown ─────────────────────────────────────────────────
    md_path = output_dir / (_safe_slug(blog_title) + ".md")
    md_path.write_text(md, encoding="utf-8")
    print(f"[reducer] wrote {md_path}  ({len(md)} chars)")

    return {
        "final":                md,
        "merged_md":            merged_md,
        "md_with_placeholders": md,
        "image_specs":          active_specs,
    }


# =============================================================================
# Graph
# =============================================================================

g = StateGraph(State)
g.add_node("router",       router_node)
g.add_node("research",     research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker",       worker_node)
g.add_node("reducer",      reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges(
    "router", route_next,
    {"research": "research", "orchestrator": "orchestrator"},
)
g.add_edge("research",     "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker",       "reducer")
g.add_edge("reducer",      END)

app = g.compile()