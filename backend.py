from __future__ import annotations

import operator
import os
import re
import time
import streamlit as st
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


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
    output_dir: str
    final: str


# =============================================================================
# LLM helpers
# =============================================================================

_BASE_LLM = ChatGroq(
    model=MODEL_NAME,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

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


def _invoke_structured(schema, messages, retries: int = 3):
    for attempt in range(retries):
        try:
            result = llm.with_structured_output(schema).invoke(messages)
            if result is not None:
                return result
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err.lower():
                wait = (attempt + 1) * 10
                print(f"[structured] rate limited, waiting {wait}s…")
                time.sleep(wait)
                continue
            print(f"[structured] with_structured_output failed ({type(e).__name__}: {e}), trying raw JSON…")
            break

    import json
    from langchain_core.messages import HumanMessage as HM
    msgs = list(messages)
    last = msgs[-1]
    if hasattr(last, "content"):
        msgs[-1] = HM(content=last.content + "\n\nRespond with valid JSON ONLY. No markdown fences, no explanation.")

    for attempt in range(retries):
        try:
            raw = llm.invoke(msgs).content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)
            data = json.loads(raw)
            return schema(**data)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e).lower():
                wait = (attempt + 1) * 10
                print(f"[structured] rate limited, waiting {wait}s…")
                time.sleep(wait)
                continue
            print(f"[structured] raw JSON attempt {attempt + 1} failed: {e}")

    raise RuntimeError(
        f"Could not get structured output for {schema.__name__} after {retries} attempts."
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
# Research (Tavily — optional)
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
You are a senior technical writer. Create a blog outline with EXACTLY 4 sections.

RULES:
- Each section MUST have a unique, descriptive title (NOT the blog title).
- Each section needs: goal (1 sentence), 3-4 bullets, target_words (150-250).
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
      "target_words": 200,
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

    plan.tasks = plan.tasks[:4]

    for i, task in enumerate(plan.tasks):
        if task.title.strip().lower() == plan.blog_title.strip().lower():
            task.title = task.goal[:60].rstrip(".").strip() or f"Section {i + 1}"

    return {"plan": plan}


# =============================================================================
# Sections (sequential — avoids Groq rate limits)
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


def _write_section(task: Task, plan: Plan, evidence: list) -> tuple:
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

    for attempt in range(4):
        try:
            raw = llm.invoke([
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(content=prompt),
            ]).content.strip()

            section_md = _strip_think(raw)
            lines = section_md.splitlines()
            expected = f"## {task.title}"
            blog_title_lower = plan.blog_title.strip().lower()

            if not lines:
                section_md = f"{expected}\n\n(empty section)"
            else:
                first = lines[0].strip()
                if not first.startswith("#"):
                    section_md = expected + "\n\n" + section_md
                elif first.lstrip("#").strip().lower() == blog_title_lower:
                    section_md = expected + "\n\n" + "\n".join(lines[1:]).strip()
                elif first.lstrip("#").strip() != task.title:
                    section_md = expected + "\n\n" + "\n".join(lines[1:]).strip()

            return (task.id, section_md)

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e).lower():
                wait = (attempt + 1) * 15
                print(f"[worker] rate limited on '{task.title}', waiting {wait}s…")
                time.sleep(wait)
            else:
                print(f"[worker] failed on '{task.title}': {e}")
                return (task.id, f"## {task.title}\n\n*(generation failed: {e})*")

    return (task.id, f"## {task.title}\n\n*(generation failed after retries)*")


def sections_node(state: State) -> dict:
    plan = state.get("plan")
    if plan is None:
        print("[sections] plan is None — orchestrator failed")
        return {"sections": []}

    evidence = [
        EvidenceItem(**e) if isinstance(e, dict) else e
        for e in state.get("evidence", [])
    ]

    results = []
    for i, task in enumerate(plan.tasks):
        print(f"[sections] writing {i+1}/{len(plan.tasks)}: {task.title}")
        result = _write_section(task, plan, evidence)
        results.append(result)
        if i < len(plan.tasks) - 1:
            time.sleep(5)

    return {"sections": results}


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

    sections = state.get("sections") or []
    print(f"[reducer] received {len(sections)} sections")

    if not sections:
        md = f"# {blog_title}\n\n*(No sections were generated — check worker logs.)*\n"
    else:
        ordered = [s for _, s in sorted(sections, key=lambda x: x[0])]
        body = "\n\n".join(ordered).strip()
        md = f"# {blog_title}\n\n{body}\n"

    print(f"[reducer] merged_md: {len(md)} chars")

    output_dir = Path(state.get("output_dir") or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / (_safe_slug(blog_title) + ".md")
    md_path.write_text(md, encoding="utf-8")
    print(f"[reducer] wrote {md_path}  ({len(md)} chars)")

    return {
        "final":     md,
        "merged_md": md,
    }


# =============================================================================
# Graph
# =============================================================================

g = StateGraph(State)
g.add_node("router",       router_node)
g.add_node("research",     research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("sections",     sections_node)
g.add_node("reducer",      reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges(
    "router", route_next,
    {"research": "research", "orchestrator": "orchestrator"},
)
g.add_edge("research",     "orchestrator")
g.add_edge("orchestrator", "sections")
g.add_edge("sections",     "reducer")
g.add_edge("reducer",      END)

app = g.compile()