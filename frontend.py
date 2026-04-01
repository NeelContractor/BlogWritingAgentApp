from __future__ import annotations

import base64
import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from backend import app

# =============================================================================
# Utilities
# =============================================================================

def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def get_output_dir() -> Path:
    if "output_dir" not in st.session_state:
        base = Path(__file__).parent / "blog_outputs"
        base.mkdir(parents=True, exist_ok=True)
        st.session_state["output_dir"] = str(base)
    return Path(st.session_state["output_dir"])


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(images_dir.parent)))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(images_dir.parent)))
    return buf.getvalue()


# =============================================================================
# Graph runner  — streams once, accumulates state, never calls invoke() again
# =============================================================================

def run_graph(graph_app, inputs: Dict[str, Any]):
    """
    Yields:
      ("node",  node_name, update_dict)
      ("final", None,      full_state_dict)
      ("error", None,      {"error": str})
    """
    accumulated: Dict[str, Any] = dict(inputs)
    try:
        for chunk in graph_app.stream(inputs, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue
            for node_name, update in chunk.items():
                if isinstance(update, dict):
                    # operator.add keys: merge lists instead of replacing
                    if "sections" in update and isinstance(update["sections"], list):
                        existing = accumulated.get("sections") or []
                        accumulated["sections"] = existing + update["sections"]
                        update_without_sections = {k: v for k, v in update.items() if k != "sections"}
                        accumulated.update(update_without_sections)
                    else:
                        accumulated.update(update)
                yield ("node", node_name, update if isinstance(update, dict) else {})
        yield ("final", None, accumulated)
    except Exception as stream_err:
        # Hard fallback: plain invoke
        yield ("node", "⚠️ streaming failed, invoking directly…", {})
        try:
            result = graph_app.invoke(inputs)
            yield ("final", None, result)
        except Exception as inv_err:
            yield ("error", None, {"error": f"stream: {stream_err} | invoke: {inv_err}"})


def extract_final(out: Dict[str, Any]) -> str:
    """
    Pull the blog markdown out of the final state dict,
    trying several fallback keys so we never show 'None'.
    """
    for key in ("final", "merged_md", "md_with_placeholders"):
        val = out.get(key)
        if val and isinstance(val, str) and val.strip() and val.strip().lower() != "none":
            return val

    # Last resort: reassemble from sections
    sections = out.get("sections") or []
    if sections:
        plan = out.get("plan")
        title = ""
        if hasattr(plan, "blog_title"):
            title = plan.blog_title
        elif isinstance(plan, dict):
            title = plan.get("blog_title", "")
        ordered = [md for _, md in sorted(sections, key=lambda x: x[0])]
        body = "\n\n".join(ordered).strip()
        return f"# {title}\n\n{body}\n" if title else body

    return ""


# =============================================================================
# Markdown → styled HTML  (images embedded as base64 data URIs)
# =============================================================================

def _image_to_data_uri(src: str, output_dir: Path) -> str:
    """Return a base64 data URI for a local image, or the original src if remote/missing."""
    if src.startswith("http://") or src.startswith("https://"):
        return src
    for base in [output_dir, Path(".")]:
        candidate = (base / src).resolve()
        if candidate.exists():
            suffix = candidate.suffix.lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "gif": "image/gif",  "webp": "image/webp"}.get(suffix.lstrip("."), "image/png")
            data = base64.b64encode(candidate.read_bytes()).decode()
            return f"data:{mime};base64,{data}"
    return src  # not found — let the browser show a broken image


def _md_to_html(md: str, output_dir: Path) -> str:
    # Embed local images as data URIs
    def _replace_img(m: re.Match) -> str:
        alt = m.group("alt")
        src = m.group("src").strip()
        return f"![{alt}]({_image_to_data_uri(src, output_dir)})"

    md_patched = re.sub(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)", _replace_img, md)

    # Convert markdown → HTML
    try:
        import markdown as md_lib
        html_body = md_lib.markdown(md_patched, extensions=["fenced_code", "tables", "nl2br"])
    except ImportError:
        html_body = _simple_md_to_html(md_patched)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.75;
    max-width: 820px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    color: #1a1a1a;
    background: #fff;
  }}
  h1 {{ font-size: 2rem; border-bottom: 2px solid #e5e7eb; padding-bottom: .4rem; margin-top: 0; }}
  h2 {{ font-size: 1.4rem; margin-top: 2.2rem; color: #111; }}
  h3 {{ font-size: 1.1rem; color: #333; }}
  pre {{
    background: #f6f8fa; border: 1px solid #ddd; border-radius: 6px;
    padding: 1rem; overflow-x: auto; font-size: .85rem;
  }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: .88em; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{
    border-left: 4px solid #d1d5db; margin: 1rem 0;
    padding: .5rem 1rem; color: #555; background: #fafafa;
  }}
  img {{ max-width: 100%; border-radius: 6px; margin: 1.2rem 0; display: block; }}
  em {{ color: #555; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: .45rem .75rem; text-align: left; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  a {{ color: #2563eb; }}
  hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 2rem 0; }}
  ul, ol {{ padding-left: 1.5rem; }}
  li {{ margin-bottom: .3rem; }}
</style>
</head>
<body>{html_body}</body>
</html>"""


def _simple_md_to_html(md: str) -> str:
    """Bare-minimum markdown renderer used only when the `markdown` package is absent."""
    lines, out, in_code, in_ul = md.split("\n"), [], False, False
    for line in lines:
        if line.startswith("```"):
            if in_code:
                out.append("</code></pre>"); in_code = False
            else:
                out.append("<pre><code>"); in_code = True
            continue
        if in_code:
            out.append(line); continue
        if in_ul and not (line.startswith("- ") or line.startswith("* ")):
            out.append("</ul>"); in_ul = False
        if re.match(r"^#{1,6} ", line):
            lvl = len(line) - len(line.lstrip("#"))
            out.append(f"<h{lvl}>{line[lvl+1:]}</h{lvl}>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{line[2:]}</blockquote>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append(f"<li>{line[2:]}</li>")
        elif not line.strip():
            out.append("<br>")
        else:
            l = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', line)
            l = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', l)
            l = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", l)
            l = re.sub(r"\*(.+?)\*", r"<em>\1</em>", l)
            l = re.sub(r"`(.+?)`", r"<code>\1</code>", l)
            out.append(f"<p>{l}</p>")
    if in_ul: out.append("</ul>")
    return "\n".join(out)


def render_markdown_preview(md: str, output_dir: Path):
    if not md or not md.strip():
        st.warning("No content to preview.")
        return
    components.html(_md_to_html(md, output_dir), height=920, scrolling=True)


# =============================================================================
# Past-blog helpers
# =============================================================================

def list_past_blogs(output_dir: Path) -> List[Path]:
    files = [p for p in output_dir.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")
st.title("✍️ Blog Writing Agent")

output_dir = get_output_dir()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area("Topic", height=120,
                         placeholder="e.g. How attention mechanisms work in transformers")
    as_of    = st.date_input("As-of date", value=date.today())
    run_btn  = st.button("🚀 Generate Blog", type="primary", use_container_width=True)

    st.divider()
    st.subheader("📂 Past Blogs")
    past_files = list_past_blogs(output_dir)
    if not past_files:
        st.caption(f"No saved blogs yet.")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                text  = p.read_text(encoding="utf-8", errors="replace")
                title = extract_title_from_md(text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label   = st.radio("", options=options, index=0, label_visibility="collapsed")
        selected_md_file = file_by_label.get(selected_label) if selected_label else None

        if st.button("📖 Load selected blog", use_container_width=True):
            if selected_md_file:
                md_text = selected_md_file.read_text(encoding="utf-8", errors="replace")
                st.session_state["last_out"] = {"plan": None, "evidence": [],
                                                "image_specs": [], "final": md_text}
                st.success(f"Loaded: {selected_md_file.name}")

# ── Session state ─────────────────────────────────────────────────────────────
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Preview", "🖼️ Images", "🧾 Logs"]
)

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic":                topic.strip(),
        "mode":                 "",
        "needs_research":       False,
        "queries":              [],
        "evidence":             [],
        "plan":                 None,
        "as_of":                as_of.isoformat(),
        "recency_days":         7,
        "sections":             [],
        "merged_md":            "",
        "md_with_placeholders": "",
        "image_specs":          [],
        "output_dir":           str(output_dir),
        "final":                "",
    }

    st.session_state["logs"] = []
    status            = st.status("Running graph…", expanded=True)
    progress_holder   = st.empty()
    last_node         = None

    for kind, node_name, payload in run_graph(app, inputs):

        if kind == "node":
            if node_name != last_node:
                status.write(f"➡️ `{node_name}`")
                last_node = node_name
            if payload:
                preview_keys = ("mode", "needs_research", "queries",
                                "plan", "image_specs", "sections", "final")
                summary = {k: v for k, v in payload.items() if k in preview_keys}
                if summary:
                    try:
                        progress_holder.json(json.loads(json.dumps(summary, default=str)))
                    except Exception:
                        pass
            st.session_state["logs"].append(
                f"[{node_name}] " + json.dumps(payload, default=str)[:600]
            )

        elif kind == "final":
            st.session_state["last_out"] = payload
            status.update(label="✅ Done!", state="complete", expanded=False)
            st.session_state["logs"].append(
                f"[final] final={repr(payload.get('final', ''))[:120]}"
            )
            st.rerun()

        elif kind == "error":
            err = payload.get("error", "unknown")
            st.error(f"❌ {err}")
            st.session_state["logs"].append(f"[error] {err}")

# ── Render output ─────────────────────────────────────────────────────────────
out = st.session_state.get("last_out")
if out:

    # ── Plan ──────────────────────────────────────────────────────────────────
    with tab_plan:
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan metadata (loaded from file).")
        else:
            plan_dict = (plan_obj.model_dump() if hasattr(plan_obj, "model_dump")
                         else plan_obj if isinstance(plan_obj, dict)
                         else json.loads(json.dumps(plan_obj, default=str)))
            st.markdown(f"### {plan_dict.get('blog_title', '—')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Audience", plan_dict.get("audience", "—"))
            c2.metric("Tone",     plan_dict.get("tone",     "—"))
            c3.metric("Kind",     plan_dict.get("blog_kind","—"))
            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame([{
                    "id":       t.get("id"),
                    "title":    t.get("title"),
                    "words":    t.get("target_words"),
                    "research": t.get("requires_research"),
                    "code":     t.get("requires_code"),
                    "tags":     ", ".join(t.get("tags") or []),
                } for t in tasks]).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)
                with st.expander("Full JSON"):
                    st.json(tasks)

    # ── Evidence ──────────────────────────────────────────────────────────────
    with tab_evidence:
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence (closed_book mode or Tavily not configured).")
        else:
            rows = []
            for e in evidence:
                e = e.model_dump() if hasattr(e, "model_dump") else e
                rows.append({"title": e.get("title"), "published_at": e.get("published_at"),
                             "source": e.get("source"), "url": e.get("url")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Preview ───────────────────────────────────────────────────────────────
    with tab_preview:
        final_md = extract_final(out)          # ← robust extraction, never returns "None"

        if not final_md:
            st.warning("No content yet — check the Logs tab for errors.")
        else:
            plan_obj   = out.get("plan")
            blog_title = (plan_obj.blog_title if hasattr(plan_obj, "blog_title")
                          else plan_obj.get("blog_title", "") if isinstance(plan_obj, dict)
                          else extract_title_from_md(final_md, "blog"))
            md_filename = f"{safe_slug(blog_title)}.md"

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("⬇️ Download Markdown", data=final_md.encode(),
                                   file_name=md_filename, mime="text/markdown",
                                   use_container_width=True)
            with col_dl2:
                bundle = bundle_zip(final_md, md_filename, output_dir / "images")
                st.download_button("📦 Download Bundle (MD + images)", data=bundle,
                                   file_name=f"{safe_slug(blog_title)}_bundle.zip",
                                   mime="application/zip", use_container_width=True)

            render_markdown_preview(final_md, output_dir)

            with st.expander("🔤 Raw Markdown"):
                st.code(final_md, language="markdown")

    # ── Images ────────────────────────────────────────────────────────────────
    with tab_images:
        specs       = out.get("image_specs") or []
        images_dir  = output_dir / "images"

        if not specs and not images_dir.exists():
            st.info("No images generated.")
        else:
            if specs:
                with st.expander("Image prompts & specs"):
                    st.json(specs)
            if images_dir.exists():
                files = sorted(p for p in images_dir.iterdir() if p.is_file())
                if files:
                    cols = st.columns(min(len(files), 3))
                    for i, p in enumerate(files):
                        with cols[i % 3]:
                            st.image(str(p), caption=p.name, use_container_width=True)
                else:
                    st.warning("images/ folder is empty.")
                z = images_zip(images_dir)
                if z:
                    st.download_button("⬇️ Download All Images", data=z,
                                       file_name="images.zip", mime="application/zip")

    # ── Logs ──────────────────────────────────────────────────────────────────
    with tab_logs:
        st.text_area("Event log",
                     value="\n\n".join(st.session_state["logs"][-100:]),
                     height=540)

else:
    with tab_preview:
        st.info("👈 Enter a topic in the sidebar and click **Generate Blog**.")