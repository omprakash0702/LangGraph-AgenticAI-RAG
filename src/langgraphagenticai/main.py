import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.nodes.agentic_nodes import _strip_thinking
from src.langgraphagenticai.ui.streamlitui.dashboard import render_dashboard
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI

AGENT_ICONS = {
    "planner":        "📋",
    "supervisor":     "🧠",
    "research_agent": "🔍",
    "rag_agent":      "📄",
    "news_agent":     "📰",
    "writer_agent":   "✍️",
    "reflector":      "🔄",
    "evaluator":      "✅",
}


_GRAPH_VERSION = 12  # increment any time agent code changes to bust the cache

@st.cache_resource(show_spinner=False)
def _build_graph(groq_model: str, temperature: float, groq_api_key: str, version: int = _GRAPH_VERSION):
    from langchain_groq import ChatGroq
    model = ChatGroq(api_key=groq_api_key, model=groq_model, temperature=temperature)
    return GraphBuilder(model).setup_graph("Pure Agentic AI")


def _format_history_for_context(chat_turns: list) -> list[str]:
    formatted = []
    for turn in chat_turns[-4:]:
        formatted.append(f"User: {turn['user_msg']}\nAI: {turn['ai_msg'][:600]}")
    return formatted


def _render_agent_path(scratchpad: list, called: list, iteration: int):
    label_str = f"agents: {', '.join(called)}" if called else "direct response"
    with st.expander(f"🔍 Agent path — {iteration} step(s) · {label_str}", expanded=False):
        if scratchpad:
            for i, step in enumerate(scratchpad, 1):
                if "]:" in step:
                    label = step.split("]:")[0].replace("[", "").strip()
                    content = step.split("]:")[1].strip()
                else:
                    label, content = "agent", step
                base = label.split("(")[0].strip()
                icon = AGENT_ICONS.get(base, "🤖")
                preview = content[:300] + "…" if len(content) > 300 else content
                cols = st.columns([0.07, 0.93])
                cols[0].markdown(f"**{i}. {icon}**")
                cols[1].markdown(f"**`{label}`** — {preview}")
        else:
            st.caption("No specialist agents — answered directly from built-in knowledge.")


def _render_report_card(called: list, elapsed: float, success: bool):
    specialists = [a for a in called if a != "supervisor"]
    n = len(specialists)

    if elapsed < 2:
        speed = f"⚡ {elapsed:.1f}s"
    elif elapsed < 8:
        speed = f"🕐 {elapsed:.1f}s"
    else:
        speed = f"🐢 {elapsed:.1f}s"

    effort = (
        "💡 Direct" if n == 0
        else "🔍 Light" if n == 1
        else "🧠 Moderate" if n == 2
        else "🔬 Deep"
    )
    result = "✅ OK" if success else "⚠️ Partial"

    with st.expander("📋 Response card", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Speed", speed)
        c2.metric("Effort", effort, f"{n} specialist(s)")
        c3.metric("Result", result)
        if specialists:
            steps = {
                "research_agent": "searched the web",
                "rag_agent": "read your document",
                "news_agent": "checked latest news",
            }
            journey = "The AI " + ", then ".join(steps.get(a, a) for a in specialists if a in steps) + "."
            st.caption(journey)


def _preload_rag_if_needed():
    """Eagerly build the FAISS index after upload so the first query is fast."""
    if not st.session_state.get("rag_file_bytes"):
        return
    file_name = st.session_state.get("rag_file_name", "")
    file_size = st.session_state.get("rag_file_size", 0)
    file_key = f"{file_name}_{file_size}"
    cache = st.session_state.get("_rag_cache", {})
    if cache.get("key") == file_key and cache.get("index") is not None:
        return  # already built

    from src.langgraphagenticai.rag.retriever import build_or_get_index_from_session, search
    with st.spinner(f"Indexing **{file_name}**…"):
        try:
            index = build_or_get_index_from_session()
            preview_chunks = search(index, "main content overview summary introduction", k=3)
            st.session_state["_rag_preview"] = preview_chunks
        except Exception as exc:
            st.error(f"Could not index document: {exc}")


def load_langgraph_agenticai_app():
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Failed to load UI configuration.")
        return

    # ── API key propagation ───────────────────────────────────────────────────
    tavily_key = user_input.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    groq_key = user_input.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")

    # Eagerly index document if freshly uploaded
    _preload_rag_if_needed()

    # ── Session state init ────────────────────────────────────────────────────
    st.session_state.setdefault("chat_turns", [])

    tab_agent, tab_dashboard = st.tabs(["🤖 Agent", "📊 Dashboard"])

    with tab_dashboard:
        render_dashboard()

    with tab_agent:
        st.markdown("## 🤖 Pure Agentic AI")
        st.caption(
            "The Supervisor autonomously routes your query to specialist agents — "
            "Web Research 🔍, Document Q&A 📄, AI News 📰, or answers directly."
        )

        # ── Document context banner ───────────────────────────────────────────
        rag_file_name = st.session_state.get("rag_file_name")
        if rag_file_name:
            size_kb = round(st.session_state.get("rag_file_size", 0) / 1024, 1)
            is_indexed = st.session_state.get("_rag_cache", {}).get("key") == \
                         f"{rag_file_name}_{st.session_state.get('rag_file_size', 0)}"
            status = "indexed ✅" if is_indexed else "indexing…"
            col_info, col_rm = st.columns([6, 1])
            col_info.info(
                f"📄 **{rag_file_name}** ({size_kb} KB) — {status}  \n"
                "Ask any question about this document and the AI will answer from it."
            )
            if col_rm.button("✕", help="Remove document", width='stretch'):
                for _k in ("rag_file", "rag_file_name", "rag_file_bytes", "rag_file_size", "_rag_cache", "_rag_preview"):
                    st.session_state.pop(_k, None)
                st.rerun()

            if st.session_state.get("_rag_preview"):
                with st.expander("🔍 Verify: what the AI extracted from your document", expanded=True):
                    st.caption("If this text looks wrong, upload a .txt file instead (copy-paste the content into Notepad → Save As .txt).")
                    st.text(st.session_state["_rag_preview"][:1500])

        # ── Clear conversation button ─────────────────────────────────────────
        col_clear, col_space = st.columns([1, 5])
        with col_clear:
            if st.button("🗑️ Clear chat", width='stretch'):
                st.session_state.chat_turns = []
                st.rerun()

        st.divider()

        # ── Render all previous turns ─────────────────────────────────────────
        for turn in st.session_state.chat_turns:
            with st.chat_message("user"):
                st.markdown(turn["user_msg"])
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(turn["ai_msg"])
                if "rag_agent" in turn.get("agents_called", []):
                    st.caption(f"📄 Answered from **{rag_file_name or 'your document'}**")
            if turn.get("rag_context") and "rag_agent" in turn.get("agents_called", []):
                with st.expander("🔎 Document sections retrieved for this query", expanded=False):
                    st.text(turn["rag_context"][:3000])
            _render_agent_path(turn["scratchpad"], turn["agents_called"], turn["iteration"])
            _render_report_card(turn["agents_called"], turn["elapsed"], turn["success"])

        # ── Welcome message if no history ─────────────────────────────────────
        if not st.session_state.chat_turns:
            with st.chat_message("assistant", avatar="🤖"):
                if rag_file_name:
                    st.markdown(
                        f"**{rag_file_name}** is loaded and indexed. Ask me anything about it!\n\n"
                        "I can also help with:\n"
                        "- 🔍 **Research** — factual questions, web search\n"
                        "- 📰 **AI News** — what's happening in AI/ML this week\n"
                        "- 💬 **General chat** — coding, explanations, brainstorming"
                    )
                else:
                    st.markdown(
                        "Hello! I'm your **Pure Agentic AI** assistant.\n\n"
                        "I can help you with:\n"
                        "- 🔍 **Research** — factual questions, how-things-work, comparisons\n"
                        "- 📰 **AI News** — what's happening in AI/ML this week\n"
                        "- 📄 **Document Q&A** — upload a PDF or TXT in the sidebar, then ask questions\n"
                        "- 💬 **General chat** — coding, explanations, brainstorming\n\n"
                        "What would you like to know?"
                    )

        # ── New message input ─────────────────────────────────────────────────
        placeholder = (
            f"Ask about {rag_file_name}…" if rag_file_name else "Ask anything…"
        )
        user_message = st.chat_input(placeholder)
        if not user_message:
            return

        if not groq_key:
            st.error("Groq API key is missing. Add it to .env or the sidebar.")
            return

        with st.chat_message("user"):
            st.markdown(user_message)

        # ── Build / retrieve cached graph ─────────────────────────────────────
        try:
            graph = _build_graph(
                groq_model=user_input.get("selected_groq_model", "llama-3.3-70b-versatile"),
                temperature=float(user_input.get("temperature", 0.3)),
                groq_api_key=groq_key,
                version=_GRAPH_VERSION,
            )
        except Exception as exc:
            st.error(f"Could not initialise model: {exc}")
            return

        # ── Build initial state ───────────────────────────────────────────────
        history_context = _format_history_for_context(st.session_state.chat_turns)

        initial_state: dict = {
            "messages": [HumanMessage(content=user_message)],
            "next_agent": "",
            "agent_scratchpad": [],
            "called_agents": [],
            "iteration": 0,
            "conversation_history": history_context,
            "rag_context_text": None,
            # Pass file name (truthy string) so supervisor knows RAG is available;
            # the rag_agent reads bytes directly from session_state — no UploadedFile in state.
            "rag_file": rag_file_name or None,
            # Agentic fields
            "plan": None,
            "sub_goals": [],
            "reflection": None,
            "reflection_count": 0,
            "evaluation": None,
        }

        # ── Run the graph ─────────────────────────────────────────────────────
        with st.spinner("Thinking…"):
            t0 = time.time()
            try:
                final_state = graph.invoke(initial_state)
            except Exception as exc:
                st.error(f"Agent execution failed: {exc}")
                return
            elapsed = time.time() - t0

        scratchpad = final_state.get("agent_scratchpad", [])
        called = final_state.get("called_agents", [])
        iteration = final_state.get("iteration", 0)

        final_answer = next(
            (_strip_thinking(m.content) for m in reversed(final_state.get("messages", [])) if isinstance(m, AIMessage) and m.content),
            None,
        )

        rag_context = st.session_state.get("_rag_last_context") if "rag_agent" in called else None

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(final_answer or "No answer generated.")
            if "rag_agent" in called:
                st.caption(f"📄 Answered from **{rag_file_name or 'your document'}**")

        if rag_context:
            with st.expander("🔎 Document sections retrieved for this query", expanded=False):
                st.text(rag_context[:3000])

        _render_agent_path(scratchpad, called, iteration)
        _render_report_card(called, elapsed, bool(final_answer))

        st.session_state.chat_turns.append({
            "user_msg": user_message,
            "ai_msg": final_answer or "",
            "agents_called": called,
            "scratchpad": scratchpad,
            "iteration": iteration,
            "elapsed": elapsed,
            "success": bool(final_answer),
            "rag_context": rag_context,
        })
