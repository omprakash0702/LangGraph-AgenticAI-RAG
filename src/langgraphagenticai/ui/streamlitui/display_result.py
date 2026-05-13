import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

AGENT_ICONS = {
    "supervisor": "🧠",
    "research_agent": "🔍",
    "rag_agent": "📄",
    "news_agent": "📰",
    "writer_agent": "✍️",
}

_AGENT_PLAIN = {
    "research_agent": "searched the web for you",
    "rag_agent": "read through your document",
    "news_agent": "checked the latest AI news",
    "writer_agent": "wrote the final answer",
}


def _render_report_card(called_agents: list, elapsed_sec: float, success: bool):
    """Plain-English summary card shown after every Pure Agentic AI response."""

    # Speed label
    if elapsed_sec < 2:
        speed_label, speed_help = "⚡ Very fast", f"{elapsed_sec:.1f}s — almost instant"
    elif elapsed_sec < 5:
        speed_label, speed_help = "🕐 Normal", f"{elapsed_sec:.1f}s — typical for web research"
    elif elapsed_sec < 10:
        speed_label, speed_help = "🐌 A bit slow", f"{elapsed_sec:.1f}s — complex query"
    else:
        speed_label, speed_help = "🐢 Slow", f"{elapsed_sec:.1f}s — heavy workload"

    # Effort label
    specialists = [a for a in called_agents if a not in ("supervisor",)]
    n = len(specialists)
    if n == 0:
        effort_label = "💡 Instant reply"
        effort_help = "Answered directly — no research needed"
    elif n == 1:
        effort_label = "🔍 Light research"
        effort_help = "1 specialist consulted"
    elif n == 2:
        effort_label = "🧠 Moderate effort"
        effort_help = "2 specialists consulted"
    else:
        effort_label = "🔬 Deep investigation"
        effort_help = f"{n} specialists consulted"

    # Quality label
    quality_label = "✅ Success" if success else "⚠️ Partial"
    quality_help = "All steps completed without errors" if success else "Some steps encountered issues"

    # Journey sentence
    steps = [_AGENT_PLAIN[a] for a in called_agents if a in _AGENT_PLAIN]
    if steps:
        journey = "The AI " + ", then ".join(steps) + "."
    else:
        journey = "The AI answered directly from its built-in knowledge — no external tools needed."

    with st.expander("📋 Response Report Card — plain English summary", expanded=True):
        st.caption("A simple breakdown of how this answer was produced.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Speed", speed_label, speed_help)
        c2.metric("Effort", effort_label, effort_help)
        c3.metric("Result", quality_label, quality_help)
        st.info(f"**What happened:** {journey}", icon="💬")


class DisplayResultStreamlit:
    def __init__(self, usecase, graph, initial_state):
        self.usecase = usecase
        self.graph = graph
        self.initial_state = initial_state

    def display_result_on_ui(self):

        # ── Pure Agentic AI ──────────────────────────────────────────────────
        if self.usecase == "Pure Agentic AI":
            with st.spinner("🤖 Multi-agent system is working on your request..."):
                t0 = time.time()
                final_state = self.graph.invoke(self.initial_state)
                elapsed = time.time() - t0

            scratchpad = final_state.get("agent_scratchpad", [])
            iteration = final_state.get("iteration", 0)

            called = final_state.get("called_agents", [])
            label_str = f"  •  agents used: {', '.join(called)}" if called else "  •  direct response"
            with st.expander(
                f"🔍 Agent Execution Path  •  {iteration} step(s){label_str}", expanded=True
            ):
                if scratchpad:
                    for i, step in enumerate(scratchpad, 1):
                        if "]:" in step:
                            label = step.split("]:")[0].replace("[", "").strip()
                            content = step.split("]:")[1].strip()
                        else:
                            label, content = "agent", step

                        # label may be "research_agent (searched 3x: ...)" — extract base name for icon
                        base_agent = label.split("(")[0].strip()
                        icon = AGENT_ICONS.get(base_agent, "🤖")
                        preview = content[:280] + "…" if len(content) > 280 else content

                        cols = st.columns([0.08, 0.92])
                        cols[0].markdown(f"**{i}. {icon}**")
                        cols[1].markdown(f"**`{label}`** — {preview}")
                else:
                    st.info(
                        "✍️ **Direct response** — the Supervisor determined no specialist "
                        "agents were needed for this query.",
                        icon="💡",
                    )

            st.divider()

            # Final answer from last AI message
            messages = final_state.get("messages", [])
            final_answer = next(
                (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
                None,
            )
            if final_answer:
                st.subheader("💬 Final Answer")
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(final_answer)
            else:
                st.warning("No final answer was generated.")

            st.divider()
            _render_report_card(
                called_agents=called,
                elapsed_sec=elapsed,
                success=bool(final_answer),
            )

        # ── Basic Chatbot ────────────────────────────────────────────────────
        elif self.usecase == "Basic Chatbot":
            for event in self.graph.stream(self.initial_state):
                for value in event.values():
                    with st.chat_message("assistant"):
                        st.write(value["messages"].content)

        # ── Chatbot With Web ─────────────────────────────────────────────────
        elif self.usecase == "Chatbot With Web":
            res = self.graph.invoke(self.initial_state)
            for msg in res["messages"]:
                if isinstance(msg, HumanMessage):
                    with st.chat_message("user"):
                        st.write(msg.content)
                elif isinstance(msg, ToolMessage):
                    with st.chat_message("assistant"):
                        st.write(msg.content)
                elif isinstance(msg, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(msg.content)

        # ── AI News ──────────────────────────────────────────────────────────
        elif self.usecase == "AI News":
            self.graph.invoke(self.initial_state)
            path = f"./AINews/{self.initial_state['messages'][0].content.lower()}_summary.md"
            with open(path, "r", encoding="utf-8") as f:
                st.markdown(f.read(), unsafe_allow_html=True)

        # ── RAG ──────────────────────────────────────────────────────────────
        elif self.usecase == "RAG":
            res = self.graph.invoke(self.initial_state)
            with st.chat_message("assistant"):
                st.write(res.get("rag_answer", "No answer found."))
