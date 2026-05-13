import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.langgraphagenticai.metrics.tracker import MetricsDB

AGENT_COLORS = {
    "supervisor": "#636EFA",
    "research_agent": "#EF553B",
    "rag_agent": "#00CC96",
    "news_agent": "#AB63FA",
    "writer_agent": "#FFA15A",
}

AGENT_ICONS = {
    "supervisor": "🧠",
    "research_agent": "🔍",
    "rag_agent": "📄",
    "news_agent": "📰",
    "writer_agent": "✍️",
}


_AGENT_ROLE = {
    "supervisor": "Planner",
    "research_agent": "Web Researcher",
    "rag_agent": "Document Reader",
    "news_agent": "News Checker",
    "writer_agent": "Answer Writer",
}


def _color_sequence(agent_names: list) -> list:
    return [AGENT_COLORS.get(n, "#19D3F3") for n in agent_names]


def _render_plain_english_section(df_sum, df_all):
    """Non-technical summary placed before all charts."""
    st.subheader("📝 What This All Means — Plain English")
    st.caption("A jargon-free summary of how the AI performed.")

    total_questions = int(df_sum["total_calls"].sum())
    avg_latency_ms = float(df_all["latency_ms"].mean()) if not df_all.empty else 0
    success_rate = (
        float(df_all["success"].sum() / len(df_all) * 100) if not df_all.empty else 0
    )
    top_agent_row = df_sum.loc[df_sum["total_calls"].idxmax()]
    top_agent_name = top_agent_row["agent_name"]
    top_agent_plain = _AGENT_ROLE.get(top_agent_name, top_agent_name)
    top_agent_icon = AGENT_ICONS.get(top_agent_name, "🤖")

    # Overall grade
    if success_rate >= 98 and avg_latency_ms < 3000:
        grade, border_color, grade_note = "A", "#00CC96", "Excellent — fast & reliable"
    elif success_rate >= 90 and avg_latency_ms < 6000:
        grade, border_color, grade_note = "B", "#FFA15A", "Good — mostly smooth"
    elif success_rate >= 70:
        grade, border_color, grade_note = "C", "#EF553B", "Fair — some hiccups"
    else:
        grade, border_color, grade_note = "D", "#EF553B", "Needs attention"

    # Speed in human terms
    avg_sec = avg_latency_ms / 1000
    if avg_sec < 2:
        speed_plain = f"⚡ Very fast — answered in under 2 seconds on average ({avg_sec:.1f}s)"
    elif avg_sec < 5:
        speed_plain = f"🕐 Normal speed — about {avg_sec:.1f} seconds per answer"
    elif avg_sec < 10:
        speed_plain = f"🐌 A bit slow — {avg_sec:.1f} seconds per answer; complex queries take time"
    else:
        speed_plain = f"🐢 Slow — {avg_sec:.1f}s per answer; consider a faster model"

    # Reliability in human terms
    if success_rate == 100:
        reliability_plain = "✅ Everything worked perfectly — zero errors"
    elif success_rate >= 95:
        reliability_plain = f"✅ Mostly perfect — {success_rate:.0f}% success rate"
    elif success_rate >= 80:
        reliability_plain = f"⚠️ Generally fine — {success_rate:.0f}% success, some minor issues"
    else:
        reliability_plain = f"❌ Struggling — only {success_rate:.0f}% success; check API keys"

    col_grade, col_info = st.columns([1, 3])
    with col_grade:
        st.markdown(
            f"""<div style='text-align:center;padding:24px 12px;border:3px solid {border_color};
            border-radius:12px;'>
            <span style='font-size:52px;font-weight:bold;color:{border_color}'>{grade}</span>
            <p style='margin:4px 0 0;font-size:13px;color:{border_color}'>{grade_note}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_info:
        st.markdown(f"""
The AI handled **{total_questions} API call(s)** in this session.

- **Speed:** {speed_plain}
- **Reliability:** {reliability_plain}
- **Hardest worker:** {top_agent_icon} **{top_agent_plain}** — was called the most to help answer your questions
""")

    st.divider()


def render_dashboard():
    db = MetricsDB()

    # ── Header ──────────────────────────────────────────────────────────────
    col_title, col_actions = st.columns([4, 1])
    with col_title:
        st.title("📊 Agent Performance Dashboard")
        st.caption("Real-time metrics for every LLM call made by each agent.")
    with col_actions:
        st.write("")
        if st.button("🔄 Refresh", width='stretch', type="primary"):
            st.rerun()
        if st.button("🗑️ Clear All", width='stretch'):
            db.clear()
            st.rerun()

    summary = db.get_summary()
    all_metrics = db.get_all()

    if not summary:
        st.info("No metrics recorded yet. Run a **Pure Agentic AI** query first!", icon="ℹ️")
        return

    df_sum = pd.DataFrame(summary)
    df_all = pd.DataFrame(all_metrics)

    # ── Plain English Section ────────────────────────────────────────────────
    _render_plain_english_section(df_sum, df_all)

    # ── Overview KPI Cards ───────────────────────────────────────────────────
    st.subheader("Overview")
    total_calls = int(df_sum["total_calls"].sum())
    total_tokens = int(df_sum["total_tokens"].sum())
    avg_latency = float(df_all["latency_ms"].mean()) if not df_all.empty else 0.0
    success_rate = (
        float(df_all["success"].sum() / len(df_all) * 100) if not df_all.empty else 0.0
    )
    unique_agents = df_sum["agent_name"].nunique()
    avg_temp = float(df_all["temperature"].mean()) if "temperature" in df_all.columns and not df_all.empty else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total API Calls", total_calls)
    c2.metric("Total Tokens", f"{total_tokens:,}")
    c3.metric("Avg Latency", f"{avg_latency:.0f} ms")
    c4.metric("Success Rate", f"{success_rate:.1f}%")
    c5.metric("Active Agents", unique_agents)
    c6.metric("Avg Temperature", f"{avg_temp:.2f}")

    st.divider()

    # ── Row 1: Calls & Latency ───────────────────────────────────────────────
    st.subheader("Agent Performance Comparison")
    col_l, col_r = st.columns(2)

    with col_l:
        fig_calls = px.bar(
            df_sum,
            x="agent_name",
            y="total_calls",
            title="API Calls per Agent",
            color="agent_name",
            color_discrete_map=AGENT_COLORS,
            labels={"agent_name": "Agent", "total_calls": "Calls"},
            text="total_calls",
        )
        fig_calls.update_traces(textposition="outside")
        fig_calls.update_layout(showlegend=False, height=360, xaxis_tickangle=-20)
        st.plotly_chart(fig_calls, width='stretch')

    with col_r:
        fig_lat = px.bar(
            df_sum,
            x="agent_name",
            y="avg_latency_ms",
            title="Avg Latency per Agent (ms)",
            color="agent_name",
            color_discrete_map=AGENT_COLORS,
            labels={"agent_name": "Agent", "avg_latency_ms": "Avg Latency (ms)"},
            text="avg_latency_ms",
        )
        fig_lat.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_lat.update_layout(showlegend=False, height=360, xaxis_tickangle=-20)
        st.plotly_chart(fig_lat, width='stretch')

    # ── Row 2: Stacked Token Bar ─────────────────────────────────────────────
    st.subheader("Token Usage Breakdown")
    fig_tok = go.Figure()
    fig_tok.add_trace(
        go.Bar(
            name="Input Tokens",
            x=df_sum["agent_name"],
            y=df_sum["total_input"],
            marker_color="#636EFA",
        )
    )
    fig_tok.add_trace(
        go.Bar(
            name="Output Tokens",
            x=df_sum["agent_name"],
            y=df_sum["total_output"],
            marker_color="#EF553B",
        )
    )
    fig_tok.update_layout(
        barmode="stack",
        title="Input vs Output Tokens per Agent",
        xaxis_title="Agent",
        yaxis_title="Tokens",
        height=380,
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig_tok, width='stretch')

    # ── Row 3: Timeline ──────────────────────────────────────────────────────
    st.subheader("Latency Timeline")
    timeline = db.get_latency_timeline()
    if timeline:
        df_time = pd.DataFrame(timeline)
        df_time["timestamp"] = pd.to_datetime(df_time["timestamp"])
        fig_time = px.line(
            df_time,
            x="timestamp",
            y="latency_ms",
            color="agent_name",
            markers=True,
            title="Latency Over Time by Agent",
            labels={"latency_ms": "Latency (ms)", "timestamp": "Time", "agent_name": "Agent"},
            color_discrete_map=AGENT_COLORS,
        )
        fig_time.update_layout(height=380)
        st.plotly_chart(fig_time, width='stretch')

    # ── Row 4: Pie charts ────────────────────────────────────────────────────
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fig_pie_tok = px.pie(
            df_sum,
            values="total_tokens",
            names="agent_name",
            title="Token Share by Agent",
            color="agent_name",
            color_discrete_map=AGENT_COLORS,
            hole=0.35,
        )
        fig_pie_tok.update_layout(height=360)
        st.plotly_chart(fig_pie_tok, width='stretch')

    with col_p2:
        fig_pie_calls = px.pie(
            df_sum,
            values="total_calls",
            names="agent_name",
            title="API Call Share by Agent",
            color="agent_name",
            color_discrete_map=AGENT_COLORS,
            hole=0.35,
        )
        fig_pie_calls.update_layout(height=360)
        st.plotly_chart(fig_pie_calls, width='stretch')

    # ── Row 5: Temperature ───────────────────────────────────────────────────
    if "avg_temperature" in df_sum.columns:
        st.subheader("Temperature Settings per Agent")
        fig_temp = px.bar(
            df_sum,
            x="agent_name",
            y="avg_temperature",
            title="Avg Temperature Used per Agent",
            color="agent_name",
            color_discrete_map=AGENT_COLORS,
            labels={"agent_name": "Agent", "avg_temperature": "Avg Temperature"},
            range_y=[0.0, 1.0],
            text="avg_temperature",
        )
        fig_temp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_temp.update_layout(showlegend=False, height=320, xaxis_tickangle=-20)
        st.plotly_chart(fig_temp, width='stretch')

    # ── Summary Table ────────────────────────────────────────────────────────
    st.subheader("Agent Summary Table")
    rename = {
        "agent_name": "Agent",
        "total_calls": "Calls",
        "avg_latency_ms": "Avg Latency (ms)",
        "total_tokens": "Total Tokens",
        "total_input": "Input Tokens",
        "total_output": "Output Tokens",
        "avg_temperature": "Avg Temp",
        "successes": "Successes",
    }
    styled = df_sum.rename(columns=rename)
    # Add icon column
    styled.insert(0, "Icon", styled["Agent"].map(lambda a: AGENT_ICONS.get(a, "🤖")))
    st.dataframe(styled, width='stretch', hide_index=True)

    # ── Raw Log ──────────────────────────────────────────────────────────────
    with st.expander("📋 Raw Metrics Log", expanded=False):
        if not df_all.empty:
            show_cols = [
                "agent_name", "model", "temperature",
                "input_tokens", "output_tokens", "total_tokens",
                "latency_ms", "timestamp", "success", "query_preview",
            ]
            avail = [c for c in show_cols if c in df_all.columns]
            st.dataframe(df_all[avail], width='stretch', hide_index=True)
