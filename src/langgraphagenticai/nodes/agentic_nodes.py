import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..metrics.callbacks import MetricsCallbackHandler


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks emitted by reasoning models (e.g. Qwen3).

    If stripping leaves nothing — meaning the model wrapped its entire output in
    <think> — fall back to the content inside the last think block rather than
    returning an empty string.
    """
    if not text:
        return ""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if stripped:
        return stripped
    # Model put everything inside <think>; extract the last think block's content
    blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    # No closing tag — strip the opening tag and return the rest
    return re.sub(r"<think>", "", text, flags=re.IGNORECASE).strip()

MAX_ITERATIONS = 5
MAX_SEARCHES = 4   # max tool calls per ReAct agent per turn

# ── Tavily helpers ────────────────────────────────────────────────────────────

def _tavily_client():
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY")
    return TavilyClient(api_key=api_key) if api_key else TavilyClient()


def _search_web(query: str) -> str:
    client = _tavily_client()
    results = client.search(
        query=query,
        max_results=5,
        search_depth="advanced",
        include_answer=True,
    )
    return _format_tool_result(results)


def _search_news(query: str) -> str:
    client = _tavily_client()
    results = client.search(
        query=query,
        topic="news",
        days=7,
        max_results=7,
        search_depth="basic",
        include_answer=True,
    )
    return _format_tool_result(results)


def _format_tool_result(results: dict) -> str:
    answer = results.get("answer", "")
    items = sorted(results.get("results", []), key=lambda r: r.get("score", 0), reverse=True)
    parts = []
    if answer:
        parts.append(f"Direct answer: {answer}")
    for r in items[:5]:
        title = r.get("title", "Untitled")
        content = r.get("content", "").strip()
        url = r.get("url", "")
        score = r.get("score", 0)
        date = r.get("published_date", "")
        meta = f"relevance={score:.2f}" + (f", {date}" if date else "")
        parts.append(f"**{title}** ({meta})\n{content}\nSource: {url}")
    return "\n\n---\n\n".join(parts) if parts else "No results found."


# ── Manual ReAct loop ─────────────────────────────────────────────────────────

_REACT_SYSTEM = """You are a {role} with access to {tool_description}.

Solve the user's query step-by-step using this protocol:

To search: output a line starting with exactly "SEARCH:" followed by your query.
  Example: SEARCH: {example_query}

When you have gathered enough information: output "DONE" on its own line,
then immediately write your full answer below it.

Rules:
- Always start by searching — never answer purely from memory.
- You may search up to {max_searches} times. Use different, specific queries each time.
- After DONE, write a complete, well-structured markdown answer.
- Do not output anything other than SEARCH:... or DONE + answer."""


def _react_loop(
    llm,
    query: str,
    role: str,
    search_fn,
    max_searches: int = MAX_SEARCHES,
    tool_description: str = "a real-time web search tool",
    example_query: str = "your search query here",
) -> tuple[str, list[str]]:
    """
    Manual ReAct: the LLM decides what to search and when to stop.
    Returns (final_answer, list_of_search_queries_used).
    """
    system = _REACT_SYSTEM.format(
        role=role,
        max_searches=max_searches,
        tool_description=tool_description,
        example_query=example_query,
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=query),
    ]
    searches_done = []

    for step in range(max_searches + 2):
        response = llm.invoke(messages)
        raw = _strip_thinking(response.content)

        # Scan for SEARCH: directive
        if "SEARCH:" in raw:
            # Take first SEARCH: line only
            for line in raw.splitlines():
                line = line.strip()
                if line.upper().startswith("SEARCH:"):
                    search_query = line[len("SEARCH:"):].strip()
                    if search_query and search_query not in searches_done:
                        searches_done.append(search_query)
                        try:
                            tool_result = search_fn(search_query)
                        except Exception as e:
                            tool_result = f"Search error: {e}"

                        messages.append(AIMessage(content=raw))
                        messages.append(HumanMessage(
                            content=(
                                f"Search results for '{search_query}':\n\n{tool_result}\n\n"
                                f"Searches used so far: {len(searches_done)}/{max_searches}. "
                                "Continue searching or output DONE then your answer."
                            )
                        ))
                    break
            continue

        # DONE keyword detected — extract answer below it
        if "DONE" in raw.upper():
            idx = raw.upper().find("DONE")
            answer = raw[idx + len("DONE"):].strip().lstrip("\n").strip()
            if not answer:
                # LLM said DONE but answer is on next turn — request it
                messages.append(AIMessage(content=raw))
                messages.append(HumanMessage(content=f"Now write the complete answer to: {query}"))
                follow = llm.invoke(messages)
                answer = _strip_thinking(follow.content)
            return answer, searches_done

        # Neither SEARCH nor DONE — treat as final answer
        return raw, searches_done

    # Exhausted steps — synthesise from conversation
    messages.append(HumanMessage(content=f"You've done {len(searches_done)} searches. Now write the final answer to: {query}"))
    final = llm.invoke(messages)
    return _strip_thinking(final.content), searches_done


# ── System prompts ────────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are a strategic planner for a multi-agent AI system.

AVAILABLE AGENTS:
- research_agent : real-time web search for facts, current events, technical info
- rag_agent      : searches an uploaded document (only if one is available)
- news_agent     : AI/ML news from the past 7 days
- writer_agent   : synthesises all gathered info into the final answer (always last)

Analyze the user query and respond in EXACTLY this format — no extra text:
COMPLEXITY: simple | moderate | complex
PLAN: agent1, agent2, ...
GOAL: <one sentence — what does a complete answer look like>
SUB_GOALS:
- <specific question or piece of info needed>
- <another specific question>

RULES:
- Greetings, math, coding, small talk → COMPLEXITY: simple, PLAN: writer_agent
- Single document question → PLAN: rag_agent, writer_agent
- Single research question → PLAN: research_agent, writer_agent
- Multi-part question → chain multiple agents before writer_agent
- Always end with writer_agent"""

SUPERVISOR_PROMPT = """You are an AI Orchestrator managing a team of specialist agents.

AVAILABLE AGENTS:
- research_agent : web search — factual, technical, or general-knowledge queries needing current info
- rag_agent      : document Q&A — searches the uploaded file; ONLY when rag_file IS available
- news_agent     : AI/ML news — news, model releases, industry trends from the past 7 days
- writer_agent   : final answer — synthesises all gathered info; also answers directly from knowledge

DECISION RULES (apply top-down, first match wins):
1. Follow the EXECUTION PLAN if one is provided — respect its agent order
2. Act on REFLECTION FEEDBACK — if it says RETRY or ESCALATE, follow that guidance
3. Act on EVALUATOR FEEDBACK — if the answer needs revision, research the missing info
4. Greetings, small talk, "thanks", simple arithmetic → writer_agent immediately
5. Code, programming, math, logic, debugging → writer_agent immediately
6. Any agent already in "Already called" must NOT be called again (unless reflection says RETRY)
7. rag_file IS available AND query is about the document → rag_agent FIRST
8. Query about AI/ML news, recent model releases → news_agent
9. Factual questions, current events needing current data → research_agent
10. Enough information gathered to answer → writer_agent
11. Iteration >= {max_iter} → writer_agent immediately, no exceptions

RESPOND with EXACTLY ONE token — no punctuation, no explanation:
research_agent | rag_agent | news_agent | writer_agent | FINISH""".format(
    max_iter=MAX_ITERATIONS
)

REFLECTOR_PROMPT = """You are a quality assessor for an AI research pipeline.

Evaluate whether the latest agent output sufficiently addresses the required information for the query.

Respond with EXACTLY ONE line — no extra text:
SUFFICIENT          — output is useful, let supervisor decide next step
DONE                — enough information gathered, go directly to writer
RETRY: <new query>  — re-run the same agent but with this improved search query
ESCALATE: <agent>   — switch to this agent instead (research_agent | rag_agent | news_agent)

Be decisive. One line only."""

EVALUATOR_PROMPT = """You are a final answer quality checker.

Check whether the writer's answer completely and accurately addresses the original question and all sub-goals.

Respond with EXACTLY ONE line:
APPROVED              — answer is complete and accurate, ready to show the user
REVISE: <what is missing> — answer is incomplete, specify exactly what information is still needed

One line only. Be specific if requesting revision."""

RAG_PROMPT = (
    "You extract information from documents. "
    "Use ONLY the text inside <document> tags. "
    "Never use training data or invented examples. "
    "If a fact is absent, say: Not found in document."
)

WRITER_PROMPT = """You are a Senior Writer and Answer Synthesiser.
Compile the original question and all research notes into a clear markdown answer.

CRITICAL RULES:
- If research notes exist, your answer MUST come from those notes — do not contradict or ignore them.
- Do NOT add facts not present in the research notes.
- Open with the direct answer, then add supporting detail and evidence.
- Cite sources inline where available.
- Only if NO research notes exist: answer from built-in knowledge."""


# ── Metrics helper ────────────────────────────────────────────────────────────

def _llm_with_metrics(llm, agent_name: str, query_preview: str):
    temperature = getattr(llm, "temperature", 0.3)
    handler = MetricsCallbackHandler(
        agent_name=agent_name,
        temperature=temperature,
        query_preview=query_preview,
    )
    return llm.with_config({"callbacks": [handler]})


def _get_query(state: dict) -> str:
    for m in state.get("messages", []):
        if isinstance(m, HumanMessage):
            return str(m.content).strip()
    return ""


# ── Node class ────────────────────────────────────────────────────────────────

class AgenticNodes:
    def __init__(self, llm):
        self.llm = llm

    # ── Supervisor ────────────────────────────────────────────────────────────
    def supervisor(self, state: dict) -> dict:
        iteration = state.get("iteration", 0)
        query = _get_query(state)
        called = list(state.get("called_agents", []))

        if iteration >= MAX_ITERATIONS:
            return {**state, "next_agent": "writer_agent", "iteration": iteration + 1}

        # RAG answer is the final answer — skip writer to avoid hallucination
        if "rag_agent" in called and state.get("rag_context_text"):
            return {**state, "next_agent": "FINISH", "iteration": iteration + 1}

        llm = _llm_with_metrics(self.llm, "supervisor", query[:120])

        has_file = state.get("rag_file") is not None
        rag_note = (
            f"rag_file IS available ('{state.get('rag_file', 'document')}') — rag_agent CAN be used."
            if has_file
            else "rag_file NOT available — do NOT call rag_agent."
        )
        called_str = ", ".join(called) if called else "none"
        scratchpad = state.get("agent_scratchpad", [])
        context_so_far = "\n\n".join(scratchpad) if scratchpad else "Nothing gathered yet."

        history = state.get("conversation_history") or []
        history_block = (
            "--- Previous conversation ---\n" + "\n\n".join(history[-4:]) + "\n---\n\n"
            if history else ""
        )

        plan = state.get("plan") or ""
        sub_goals = state.get("sub_goals") or []
        reflection = state.get("reflection") or ""
        evaluation = state.get("evaluation") or ""

        plan_block = f"--- Execution plan ---\n{plan}\n---\n\n" if plan else ""
        sub_goals_block = (
            "Sub-goals:\n" + "\n".join(f"  • {g}" for g in sub_goals) + "\n\n"
            if sub_goals else ""
        )
        reflection_block = f"Reflection feedback: {reflection}\n\n" if reflection else ""
        evaluation_block = f"Evaluator feedback: {evaluation}\n\n" if evaluation else ""

        prompt = (
            f"{history_block}"
            f"{plan_block}"
            f"{sub_goals_block}"
            f"{reflection_block}"
            f"{evaluation_block}"
            f"Iteration: {iteration + 1} / {MAX_ITERATIONS}\n"
            f"Current query: {query}\n"
            f"Already called: {called_str}\n"
            f"RAG status: {rag_note}\n\n"
            f"--- Gathered so far ---\n{context_so_far}\n-----------------------\n\n"
            "Which agent should run next?"
        )

        response = llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=prompt),
        ])

        decision = _strip_thinking(response.content).strip().lower().split()[0]
        valid = {"research_agent", "rag_agent", "news_agent", "writer_agent", "finish"}
        if decision not in valid:
            decision = "writer_agent"
        if decision in called and decision != "writer_agent":
            decision = "writer_agent"

        next_agent = "FINISH" if decision == "finish" else decision
        return {**state, "next_agent": next_agent, "iteration": iteration + 1}

    # ── Research Agent (manual ReAct) ─────────────────────────────────────────
    def research_agent(self, state: dict) -> dict:
        query = _get_query(state)
        llm = _llm_with_metrics(self.llm, "research_agent", query[:120])
        scratchpad = list(state.get("agent_scratchpad", []))
        called = list(state.get("called_agents", [])) + ["research_agent"]

        try:
            answer, searches = _react_loop(
                llm=llm,
                query=query,
                role="Research Specialist",
                search_fn=_search_web,
            )
            label = f"(searched {len(searches)}x: {'; '.join(searches[:3])})" if searches else "(no searches)"
            scratchpad.append(f"[research_agent {label}]: {answer}")
            new_msg = AIMessage(content=f"**Research findings:**\n{answer}")
        except Exception as exc:
            scratchpad.append(f"[research_agent error]: {exc}")
            new_msg = AIMessage(content=f"Research agent error: {exc}")

        return {
            **state,
            "messages": state["messages"] + [new_msg],
            "agent_scratchpad": scratchpad,
            "called_agents": called,
        }

    # ── News Agent (manual ReAct) ─────────────────────────────────────────────
    def news_agent(self, state: dict) -> dict:
        query = _get_query(state)
        llm = _llm_with_metrics(self.llm, "news_agent", query[:120])
        scratchpad = list(state.get("agent_scratchpad", []))
        called = list(state.get("called_agents", [])) + ["news_agent"]

        try:
            answer, searches = _react_loop(
                llm=llm,
                query=query,
                role="AI/ML News Analyst",
                search_fn=_search_news,
            )
            label = f"(searched {len(searches)}x: {'; '.join(searches[:3])})" if searches else "(no searches)"
            scratchpad.append(f"[news_agent {label}]: {answer}")
            new_msg = AIMessage(content=f"**AI News Summary:**\n{answer}")
        except Exception as exc:
            scratchpad.append(f"[news_agent error]: {exc}")
            new_msg = AIMessage(content=f"News agent error: {exc}")

        return {
            **state,
            "messages": state["messages"] + [new_msg],
            "agent_scratchpad": scratchpad,
            "called_agents": called,
        }

    # ── RAG Agent (retrieve → extract) ───────────────────────────────────────
    def rag_agent(self, state: dict) -> dict:
        import streamlit as st
        from ..rag.retriever import build_or_get_index_from_session, search

        query = _get_query(state)
        scratchpad = list(state.get("agent_scratchpad", []))
        called = list(state.get("called_agents", [])) + ["rag_agent"]

        if not st.session_state.get("rag_file_bytes"):
            msg = "No document uploaded — upload a PDF or TXT in the sidebar first."
            scratchpad.append(f"[rag_agent]: {msg}")
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=msg)],
                "agent_scratchpad": scratchpad,
                "called_agents": called,
            }

        file_name = st.session_state.get("rag_file_name", "document")

        try:
            index = build_or_get_index_from_session()
            total_chunks = index.total_chunks

            # Short docs: retrieve every chunk; long docs: top-12 by BM25.
            k = total_chunks if total_chunks <= 30 else 12
            context = search(index, query, k=k)

            st.session_state["_rag_last_context"] = context
            st.session_state["_rag_last_query"] = query

            # Gemini is used exclusively for RAG answers — it follows grounding
            # instructions reliably and doesn't hallucinate from training priors.
            # All other agents (supervisor, research, news, writer) use Groq.
            import os
            from langchain_google_genai import ChatGoogleGenerativeAI
            rag_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0,
            )

            handler = MetricsCallbackHandler(
                agent_name="rag_agent",
                temperature=0,
                query_preview=query[:120],
            )
            system_msg = (
                "You are a document Q&A assistant. "
                "Answer using ONLY the text inside the <document> tags. "
                "Never use prior knowledge or invent information. "
                "If the answer is not in the document, say: Not found in document."
            )
            user_msg = (
                f'<document name="{file_name}">\n'
                f"{context}\n"
                f"</document>\n\n"
                f"Question: {query}\n\n"
                f"Answer using only the document above. Copy exact values verbatim."
            )
            response = rag_llm.with_config({"callbacks": [handler]}).invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ])

            answer = _strip_thinking(response.content).strip()
            if not answer:
                answer = "No answer generated — check the retrieved sections in the expander below."
            scratchpad.append(f"[rag_agent ('{file_name}')]: {answer}")
            new_msg = AIMessage(content=f"**From document — {file_name}:**\n\n{answer}")
            return {
                **state,
                "messages": state["messages"] + [new_msg],
                "agent_scratchpad": scratchpad,
                "called_agents": called,
                "rag_context_text": answer,
            }
        except Exception as exc:
            scratchpad.append(f"[rag_agent error]: {exc}")
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=f"Document agent error: {exc}")],
                "agent_scratchpad": scratchpad,
                "called_agents": called,
            }

    # ── Writer Agent ──────────────────────────────────────────────────────────
    def writer_agent(self, state: dict) -> dict:
        query = _get_query(state)
        llm = _llm_with_metrics(self.llm, "writer_agent", query[:120])

        scratchpad = state.get("agent_scratchpad", [])
        called = state.get("called_agents", [])

        history = state.get("conversation_history") or []
        history_block = (
            "--- Prior conversation context ---\n" + "\n\n".join(history[-3:]) + "\n---\n\n"
            if history else ""
        )

        if scratchpad:
            agents_used = ", ".join(called) if called else "none"
            research_block = f"Research by: {agents_used}\n\n" + "\n\n".join(scratchpad)
        else:
            research_block = "No external research — answer from built-in knowledge."

        response = llm.invoke([
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(
                content=(
                    f"{history_block}"
                    f"Current question: {query}\n\n"
                    f"--- Research notes ---\n{research_block}\n----------------------\n\n"
                    "Write the final, polished answer. "
                    "If the user is asking a follow-up, acknowledge the prior context naturally."
                )
            ),
        ])

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=_strip_thinking(response.content))],
            "next_agent": "FINISH",
        }

    # ── Planner ───────────────────────────────────────────────────────────────
    def planner(self, state: dict) -> dict:
        """Decomposes the query into an ordered execution plan and sub-goals."""
        query = _get_query(state)
        has_file = state.get("rag_file") is not None
        history = state.get("conversation_history") or []

        history_block = (
            "Prior conversation:\n" + "\n".join(history[-2:]) + "\n\n"
            if history else ""
        )
        rag_note = (
            f"An uploaded document IS available ('{state.get('rag_file')}')."
            if has_file else "No uploaded document."
        )

        llm = _llm_with_metrics(self.llm, "planner", query[:120])
        response = llm.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=(
                f"{history_block}"
                f"Document status: {rag_note}\n\n"
                f"User query: {query}"
            )),
        ])

        plan_text = _strip_thinking(response.content).strip()

        # Parse SUB_GOALS lines
        sub_goals = []
        in_sub = False
        for line in plan_text.splitlines():
            if line.strip().upper().startswith("SUB_GOALS"):
                in_sub = True
                continue
            if in_sub and line.strip().startswith("-"):
                sub_goals.append(line.strip().lstrip("-").strip())
            elif in_sub and line.strip() and not line.strip().startswith("-"):
                in_sub = False

        return {
            **state,
            "plan": plan_text,
            "sub_goals": sub_goals,
            "reflection": None,
            "reflection_count": 0,
            "evaluation": None,
        }

    # ── Reflector ─────────────────────────────────────────────────────────────
    def reflector(self, state: dict) -> dict:
        """Evaluates the last agent's output quality and guides the supervisor."""
        query = _get_query(state)
        scratchpad = state.get("agent_scratchpad", [])
        called = state.get("called_agents", [])
        plan = state.get("plan", "")
        sub_goals = state.get("sub_goals") or []
        reflection_count = state.get("reflection_count", 0)

        latest_output = scratchpad[-1] if scratchpad else "No output yet."
        sub_goals_str = "\n".join(f"- {g}" for g in sub_goals) or "None."

        llm = _llm_with_metrics(self.llm, "reflector", query[:120])
        response = llm.invoke([
            SystemMessage(content=REFLECTOR_PROMPT),
            HumanMessage(content=(
                f"Original query: {query}\n\n"
                f"Plan:\n{plan}\n\n"
                f"Sub-goals:\n{sub_goals_str}\n\n"
                f"Agents called so far: {', '.join(called) or 'none'}\n\n"
                f"Latest agent output:\n{latest_output}"
            )),
        ])

        reflection = _strip_thinking(response.content).strip()
        new_scratchpad = list(scratchpad) + [f"[reflector]: {reflection}"]

        return {
            **state,
            "reflection": reflection,
            "reflection_count": reflection_count + 1,
            "agent_scratchpad": new_scratchpad,
        }

    # ── Evaluator ─────────────────────────────────────────────────────────────
    def evaluator(self, state: dict) -> dict:
        """Checks the writer's final answer — approves or requests revision."""
        query = _get_query(state)
        plan = state.get("plan", "")
        sub_goals = state.get("sub_goals") or []

        final_answer = next(
            (m.content for m in reversed(state.get("messages", []))
             if isinstance(m, AIMessage) and m.content),
            "No answer generated."
        )

        sub_goals_str = "\n".join(f"- {g}" for g in sub_goals) or "None."

        llm = _llm_with_metrics(self.llm, "evaluator", query[:120])
        response = llm.invoke([
            SystemMessage(content=EVALUATOR_PROMPT),
            HumanMessage(content=(
                f"Original query: {query}\n\n"
                f"Plan:\n{plan}\n\n"
                f"Sub-goals to address:\n{sub_goals_str}\n\n"
                f"Writer's answer:\n{final_answer}"
            )),
        ])

        evaluation = _strip_thinking(response.content).strip()
        new_scratchpad = list(state.get("agent_scratchpad", [])) + [f"[evaluator]: {evaluation}"]

        return {
            **state,
            "evaluation": evaluation,
            "agent_scratchpad": new_scratchpad,
        }
