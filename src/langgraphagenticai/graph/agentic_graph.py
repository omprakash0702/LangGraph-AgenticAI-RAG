from langgraph.graph import END, START, StateGraph

from ..nodes.agentic_nodes import MAX_ITERATIONS, AgenticNodes
from ..state.state import AgenticState

MAX_REFLECTIONS = 2


def build_agentic_graph(llm):
    nodes = AgenticNodes(llm)
    graph = StateGraph(AgenticState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("planner",        nodes.planner)
    graph.add_node("supervisor",     nodes.supervisor)
    graph.add_node("research_agent", nodes.research_agent)
    graph.add_node("rag_agent",      nodes.rag_agent)
    graph.add_node("news_agent",     nodes.news_agent)
    graph.add_node("reflector",      nodes.reflector)
    graph.add_node("writer_agent",   nodes.writer_agent)
    graph.add_node("evaluator",      nodes.evaluator)

    # ── Entry: plan before routing ────────────────────────────────────────────
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "supervisor")

    # ── Supervisor → specialist ───────────────────────────────────────────────
    def route_supervisor(state: dict) -> str:
        next_a = state.get("next_agent", "writer_agent")
        if next_a == "FINISH":
            return END
        valid = {"research_agent", "rag_agent", "news_agent", "writer_agent"}
        return next_a if next_a in valid else "writer_agent"

    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research_agent": "research_agent",
            "rag_agent":      "rag_agent",
            "news_agent":     "news_agent",
            "writer_agent":   "writer_agent",
            END:              END,
        },
    )

    # ── Specialists → reflector (self-reflection after every worker) ──────────
    graph.add_edge("research_agent", "reflector")
    graph.add_edge("rag_agent",      "reflector")
    graph.add_edge("news_agent",     "reflector")

    # ── Reflector → supervisor or writer ─────────────────────────────────────
    def route_reflector(state: dict) -> str:
        reflection = (state.get("reflection") or "").upper()
        reflection_count = state.get("reflection_count", 0)
        iteration = state.get("iteration", 0)

        # Safety guards — don't loop forever
        if reflection_count >= MAX_REFLECTIONS or iteration >= MAX_ITERATIONS:
            return "writer_agent"

        if reflection.startswith("DONE"):
            return "writer_agent"

        # SUFFICIENT, RETRY, ESCALATE → all go back to supervisor
        # Supervisor reads the reflection field and acts accordingly
        return "supervisor"

    graph.add_conditional_edges(
        "reflector",
        route_reflector,
        {
            "supervisor":   "supervisor",
            "writer_agent": "writer_agent",
        },
    )

    # ── Writer → evaluator (self-correction check) ────────────────────────────
    graph.add_edge("writer_agent", "evaluator")

    # ── Evaluator → END or back to supervisor for revision ───────────────────
    def route_evaluator(state: dict) -> str:
        evaluation = (state.get("evaluation") or "").upper()
        iteration = state.get("iteration", 0)

        if iteration >= MAX_ITERATIONS:
            return END
        if evaluation.startswith("APPROVED"):
            return END
        # REVISE → supervisor researches the missing info
        return "supervisor"

    graph.add_conditional_edges(
        "evaluator",
        route_evaluator,
        {
            "supervisor": "supervisor",
            END:          END,
        },
    )

    return graph.compile()
