from typing import Annotated, Any, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    """State for the original single-agent graphs."""

    messages: Annotated[List, add_messages]
    rag_file: Optional[Any]
    context: Optional[Any]
    rag_answer: Optional[str]
    rag_context_text: Optional[str]
    # Used by AINewsNode
    news_data: Optional[List]
    summary: Optional[str]


class AgenticState(TypedDict):
    """State for the multi-agent supervisor graph."""

    messages: Annotated[List, add_messages]
    rag_file: Optional[Any]
    context: Optional[Any]
    rag_answer: Optional[str]

    # Multi-agent control fields
    next_agent: str
    agent_scratchpad: List[str]
    called_agents: List[str]
    iteration: int

    # Conversation memory — previous Q&A turns formatted as plain text
    conversation_history: Optional[List[str]]

    # RAG formatted context (set by retriever, used by generator + rag_agent)
    rag_context_text: Optional[str]

    # Agentic fields
    plan: Optional[str]            # Planner's step-by-step execution plan
    sub_goals: Optional[List[str]] # Decomposed sub-questions from planner
    reflection: Optional[str]      # Reflector's quality assessment of last agent output
    reflection_count: int          # Guard against infinite reflection loops
    evaluation: Optional[str]      # Evaluator's verdict on the writer's final answer
