from typing_extensions import TypedDict, List
from langgraph.graph.message import add_messages
from typing import Annotated, Optional, Any


class State(TypedDict):
    """
    Represent the structure of the state used in graph
    """

    # Chat messages (LangGraph-managed)
    messages: Annotated[List, add_messages]

    # -------- RAG --------
    rag_file: Optional[Any]
    context: Optional[Any]
    rag_answer: Optional[str]
