from langgraph.graph import StateGraph, START, END
from src.langgraphagenticai.state.state import State

from src.langgraphagenticai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraphagenticai.nodes.chatbot_with_Tool_node import ChatbotWithToolNode
from src.langgraphagenticai.nodes.ai_news_node import AINewsNode

from src.langgraphagenticai.tools.search_tool import get_tools, create_tool_node
from langgraph.prebuilt import tools_condition

# RAG imports
from src.langgraphagenticai.rag.retriever import rag_retrieve
from src.langgraphagenticai.rag.generator import rag_generate


class GraphBuilder:
    def __init__(self, model):
        self.llm = model

    # ---------------- BASIC CHATBOT ----------------
    def basic_chatbot_build_graph(self):
        graph = StateGraph(State)

        chatbot_node = BasicChatbotNode(self.llm)

        graph.add_node("chatbot", chatbot_node.process)
        graph.add_edge(START, "chatbot")
        graph.add_edge("chatbot", END)

        return graph.compile()

    # ---------------- CHATBOT WITH WEB ----------------
    def chatbot_with_tools_build_graph(self):
        graph = StateGraph(State)

        tools = get_tools()
        tool_node = create_tool_node(tools)

        chatbot_node = ChatbotWithToolNode(self.llm).create_chatbot(tools)

        graph.add_node("chatbot", chatbot_node)
        graph.add_node("tools", tool_node)

        graph.add_edge(START, "chatbot")
        graph.add_conditional_edges("chatbot", tools_condition)
        graph.add_edge("tools", "chatbot")

        return graph.compile()

    # ---------------- AI NEWS ----------------
    def ai_news_builder_graph(self):
        graph = StateGraph(State)

        ai_news_node = AINewsNode(self.llm)

        graph.add_node("fetch_news", ai_news_node.fetch_news)
        graph.add_node("summarize_news", ai_news_node.summarize_news)
        graph.add_node("save_result", ai_news_node.save_result)

        graph.add_edge(START, "fetch_news")
        graph.add_edge("fetch_news", "summarize_news")
        graph.add_edge("summarize_news", "save_result")
        graph.add_edge("save_result", END)

        return graph.compile()

    # ---------------- RAG ----------------
    def rag_builder_graph(self):
        graph = StateGraph(State)

        graph.add_node("rag_retrieve", rag_retrieve)
        graph.add_node(
            "rag_generate",
            lambda state: rag_generate(state, self.llm)
        )

        graph.add_edge(START, "rag_retrieve")
        graph.add_edge("rag_retrieve", "rag_generate")
        graph.add_edge("rag_generate", END)

        return graph.compile()

    # ---------------- ROUTER ----------------
    def setup_graph(self, usecase: str):
        if usecase == "Basic Chatbot":
            return self.basic_chatbot_build_graph()

        elif usecase == "Chatbot With Web":
            return self.chatbot_with_tools_build_graph()

        elif usecase == "AI News":
            return self.ai_news_builder_graph()

        elif usecase == "RAG":
            return self.rag_builder_graph()

        else:
            raise ValueError(f"Unsupported usecase: {usecase}")
