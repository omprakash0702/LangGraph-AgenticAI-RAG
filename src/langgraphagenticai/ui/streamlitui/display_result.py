import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class DisplayResultStreamlit:
    def __init__(self, usecase, graph, initial_state):
        self.usecase = usecase
        self.graph = graph
        self.initial_state = initial_state

    def display_result_on_ui(self):

        # ---------------- BASIC CHATBOT ----------------
        if self.usecase == "Basic Chatbot":
            for event in self.graph.stream(self.initial_state):
                for value in event.values():
                    with st.chat_message("assistant"):
                        st.write(value["messages"].content)

        # ---------------- CHATBOT WITH WEB ----------------
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

        # ---------------- AI NEWS ----------------
        elif self.usecase == "AI News":
            self.graph.invoke(self.initial_state)
            path = f"./AINews/{self.initial_state['messages'][0].content.lower()}_summary.md"
            with open(path, "r", encoding="utf-8") as f:
                st.markdown(f.read(), unsafe_allow_html=True)

        # ---------------- ✅ RAG ----------------
        elif self.usecase == "RAG":
            res = self.graph.invoke(self.initial_state)
            with st.chat_message("assistant"):
                st.write(res.get("rag_answer", "No answer found."))
