import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit


def load_langgraph_agenticai_app():

    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return

    # -------- USER MESSAGE --------
    if st.session_state.IsFetchButtonClicked:
        user_message = st.session_state.timeframe
    else:
        user_message = st.chat_input("Enter your message:")

    if not user_message:
        return

    try:
        # -------- LLM --------
        obj_llm_config = GroqLLM(user_contols_input=user_input)
        model = obj_llm_config.get_llm_model()

        if not model:
            st.error("Error: LLM model could not be initialized")
            return

        # -------- USECASE --------
        usecase = user_input.get("selected_usecase")

        if not usecase:
            st.error("Error: No use case selected.")
            return

        # -------- BUILD GRAPH --------
        graph_builder = GraphBuilder(model)
        graph = graph_builder.setup_graph(usecase)

        # -------- BUILD INITIAL STATE (CRITICAL FIX) --------
        initial_state = {
            "messages": [HumanMessage(content=user_message)]
        }

        if usecase == "RAG":
            rag_file = st.session_state.get("rag_file")
            if rag_file is None:
                st.error("Please upload a document before asking a question.")
                return
            initial_state["rag_file"] = rag_file

        # -------- DISPLAY --------
        DisplayResultStreamlit(
            usecase,
            graph,
            initial_state
        ).display_result_on_ui()

    except Exception as e:
        st.error(f"Error: Graph set up failed- {e}")
