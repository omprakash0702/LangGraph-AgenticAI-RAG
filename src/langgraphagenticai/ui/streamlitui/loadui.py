import streamlit as st
import os
from src.langgraphagenticai.ui.uiconfigfile import Config


class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        st.set_page_config(
            page_title="🤖 " + self.config.get_page_title(),
            layout="wide"
        )
        st.header("🤖 " + self.config.get_page_title())

        # -------- SAFE SESSION STATE INIT --------
        st.session_state.setdefault("timeframe", "")
        st.session_state.setdefault("IsFetchButtonClicked", False)
        st.session_state.setdefault("rag_file", None)

        with st.sidebar:
            # -------- OPTIONS --------
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            self.user_controls["selected_llm"] = st.selectbox(
                "Select LLM", llm_options
            )

            # -------- GROQ --------
            if self.user_controls["selected_llm"] == "Groq":
                self.user_controls["selected_groq_model"] = st.selectbox(
                    "Select Model",
                    self.config.get_groq_model_options()
                )

                self.user_controls["GROQ_API_KEY"] = (
                    st.text_input(
                        "GROQ API Key (leave empty to use .env)",
                        type="password"
                    ) or os.getenv("GROQ_API_KEY")
                )

            # -------- USECASE --------
            self.user_controls["selected_usecase"] = st.selectbox(
                "Select Usecases",
                usecase_options
            )

            # -------- TAVILY --------
            if self.user_controls["selected_usecase"] in [
                "Chatbot With Web",
                "AI News"
            ]:
                self.user_controls["TAVILY_API_KEY"] = (
                    st.text_input(
                        "TAVILY API Key (leave empty to use .env)",
                        type="password"
                    ) or os.getenv("TAVILY_API_KEY")
                )

            # -------- AI NEWS --------
            if self.user_controls["selected_usecase"] == "AI News":
                st.subheader("📰 AI News Explorer")

                time_frame = st.selectbox(
                    "📅 Select Time Frame",
                    ["Daily", "Weekly", "Monthly"],
                    index=0
                )

                if st.button("🔍 Fetch Latest AI News", use_container_width=True):
                    st.session_state.IsFetchButtonClicked = True
                    st.session_state.timeframe = time_frame

            # -------- ✅ RAG FILE UPLOAD (FIX) --------
            if self.user_controls["selected_usecase"] == "RAG":
                st.subheader("📄 Upload Document for RAG")

                uploaded_file = st.file_uploader(
                    "Upload a document (.txt or .pdf)",
                    type=["txt", "pdf"]
                )

                # 🔑 Persist file across reruns
                if uploaded_file is not None:
                    st.session_state.rag_file = uploaded_file

                self.user_controls["rag_file"] = st.session_state.rag_file

        return self.user_controls
