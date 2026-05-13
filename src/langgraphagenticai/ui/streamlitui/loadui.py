import os

import streamlit as st

from src.langgraphagenticai.ui.uiconfigfile import Config

_MODEL_LABELS = {
    "llama-3.3-70b-versatile":                      "Llama 3.3 70B — best quality (recommended)",
    "meta-llama/llama-4-scout-17b-16e-instruct":    "Llama 4 Scout 17B — fast + capable",
    "qwen/qwen3-32b":                               "Qwen 3 32B — strong reasoning",
    "llama-3.1-8b-instant":                         "Llama 3.1 8B — fastest / cheapest",
}


class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        st.set_page_config(
            page_title="🤖 " + self.config.get_page_title(),
            layout="wide",
        )

        st.session_state.setdefault("rag_file", None)

        with st.sidebar:
            st.markdown("## ⚙️ Configuration")

            # ── LLM provider ─────────────────────────────────────────────────
            llm_options = self.config.get_llm_options()
            self.user_controls["selected_llm"] = st.selectbox("LLM Provider", llm_options)

            if self.user_controls["selected_llm"] == "Groq":
                model_ids = self.config.get_groq_model_options()
                label_to_id = {_MODEL_LABELS.get(m, m): m for m in model_ids}
                selected_label = st.selectbox("Model", list(label_to_id.keys()))
                self.user_controls["selected_groq_model"] = label_to_id[selected_label]

                groq_key = (
                    st.text_input(
                        "Groq API Key (leave blank to use .env)",
                        type="password",
                    )
                    or os.getenv("GROQ_API_KEY", "")
                )
                self.user_controls["GROQ_API_KEY"] = groq_key
                if not groq_key:
                    st.error("Groq API key required — add to .env or enter above.")

            # ── Temperature ───────────────────────────────────────────────────
            self.user_controls["temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Lower = more factual. Higher = more creative.",
            )

            st.markdown("---")

            # ── Tavily key ────────────────────────────────────────────────────
            tavily_key = (
                st.text_input(
                    "Tavily API Key (leave blank to use .env)",
                    type="password",
                )
                or os.getenv("TAVILY_API_KEY", "")
            )
            self.user_controls["TAVILY_API_KEY"] = tavily_key
            if tavily_key:
                os.environ["TAVILY_API_KEY"] = tavily_key
            elif not os.getenv("TAVILY_API_KEY"):
                st.warning("Tavily key missing — web search won't work.")

            st.markdown("---")

            # ── Document upload ───────────────────────────────────────────────
            st.markdown("#### 📄 Document Q&A")
            st.caption("Upload a PDF or TXT. **TXT recommended** — PDFs may extract incorrectly.")

            uploaded_file = st.file_uploader(
                "Upload .txt or .pdf", type=["txt", "pdf"], label_visibility="collapsed"
            )

            if uploaded_file is not None:
                old_name = st.session_state.get("rag_file_name")
                old_size = st.session_state.get("rag_file_size", -1)
                if old_name != uploaded_file.name or old_size != uploaded_file.size:
                    # Read bytes immediately — UploadedFile becomes stale after this run
                    st.session_state["rag_file_bytes"] = uploaded_file.read()
                    st.session_state["rag_file_name"] = uploaded_file.name
                    st.session_state["rag_file_size"] = uploaded_file.size
                    st.session_state.pop("_rag_cache", None)
                st.session_state.rag_file = uploaded_file  # kept only for display

            self.user_controls["rag_file"] = st.session_state.get("rag_file_name")

            if st.session_state.get("rag_file_name"):
                fname = st.session_state["rag_file_name"]
                size_kb = round(st.session_state.get("rag_file_size", 0) / 1024, 1)
                st.success(f"✅ **{fname}** ({size_kb} KB) — ready")
                st.caption("Ask anything about this document in the chat.")

                if st.button("🗑️ Remove document", width='stretch'):
                    for _k in ("rag_file", "rag_file_name", "rag_file_bytes", "rag_file_size", "_rag_cache"):
                        st.session_state.pop(_k, None)
                    self.user_controls["rag_file"] = None
                    st.rerun()
            else:
                st.info("No document uploaded yet.\nUpload a PDF or TXT to enable Document Q&A.", icon="📂")

            st.markdown("---")

            # ── Status legend ─────────────────────────────────────────────────
            st.markdown("**Agents available:**")
            st.markdown(
                "🔍 Web Research  \n"
                "📰 AI News  \n"
                + ("📄 Document Q&A ✅\n" if st.session_state.get("rag_file_name") else "📄 Document Q&A _(upload file)_\n") +
                "✍️ Writer / Synthesiser"
            )

        self.user_controls["selected_usecase"] = "Pure Agentic AI"
        return self.user_controls
