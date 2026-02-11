import os
from langchain_groq import ChatGroq


class GroqLLM:
    def __init__(self, user_contols_input):
        self.user_controls_input = user_contols_input

    def get_llm_model(self):
        groq_api_key = (
            self.user_controls_input.get("GROQ_API_KEY")
            or os.getenv("GROQ_API_KEY")
        )

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found")

        selected_groq_model = self.user_controls_input.get(
            "selected_groq_model"
        )

        # 🚨 DO NOT validate model locally
        # Let Groq API decide what is valid

        return ChatGroq(
            api_key=groq_api_key,
            model=selected_groq_model,
            temperature=0.3
        )
