from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate


class AINewsNode:
    def __init__(self, llm):
        """
        Initialize the AINewsNode with Tavily client and LLM.
        """
        self.tavily = TavilyClient()
        self.llm = llm

        # Internal state to track steps
        self.state = {}

    def fetch_news(self, state: dict) -> dict:
        """
        Fetch AI news based on the specified frequency.
        """

        frequency = state["messages"][0].content.lower()
        self.state["frequency"] = frequency

        time_range_map = {
            "daily": "d",
            "weekly": "w",
            "monthly": "m",
            "year": "y",
        }

        days_map = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
            "year": 366,
        }

        response = self.tavily.search(
            query="Top Artificial Intelligence (AI) technology news India and globally",
            topic="news",
            time_range=time_range_map[frequency],
            include_answer="advanced",
            max_results=20,
            days=days_map[frequency],
        )

        state["news_data"] = response.get("results", [])
        self.state["news_data"] = state["news_data"]

        return state

    def summarize_news(self, state: dict) -> dict:
        """
        Summarize the fetched news using the LLM.
        """

        news_items = self.state["news_data"]

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Summarize AI news articles into markdown format.

Rules:
- Date in **YYYY-MM-DD** format (IST timezone)
- Concise summary of each article
- Sort by date (latest first)
- Include source URL as a markdown link

Format strictly as:
### YYYY-MM-DD
- [Summary](URL)
""",
                ),
                ("user", "Articles:\n{articles}"),
            ]
        )

        articles_str = "\n\n".join(
            [
                f"Content: {item.get('content', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Date: {item.get('published_date', '')}"
                for item in news_items
            ]
        )

        response = self.llm.invoke(
            prompt_template.format(articles=articles_str)
        )

        state["summary"] = response.content
        self.state["summary"] = state["summary"]

        return state

    def save_result(self, state: dict) -> dict:
        """
        Save the summarized news to a markdown file (UTF-8 safe).
        """

        frequency = self.state["frequency"]
        summary = self.state["summary"]

        filename = f"./AINews/{frequency}_summary.md"

        # ✅ CRITICAL FIX: force UTF-8 encoding (Windows-safe)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
            f.write(summary)

        self.state["filename"] = filename
        return self.state
