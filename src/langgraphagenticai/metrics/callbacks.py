import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from .tracker import AgentMetric, MetricsDB


class MetricsCallbackHandler(BaseCallbackHandler):
    """Records per-LLM-call metrics (latency, tokens, temperature) to SQLite."""

    def __init__(self, agent_name: str, temperature: float = 0.3, query_preview: str = ""):
        super().__init__()
        self.agent_name = agent_name
        self.temperature = temperature
        self.query_preview = query_preview[:120]
        self.db = MetricsDB()
        self._start: Optional[float] = None
        self._model: str = "unknown"

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        self._start = time.time()
        kw = serialized.get("kwargs", {})
        self._model = kw.get("model_name", kw.get("model", "unknown"))

    def on_llm_end(self, response: Any, **kwargs):
        if self._start is None:
            return
        latency_ms = round((time.time() - self._start) * 1000, 2)

        input_t = output_t = total_t = 0
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            input_t = usage.get("prompt_tokens", 0)
            output_t = usage.get("completion_tokens", 0)
            total_t = usage.get("total_tokens", input_t + output_t)
            if self._model == "unknown":
                self._model = response.llm_output.get("model_name", "unknown")

        # Fallback: extract from generation_info
        if total_t == 0 and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    info = getattr(gen, "generation_info", {}) or {}
                    u = info.get("usage", {}) or info.get("token_usage", {})
                    input_t += u.get("prompt_tokens", 0)
                    output_t += u.get("completion_tokens", 0)
                    total_t += u.get("total_tokens", 0)

        self.db.insert(AgentMetric(
            agent_name=self.agent_name,
            model=self._model,
            temperature=self.temperature,
            input_tokens=input_t,
            output_tokens=output_t,
            total_tokens=total_t,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat(),
            success=True,
            query_preview=self.query_preview,
        ))

    def on_llm_error(self, error: Exception, **kwargs):
        latency_ms = round((time.time() - self._start) * 1000, 2) if self._start else 0.0
        self.db.insert(AgentMetric(
            agent_name=self.agent_name,
            model=self._model,
            temperature=self.temperature,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat(),
            success=False,
            query_preview=self.query_preview,
            error=str(error)[:500],
        ))
