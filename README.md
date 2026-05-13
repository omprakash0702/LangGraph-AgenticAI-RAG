# LangGraph Agentic AI System

A production-grade multi-agent AI system built with LangGraph. The system autonomously plans, routes, researches, reflects, writes, and self-corrects — without any fixed pipeline logic baked in.

**Live Demo:** [https://langgraph-agenticai-rag-aiesh8dmqsdawjbus9glpy.streamlit.app](https://langgraph-agenticai-rag-aiesh8dmqsdawjbus9glpy.streamlit.app)

---

## Deployment

Deployed on **Streamlit Community Cloud** directly from the `v2-pure-agentic` branch of this repository. The entry point is `app.py` at the repo root, which loads the Streamlit UI from `src/langgraphagenticai/main.py`. API keys (Groq, Gemini, Tavily) are configured as secrets in the Streamlit Cloud dashboard and are never stored in the repository.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Architecture](#architecture)
4. [Agent Roles](#agent-roles)
5. [Agentic Execution Flow](#agentic-execution-flow)
6. [RAG Pipeline](#rag-pipeline)
7. [Retrieval Pipeline — Hybrid Search](#retrieval-pipeline--hybrid-search)
8. [Metrics & Observability](#metrics--observability)
9. [Project Structure](#project-structure)
10. [Setup & Running](#setup--running)
11. [State Schema](#state-schema)
12. [Key Design Decisions](#key-design-decisions)

---

## Project Overview

This system takes any user query and autonomously decides how to answer it:

- Decomposes the query into a step-by-step plan
- Routes to specialist agents (web research, document Q&A, AI news)
- Reflects on each agent's output — retries or escalates if quality is poor
- Synthesises all findings into a polished final answer
- Evaluates the final answer — loops back for revision if incomplete

The user interacts through a Streamlit chat interface with a live performance dashboard.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` |
| **LLM — agents** | [Groq](https://groq.com/) — `llama-3.3-70b-versatile`, `llama-4-scout`, `qwen3-32b` |
| **LLM — RAG answers** | [Google Gemini](https://ai.google.dev/) — `gemini-2.5-flash` (temperature 0) |
| **Embeddings** | Google Generative AI — `models/gemini-embedding-2` |
| **Vector store** | [FAISS](https://github.com/facebookresearch/faiss) (CPU) |
| **Keyword retrieval** | [BM25Okapi](https://github.com/dorianbrown/rank_bm25) |
| **Web search** | [Tavily](https://tavily.com/) (advanced + news modes) |
| **Document loading** | LangChain `PyPDFLoader`, `TextLoader` |
| **UI** | [Streamlit](https://streamlit.io/) |
| **Metrics storage** | SQLite via `MetricsCallbackHandler` |
| **Frameworks** | LangChain Core, LangChain Community, LangChain Google GenAI |

---

## Architecture

```
User Query
    │
    ▼
┌──────────┐
│  Planner │  Decomposes query into plan + sub-goals
└────┬─────┘
     │
     ▼
┌────────────┐
│ Supervisor │◄──────────────────────────────────────────────┐
└─────┬──────┘                                               │
      │  routes to one of:                                   │
      ├──────────────────┬──────────────────┐                │
      │                  │                  │                │
      ▼                  ▼                  ▼                │
┌─────────────┐  ┌──────────────┐  ┌────────────┐           │
│research_agent│  │  rag_agent   │  │ news_agent │           │
│  (ReAct +   │  │ (Hybrid RAG  │  │  (ReAct +  │           │
│  Tavily web)│  │ + Gemini LLM)│  │ Tavily news│           │
└──────┬──────┘  └──────┬───────┘  └─────┬──────┘           │
       └────────────────┴─────────────────┘                  │
                         │                                   │
                         ▼                                   │
                  ┌────────────┐                             │
                  │  Reflector │  quality check              │
                  └─────┬──────┘                             │
                        │                                    │
              ┌─────────┴──────────┐                        │
              ▼                    └────────────────────────►┘
       DONE / SUFFICIENT        RETRY / ESCALATE
              │
              ▼
       ┌─────────────┐
       │ writer_agent│  synthesises all findings
       └──────┬───────┘
              │
              ▼
       ┌───────────┐
       │ Evaluator │  checks completeness
       └─────┬─────┘
             │
     ┌───────┴────────┐
     ▼                ▼
   APPROVED      REVISE → back to Supervisor
     │
    END
```

---

## Agent Roles

### Planner `📋`

Receives the raw user query first. Outputs a structured plan:

- `COMPLEXITY` — simple / moderate / complex
- `PLAN` — ordered list of agents to call
- `GOAL` — one-sentence success criterion
- `SUB_GOALS` — specific questions that need to be answered

The planner's output guides the supervisor for the entire run. Simple queries (greetings, math, coding) are routed directly to `writer_agent`.

---

### Supervisor `🧠`

The central router. At each step it reads:

- The execution plan from the planner
- Reflection feedback from the last reflector call
- Evaluator feedback if the writer was rejected
- Which agents have already been called
- All findings accumulated in the scratchpad

It responds with exactly one token: `research_agent | rag_agent | news_agent | writer_agent | FINISH`

**Decision priority (top-down, first match):**

1. Follow the planner's execution plan
2. Act on reflector feedback (RETRY / ESCALATE)
3. Act on evaluator feedback (REVISE)
4. Small talk / math / coding → writer directly
5. Never call the same agent twice (unless reflector says RETRY)
6. RAG file available + document question → rag_agent
7. AI/ML news query → news_agent
8. General factual query → research_agent
9. Enough info gathered → writer_agent

---

### Research Agent `🔍`

Runs a **manual ReAct loop** against Tavily web search:

1. LLM decides what to search (`SEARCH: <query>`)
2. Tavily returns ranked results + a direct answer
3. LLM decides: search more or output `DONE` + final answer
4. Up to 4 search iterations per call

Uses `advanced` search depth for higher-quality results.

---

### News Agent `📰`

Same ReAct loop as the research agent but calls Tavily's `news` topic endpoint, filtered to the past 7 days. Specialised for AI/ML news, model releases, and industry events.

---

### RAG Agent `📄`

Document Q&A in two stages:

1. **Retrieve** — hybrid BM25 + FAISS search (see RAG Pipeline section)
2. **Answer** — Gemini 2.5 Flash at temperature 0 with XML-grounded prompting

The answer LLM is constrained to quote only from `<document>` tags. If information is absent it outputs "Not found in document" rather than hallucinating.

---

### Reflector `🔄`

Reads the last agent's output and outputs one of:

| Verdict | Meaning |
|---|---|
| `SUFFICIENT` | Output is useful — supervisor decides next step |
| `DONE` | Enough gathered — go straight to writer |
| `RETRY: <new query>` | Re-run the same agent with a better search query |
| `ESCALATE: <agent>` | Switch to a different agent instead |

Capped at **2 reflection cycles** to prevent infinite loops.

---

### Writer Agent `✍️`

Receives the full scratchpad (all agent findings) and compiles them into a structured markdown answer.

Rules:
- Must use research notes if they exist — cannot contradict them
- Cannot add facts not present in the notes
- Only answers from built-in knowledge if no research was done

---

### Evaluator `✅`

Checks the writer's answer against the original query and all sub-goals from the planner.

| Verdict | Meaning |
|---|---|
| `APPROVED` | Answer is complete — go to END |
| `REVISE: <what is missing>` | Loops back to supervisor to research the gap |

Capped by the global `MAX_ITERATIONS = 5` guard.

---

## Agentic Execution Flow

### Example 1 — Research query

**Query:** "What are the latest breakthroughs in LLM reasoning?"

```
1. Planner
   COMPLEXITY: moderate
   PLAN: news_agent, research_agent, writer_agent
   SUB_GOALS:
     - What new reasoning techniques were released recently?
     - Which models demonstrate improved reasoning?

2. Supervisor → news_agent

3. news_agent (ReAct)
   SEARCH: LLM reasoning breakthroughs 2025
   SEARCH: chain-of-thought improvements recent models
   DONE → findings appended to scratchpad

4. Reflector → SUFFICIENT

5. Supervisor → research_agent  (plan calls for research after news)

6. research_agent (ReAct)
   SEARCH: latest reasoning model benchmarks 2025
   DONE → findings appended to scratchpad

7. Reflector → DONE  (enough gathered)

8. writer_agent
   Compiles news + research into structured markdown answer

9. Evaluator → APPROVED  (both sub-goals addressed)

END
```

---

### Example 2 — Document Q&A

**Query:** "What is the total amount in my invoice?" (PDF uploaded)

```
1. Planner
   COMPLEXITY: simple
   PLAN: rag_agent, writer_agent

2. Supervisor → rag_agent

3. rag_agent
   BM25: keyword search "total amount invoice"
   FAISS: semantic search "total amount invoice"
   RRF fusion → top chunks retrieved
   Gemini 2.5 Flash reads <document> tags
   Returns exact value verbatim from PDF

4. Supervisor checks rag_context_text → FINISH
   (skips writer to preserve Gemini's grounded answer)

END
```

---

### Example 3 — Self-correction loop

**Query:** "Compare GPT-4o and Claude 3.5 Sonnet on coding benchmarks"

```
1. Planner
   COMPLEXITY: complex
   PLAN: research_agent, writer_agent

2. Supervisor → research_agent

3. research_agent → finds GPT-4o data only

4. Reflector → RETRY: "Claude 3.5 Sonnet coding benchmark HumanEval"

5. Supervisor → research_agent (again, with new query)

6. research_agent → finds Claude data

7. Reflector → DONE

8. writer_agent → writes comparison

9. Evaluator → REVISE: "Missing specific benchmark scores for SWE-bench"

10. Supervisor → research_agent (researches the gap)

11. research_agent → finds SWE-bench scores

12. writer_agent → rewrites with complete data

13. Evaluator → APPROVED

END
```

---

## RAG Pipeline

```
PDF / TXT file
      │
      ▼
 PyPDFLoader / TextLoader
 (UTF-8 fallback to latin-1)
      │
      ▼
 RecursiveCharacterTextSplitter
 chunk_size=300, overlap=60
      │
      ├──────────────────────────────────────┐
      │                                      │
      ▼                                      ▼
 Google Gemini Embeddings              BM25Okapi tokenizer
 (gemini-embedding-2)                  (regex tokenisation)
 embedded one-at-a-time
      │
      ▼
 FAISS vector store
 (from_embeddings — manual build
  to bypass batch API limitation)
      │
      └──────────────────────────────────────┘
                        │
                   RAGIndex dataclass
                   (db, chunks, bm25)
                        │
                   Cached in session_state
                   (rebuilt only on new upload)
```

---

## Retrieval Pipeline — Hybrid Search

At query time both indexes run independently then are fused:

```
Query
  │
  ├── BM25 keyword search    catches: exact values, codes, names, numbers
  │   top-k by BM25 score
  │
  └── FAISS semantic search  catches: paraphrased questions, synonyms
      top-k by cosine distance
            │
            ▼
  Reciprocal Rank Fusion (K=60)
  score(i) = 1/(60 + bm25_rank) + 1/(60 + faiss_rank)
            │
            ▼
  Top-k chunks merged, ranked, deduplicated
            │
            ▼
  Formatted with section numbers + page numbers
            │
            ▼
  Gemini 2.5 Flash answers from <document> tags only
```

**Why RRF instead of score averaging?** BM25 scores and cosine distances live on different scales. RRF uses rank position instead of raw score, so neither retriever dominates regardless of scale.

**Short vs long documents:** Documents with ≤30 chunks retrieve all chunks. Longer documents retrieve top 12 by hybrid score.

---

## Metrics & Observability

Every LLM call in every agent is instrumented via `MetricsCallbackHandler` (a LangChain `BaseCallbackHandler`). Each call records:

| Field | Description |
|---|---|
| `agent_name` | Which agent made the call |
| `model` | Exact model ID returned by the API |
| `temperature` | Sampling temperature used |
| `input_tokens` | Prompt token count |
| `output_tokens` | Completion token count |
| `latency_ms` | Wall-clock time for the LLM call |
| `success` | Whether the call succeeded or errored |
| `query_preview` | First 120 chars of the query |
| `timestamp` | ISO-8601 datetime |

Data persists in SQLite and is visualised in the **Dashboard tab**:

- Token usage per agent (bar chart)
- Latency distribution (histogram)
- Success/error rate (pie chart)
- Raw call log (scrollable table)

---

## Project Structure

```
LANGRAPH/
├── src/langgraphagenticai/
│   ├── graph/
│   │   ├── agentic_graph.py      Full agentic StateGraph definition
│   │   └── graph_builder.py      Router — maps use case name to graph
│   ├── nodes/
│   │   └── agentic_nodes.py      All 7 agent node implementations + ReAct loop
│   ├── rag/
│   │   ├── retriever.py          Hybrid BM25+FAISS index + RRF search
│   │   └── generator.py          Legacy RAG generate node
│   ├── state/
│   │   └── state.py              AgenticState TypedDict definition
│   ├── metrics/
│   │   ├── callbacks.py          LangChain callback handler → SQLite
│   │   └── tracker.py            SQLite schema + insert helpers
│   ├── ui/streamlitui/
│   │   ├── loadui.py             Sidebar config (model, API keys, file upload)
│   │   ├── display_result.py     Result rendering helpers
│   │   └── dashboard.py          Plotly metrics dashboard
│   ├── LLMS/
│   │   └── groqllm.py            Groq model initialisation
│   └── main.py                   Streamlit app entry point + graph cache
├── AINews/
│   └── monthly_summary.md        Saved AI news summaries
├── requirements.txt
└── README.md
```

---

## Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

API keys can also be entered in the sidebar at runtime.

### 3. Run

```bash
streamlit run src/langgraphagenticai/main.py
```

### 4. Available Groq models

| Model | Best for |
|---|---|
| `llama-3.3-70b-versatile` | General purpose, highest quality |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Fast, efficient |
| `qwen/qwen3-32b` | Strong reasoning, thinking mode |
| `llama-3.1-8b-instant` | Lowest latency |

---

## State Schema

`AgenticState` is the single TypedDict that flows through the entire graph:

```python
class AgenticState(TypedDict):
    # Core
    messages: List                   # Full conversation message list
    rag_file: Optional[Any]          # Uploaded file name (signals RAG availability)
    rag_context_text: Optional[str]  # Retrieved + formatted document chunks

    # Routing
    next_agent: str                  # Supervisor's routing decision
    called_agents: List[str]         # Deduplication guard
    iteration: int                   # Global step counter (MAX = 5)

    # Memory
    agent_scratchpad: List[str]      # All agent outputs accumulated
    conversation_history: List[str]  # Prior Q&A turns (last 4)

    # Agentic fields
    plan: Optional[str]              # Planner's ordered execution plan
    sub_goals: Optional[List[str]]   # Decomposed sub-questions
    reflection: Optional[str]        # Reflector's last verdict
    reflection_count: int            # Reflection loop guard (MAX = 2)
    evaluation: Optional[str]        # Evaluator's verdict on writer output
```

---

## Key Design Decisions

**Why two LLMs (Groq + Gemini)?**
Groq with Llama 3.3 70B is fast and inexpensive for routing, research synthesis, and writing. Gemini 2.5 Flash is used exclusively for RAG answers because it follows XML grounding instructions more reliably and produces significantly lower hallucination rates on document-extraction tasks at temperature 0.

**Why manual ReAct instead of `bind_tools()`?**
The ReAct loop is implemented with a `SEARCH:` / `DONE` text protocol rather than LangChain's tool-binding API. This keeps all control logic visible in plain Python, avoids schema boilerplate, and works identically across all Groq models regardless of their tool-calling support.

**Why BM25 + FAISS instead of vector search alone?**
Vector search misses exact values — invoice numbers, dates, amounts, proper names — because they have no semantic neighbours. BM25 catches exact token matches. RRF fusion combines both without requiring score normalisation, giving the best of keyword and semantic retrieval.

**Why skip the writer for RAG-only queries?**
After `rag_agent` runs, the supervisor detects `rag_context_text` is set and returns `FINISH` directly. Passing Gemini's grounded answer through the Groq-powered `writer_agent` would risk overriding exact document values with hallucinated details from the LLM's training prior.

**Why `gemini-embedding-2` with one-at-a-time embedding?**
The Gemini embedding API does not support batch embedding for this model tier. Each chunk is embedded individually via `embed_query()`, then the FAISS index is built manually using `FAISS.from_embeddings()` with pre-computed vectors.
