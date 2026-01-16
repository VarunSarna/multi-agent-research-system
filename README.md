# 🤖 Multi-Agent Research System - Documentation

## For Your Interview: What This Demo Shows

This demo proves you can build **production-grade AI agent systems**. It demonstrates:

1. **Multi-agent orchestration** (not just a single LLM call)
2. **Tool calling** (web search integration)
3. **RAG architecture** (retrieval-augmented generation)
4. **State management** (passing context between agents)
5. **Observability** (logging every step)
6. **Production patterns** (error handling, metrics)

---

## Quick Start

```bash
# 1. Install (only streamlit needed for demo mode)
pip install streamlit

# 2. Run
streamlit run app.py

# 3. Open http://localhost:8501
```

**That's it!** The demo works without API keys using realistic mock data.

### For Live Mode (Optional)

```bash
# Install all dependencies
pip install streamlit langgraph langchain langchain-anthropic chromadb tavily-python

# Set environment variables
export ANTHROPIC_API_KEY=your_key_here
export TAVILY_API_KEY=your_key_here

# Run
streamlit run app.py
```

---

## 🎓 Theory Explained (For Your Interview)

### What is an AI Agent?

An **AI Agent** is an LLM that can:
1. **Reason** about what to do
2. **Use tools** (APIs, databases, search)
3. **Take actions** based on reasoning
4. **Observe results** and iterate

**Simple chatbot**: User → LLM → Response  
**AI Agent**: User → LLM → [Tools/Actions] → Observe → LLM → [More Actions] → Response

### What Makes This a "Multi-Agent" System?

Instead of one agent doing everything, we have **specialized agents** that each do one thing well:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Research   │ ──▶ │  Analysis   │ ──▶ │   Report    │
│   Agent     │     │   Agent     │     │   Agent     │
│             │     │             │     │             │
│ - Web search│     │ - RAG       │     │ - Synthesis │
│ - Extract   │     │ - LLM call  │     │ - Format    │
│ - Store     │     │ - Insights  │     │ - Citations │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Why multi-agent?**
- **Single Responsibility**: Each agent has one job
- **Debuggability**: Easier to find where things break
- **Reusability**: Agents can be reused in other workflows
- **Scalability**: Can run agents in parallel

---

## 🔑 Key Concepts Explained

### 1. State Management

All agents share a common **state object**:

```python
class AgentState(TypedDict):
    user_query: str           # Input from user
    research_results: list    # Output from research agent
    analysis: str             # Output from analysis agent
    final_report: str         # Output from report agent
    agent_logs: list          # For observability
```

**Why it matters**: 
- Each agent reads what it needs from state
- Each agent writes its output to state
- No direct agent-to-agent communication needed
- Makes debugging easy (inspect state at any point)

**Interview answer**: *"We use shared state for agent communication. Each agent is a pure function that takes state in and returns modified state out. This makes the system easy to test and debug."*

### 2. Tool Calling

Agents can use **tools** - external APIs or functions:

```python
def web_search(query: str) -> List[dict]:
    """Tool that searches the web"""
    results = tavily_client.search(query)
    return results
```

**Why it matters**:
- LLMs alone can only use training data
- Tools give agents access to real-time information
- Tools let agents take actions (not just generate text)

**Interview answer**: *"The research agent uses tool calling to integrate with Tavily's search API. This is production-grade - it handles errors, rate limits, and structures the output for downstream processing."*

### 3. RAG (Retrieval-Augmented Generation)

RAG is a two-step process:

1. **Retrieve**: Search a vector database for relevant context
2. **Augment**: Add that context to the LLM prompt
3. **Generate**: LLM generates response grounded in context

```python
# Step 1: Retrieve
relevant_docs = vector_db.query(user_query, n_results=5)

# Step 2: Augment prompt
prompt = f"""
Based on this context:
{relevant_docs}

Answer: {user_query}
"""

# Step 3: Generate
response = llm.invoke(prompt)
```

**Why RAG matters**:
- **Reduces hallucination** by grounding in real data
- **Provides citations** (we know where info came from)
- **Allows real-time updates** (vs fine-tuning which is static)
- **More cost-effective** than fine-tuning

**Interview answer**: *"We use RAG in the analysis agent. Research results are stored in ChromaDB, then retrieved based on semantic similarity. This grounds the LLM's analysis in actual data and lets us cite sources."*

### 4. Graph-Based Orchestration (LangGraph)

LangGraph represents workflows as **directed graphs**:

```python
# Define the graph
workflow = StateGraph(AgentState)

# Add nodes (agents)
workflow.add_node("research", research_agent)
workflow.add_node("analyze", analysis_agent)
workflow.add_node("report", report_agent)

# Add edges (flow)
workflow.set_entry_point("research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", "report")
workflow.add_edge("report", END)

# Compile and run
app = workflow.compile()
result = app.invoke(initial_state)
```

**Why graphs?**
- **Visual clarity**: Easy to understand the flow
- **Conditional routing**: Can branch based on state
- **Cycles**: Can loop back for iterative refinement
- **Parallel execution**: Independent branches run concurrently

**Interview answer**: *"We use LangGraph for orchestration because it gives us explicit control flow. Unlike chains, graphs support cycles and conditional branching, which is essential for complex agent behaviors."*

---

## 🏗️ Architecture Deep Dive

### Data Flow

```
User Query: "What are the latest AI agent developments?"
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│ RESEARCH AGENT                                       │
│                                                      │
│ 1. Generate search queries:                          │
│    - "latest AI agent developments"                  │
│    - "AI agent developments best practices"          │
│                                                      │
│ 2. Call Tavily API for each query                    │
│                                                      │
│ 3. Extract & deduplicate results                     │
│                                                      │
│ 4. Store in ChromaDB (vector embeddings)             │
│                                                      │
│ OUTPUT: research_results = [                         │
│   {title, url, content},                            │
│   {title, url, content}, ...                        │
│ ]                                                    │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│ ANALYSIS AGENT                                       │
│                                                      │
│ 1. RAG Retrieval:                                    │
│    Query ChromaDB for top 5 relevant chunks          │
│                                                      │
│ 2. Construct prompt:                                 │
│    "Based on this context: {chunks}                  │
│     Analyze: {user_query}                            │
│     Provide: key findings, patterns, data points"    │
│                                                      │
│ 3. Call Claude LLM                                   │
│                                                      │
│ OUTPUT: analysis = "## Key Findings..."              │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│ REPORT AGENT                                         │
│                                                      │
│ 1. Construct synthesis prompt:                       │
│    "Create report from: {analysis}                   │
│     Format: Executive Summary, Findings, etc."       │
│                                                      │
│ 2. Call Claude LLM                                   │
│                                                      │
│ 3. Append sources section                            │
│                                                      │
│ OUTPUT: final_report = "# Research Report..."        │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
            Final Report to User
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                    │
│                        (LangGraph)                          │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Research│───▶│ Analysis│───▶│ Report  │───▶│  END    │  │
│  │  Agent  │    │  Agent  │    │  Agent  │    │         │  │
│  └────┬────┘    └────┬────┘    └────┬────┘    └─────────┘  │
│       │              │              │                       │
└───────┼──────────────┼──────────────┼───────────────────────┘
        │              │              │
        ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   TOOLS     │  │  VECTOR DB  │  │    LLM      │
│             │  │             │  │             │
│ - Tavily    │  │ - ChromaDB  │  │ - Claude    │
│   Search    │  │ - Embeddings│  │   Sonnet    │
│             │  │ - Retrieval │  │             │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## 💬 Interview Q&A Guide

### "Walk me through how this works"

> "Sure! When a user submits a query, it flows through three specialized agents:
> 
> First, the **Research Agent** generates search queries and calls the Tavily API to gather current information. It stores results in a vector database.
> 
> Then, the **Analysis Agent** uses RAG - it retrieves the most relevant chunks from the vector DB and passes them to Claude to generate insights. This grounds the analysis in real data.
> 
> Finally, the **Report Agent** synthesizes everything into a formatted report with citations.
> 
> The key architectural decision is using LangGraph for orchestration. Each agent is a node in a graph, with edges defining the flow. State is shared between agents, so each one can access what previous agents produced."

### "Why multi-agent instead of one agent?"

> "Three main reasons:
> 
> **Single responsibility** - each agent does one thing well. The research agent is optimized for search, the analysis agent for reasoning, the report agent for synthesis.
> 
> **Debuggability** - when something goes wrong, I can check the state between agents to pinpoint exactly where.
> 
> **Flexibility** - I can swap out agents, add new ones, or run them in parallel without rewriting everything."

### "What's the hardest part of building agent systems?"

> "**Error handling and observability**. When you have multiple agents calling LLMs and external APIs, failure modes multiply. An agent can:
> - Return malformed output
> - Get rate limited
> - Produce contradictory results
> - Loop infinitely
> 
> That's why I log every step and track metrics. In production, I'd add circuit breakers and cost controls. Debugging distributed agent systems is the real challenge."

### "How would you improve this for production?"

> "Several things:
> 
> 1. **Persistent vector DB** - Switch from in-memory ChromaDB to Pinecone or Weaviate
> 2. **Caching** - Redis cache for repeated searches to reduce latency and cost
> 3. **Async execution** - Run independent agents in parallel
> 4. **Circuit breakers** - Prevent cascade failures when APIs are down
> 5. **Cost controls** - Budget limits per query, alerts on overspend
> 6. **Human-in-the-loop** - For high-stakes queries, add approval steps
> 7. **A/B testing** - Test different prompts to optimize agent performance
> 8. **Comprehensive monitoring** - Datadog/CloudWatch for agent-level metrics"

### "What's RAG and why use it?"

> "RAG is Retrieval-Augmented Generation. Instead of relying solely on the LLM's training data, we:
> 
> 1. **Retrieve** relevant documents from a vector database
> 2. **Augment** the prompt with those documents
> 3. **Generate** a response grounded in that context
> 
> Benefits over fine-tuning:
> - **Real-time updates** - no retraining needed
> - **Source attribution** - we can cite where info came from
> - **Reduced hallucination** - model is grounded in actual data
> - **Cost effective** - cheaper than fine-tuning"

---

## 📁 File Structure

```
agent_demo/
├── app.py              # Main application (Streamlit + agents)
├── requirements.txt    # Dependencies
└── README.md           # This documentation
```

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. Push to GitHub
2. Connect repo to [streamlit.io](https://streamlit.io)
3. Add API keys in Streamlit Secrets
4. Share URL in interviews

### Option 2: Local Demo

1. Run `streamlit run app.py`
2. Screenshare during interview
3. Walk through code + live demo

### Option 3: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

---

## 🎯 Key Takeaways for Your Interview

1. **You understand orchestration** - Not just single LLM calls, but coordinated multi-agent systems

2. **You know production patterns** - State management, error handling, observability, cost tracking

3. **You can explain RAG** - Why it works, when to use it, alternatives like fine-tuning

4. **You think about scale** - You know what to change for production (caching, async, monitoring)

5. **You build working demos** - This isn't theoretical - it runs and produces results

Good luck with your interview! 🚀
