# Agentic Research Runtime

Production-style reference implementation for an **agentic research workflow**: research planning, tool use, retrieval, analysis, report synthesis, execution logs, and failure-aware orchestration.

This repository is intentionally structured as a demo/reference project: it runs locally without API keys in demo mode, but also supports live Claude + Tavily execution when credentials are provided.

## Why this project matters

Most “AI agent” demos are just a prompt wrapped in a UI. This project demonstrates the control layer around an LLM workflow:

- multi-step agent orchestration
- shared workflow state
- tool calling for web search
- retrieval-augmented analysis
- execution logs for observability
- error handling and fallback behaviour
- demo/live mode separation
- report synthesis with source tracking

The goal is not to claim this is a full enterprise platform. The goal is to show the core engineering patterns needed before agentic systems can be trusted in production.

## Architecture

```text
User Query
   |
   v
Research Agent
   |-- expands query
   |-- calls search tool
   |-- stores source snippets
   v
Retrieval / Context Layer
   |-- keeps searchable context
   |-- returns relevant chunks
   v
Analysis Agent
   |-- reasons over retrieved context
   |-- extracts findings, risks, gaps
   v
Report Agent
   |-- synthesizes final answer
   |-- attaches sources
   v
Observability Layer
   |-- agent logs
   |-- execution time
   |-- source count
   |-- LLM call count
```

## Current capabilities

- Streamlit UI for running research workflows
- Demo mode with mock search/vector data
- Live mode with Anthropic Claude and Tavily search
- Typed workflow state using `TypedDict`
- Custom sequential orchestration across research, analysis, and report-generation agents
- RAG-style context retrieval
- Agent execution log timeline
- Basic workflow metrics
- Docker-ready deployment path

## Tech stack

- Python
- Streamlit
- Anthropic Claude via LangChain integration
- Tavily search API
- Custom orchestration functions with typed shared state
- Mock vector store for no-key demo mode
- Docker

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The app runs in demo mode without keys.

## Live mode

```bash
export ANTHROPIC_API_KEY=your_key_here
export TAVILY_API_KEY=your_key_here
streamlit run app.py
```

## Example use cases

- research assistant for market scans
- AI-agent workflow demo
- agent orchestration pattern reference
- RAG + tool-use prototype
- observability demo for multi-step AI workflows

## Production hardening roadmap

The current repo is a reference implementation. To make this enterprise-grade, add:

- real LangGraph `StateGraph` implementation
- durable state persistence in Postgres
- Redis-backed queue and cache
- tool permission registry
- human approval gates for risky actions
- replay/retry controls
- token/cost accounting
- OpenTelemetry traces
- eval checks before final report generation
- CI pipeline with linting and tests
- role-based access control

## Resume positioning

This repo supports the following capability claims:

- Agentic AI workflow design
- Tool-calling orchestration
- RAG-style context construction
- LLM application observability
- AI-assisted research automation
- Production-pattern thinking around agents

## Repository status

Public reference implementation. Suitable for demos and technical walkthroughs; not positioned as a production SaaS product.
