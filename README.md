# 🤖 Multi-Agent Research System - Documentation

##  What This Demo Shows

This demo shows **production-grade AI agent systems** built. It demonstrates:

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


### Option 2: Local Demo


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

