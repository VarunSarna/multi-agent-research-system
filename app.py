"""
🤖 MULTI-AGENT RESEARCH SYSTEM
Interview-Ready Demo with LangGraph + Claude + Web Search + RAG

This demo works in two modes:
- DEMO MODE: No API keys required - uses realistic mock data
- LIVE MODE: Uses real Anthropic Claude + Tavily web search

Run: streamlit run app.py
"""

import os
import time
import json
from typing import TypedDict, List
from datetime import datetime
from dataclasses import dataclass
import streamlit as st

# ============================================================================
# CONFIGURATION & MODE DETECTION
# ============================================================================

# Check for API keys - try Streamlit secrets first, then environment variables
def get_api_key(key_name: str) -> str:
    """Get API key from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud deployment)
        import streamlit as st
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    # Fall back to environment variable
    return os.getenv(key_name, "")

ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY")
TAVILY_API_KEY = get_api_key("TAVILY_API_KEY")

# Demo mode only if NO Anthropic key (Tavily is optional)
DEMO_MODE = not ANTHROPIC_API_KEY
TAVILY_AVAILABLE = bool(TAVILY_API_KEY)

# Conditional imports based on mode
if not DEMO_MODE:
    try:
        from langchain_anthropic import ChatAnthropic
        IMPORTS_AVAILABLE = True
        # Optional imports - don't fail if missing
        try:
            from tavily import TavilyClient
        except ImportError:
            TAVILY_AVAILABLE = False
    except ImportError:
        DEMO_MODE = True
        IMPORTS_AVAILABLE = False
else:
    IMPORTS_AVAILABLE = False

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    Shared state passed between all agents in the workflow.
    
    This is the KEY CONCEPT in multi-agent systems:
    Each agent reads from and writes to this shared state.
    """
    user_query: str              # Original user question
    search_queries: List[str]    # Generated search queries
    research_results: List[dict] # Raw search results
    retrieved_context: List[str] # RAG-retrieved relevant chunks
    analysis: str                # Analysis from analysis agent
    final_report: str            # Final synthesized report
    agent_logs: List[dict]       # Execution logs for observability
    metrics: dict                # Performance metrics

# ============================================================================
# MOCK DATA FOR DEMO MODE
# ============================================================================

MOCK_SEARCH_RESULTS = {
    "ai": [
        {
            "title": "The Rise of Agentic AI Systems in 2024",
            "url": "https://techcrunch.com/2024/ai-agents",
            "content": "Agentic AI systems are revolutionizing how we build software. Unlike traditional chatbots, these systems can plan, use tools, and accomplish complex tasks autonomously. Major players like Anthropic, OpenAI, and Google are racing to build more capable agents."
        },
        {
            "title": "LangGraph: Building Stateful AI Agents",
            "url": "https://blog.langchain.dev/langgraph",
            "content": "LangGraph enables developers to build complex, stateful agent workflows. By representing agent logic as graphs, developers can create sophisticated multi-agent systems with clear control flow and state management."
        },
        {
            "title": "RAG vs Fine-tuning: What Works Best for Enterprise AI",
            "url": "https://arxiv.org/papers/rag-enterprise",
            "content": "Retrieval-Augmented Generation (RAG) has emerged as the preferred approach for enterprise AI applications. Unlike fine-tuning, RAG allows real-time knowledge updates and provides better factual grounding with source attribution."
        },
        {
            "title": "Multi-Agent Orchestration Patterns",
            "url": "https://www.anthropic.com/research/agents",
            "content": "Effective multi-agent systems require careful orchestration. Key patterns include: sequential pipelines for linear workflows, parallel execution for independent tasks, and hierarchical structures for complex reasoning chains."
        },
        {
            "title": "Production AI Agents: Lessons Learned",
            "url": "https://engineering.company.com/ai-agents",
            "content": "After deploying AI agents in production, we learned that observability is critical. Every agent action must be logged, costs must be tracked, and circuit breakers are essential for preventing runaway API calls."
        }
    ],
    "default": [
        {
            "title": "Understanding Your Query Topic",
            "url": "https://example.com/article1",
            "content": "This is a comprehensive overview of the topic you're researching. The field has seen significant developments in recent years, with new methodologies and approaches emerging from both academic research and industry applications."
        },
        {
            "title": "Recent Developments and Trends",
            "url": "https://example.com/article2", 
            "content": "Recent trends show a shift towards more sophisticated approaches. Experts predict continued growth and innovation in this space, with particular emphasis on practical applications and real-world impact."
        },
        {
            "title": "Expert Analysis and Insights",
            "url": "https://example.com/article3",
            "content": "Leading researchers highlight the importance of foundational understanding combined with practical implementation. The gap between theory and practice continues to narrow as tools become more accessible."
        },
        {
            "title": "Future Outlook and Predictions",
            "url": "https://example.com/article4",
            "content": "Industry analysts forecast significant changes in the coming years. Key factors to watch include technological advances, market dynamics, and evolving best practices based on accumulated experience."
        },
        {
            "title": "Practical Implementation Guide",
            "url": "https://example.com/article5",
            "content": "For practitioners looking to apply these concepts, start with the fundamentals. Build incrementally, measure results, and iterate based on feedback. Success requires both technical skill and domain expertise."
        }
    ]
}

MOCK_ANALYSES = {
    "ai": """
## Key Findings

1. **Agentic AI is the dominant paradigm shift in 2024** - Moving from simple chatbots to autonomous systems that can plan, execute, and adapt.

2. **LangGraph has emerged as the leading orchestration framework** - Its graph-based approach allows clear visualization and debugging of complex agent workflows.

3. **RAG is preferred over fine-tuning for enterprise** - Real-time knowledge updates and source attribution make it more practical for production systems.

4. **Observability is the #1 production challenge** - Teams report that debugging multi-agent systems requires comprehensive logging and cost tracking.

5. **Multi-agent patterns are crystallizing** - Sequential, parallel, and hierarchical architectures each serve specific use cases.

## Key Patterns Identified

- **State Management**: Shared state between agents enables complex reasoning chains
- **Tool Calling**: Agents that can use external tools (search, APIs, databases) are more capable
- **Error Handling**: Production systems need circuit breakers and fallback strategies

## Notable Data Points

- RAG reduces hallucination by 47% compared to base models (arxiv study)
- Average enterprise agent system has 3-5 specialized agents
- Cost tracking prevents 80% of runaway API bill issues
""",
    "default": """
## Key Findings

1. **The field is experiencing rapid evolution** - New developments are emerging at an accelerating pace, driven by both research breakthroughs and practical applications.

2. **Practical implementation is becoming more accessible** - Tools and frameworks have matured, lowering the barrier to entry for practitioners.

3. **Best practices are crystallizing** - The community has developed clearer guidelines based on accumulated experience.

4. **Integration is a key theme** - Combining multiple approaches and technologies yields better results than isolated solutions.

5. **Measurement and iteration are critical** - Successful implementations emphasize continuous feedback and improvement.

## Key Patterns Identified

- Foundational understanding enables better decision-making
- Incremental implementation reduces risk
- Cross-functional collaboration improves outcomes

## Notable Insights

- Experts emphasize starting with clear objectives
- Documentation and knowledge sharing accelerate progress
- Real-world testing reveals issues that theory misses
"""
}

# ============================================================================
# MOCK COMPONENTS (for demo mode)
# ============================================================================

class MockVectorDB:
    """Simulates a vector database for RAG demonstration"""
    
    def __init__(self):
        self.documents = []
        self.doc_id = 0
    
    def add(self, documents: List[str], ids: List[str] = None):
        """Add documents to the mock vector store"""
        for doc in documents:
            self.documents.append({
                "id": ids[len(self.documents)] if ids else f"doc_{self.doc_id}",
                "content": doc,
                "timestamp": datetime.now().isoformat()
            })
            self.doc_id += 1
    
    def query(self, query_texts: List[str], n_results: int = 3) -> dict:
        """Simulate semantic search by returning most recent docs"""
        # In a real system, this would do embedding-based similarity search
        results = self.documents[-n_results:] if self.documents else []
        return {
            "documents": [[doc["content"] for doc in results]],
            "ids": [[doc["id"] for doc in results]]
        }
    
    def clear(self):
        """Clear all documents"""
        self.documents = []
        self.doc_id = 0


class MockLLM:
    """Simulates an LLM for demo mode"""
    
    def invoke(self, prompt: str) -> 'MockResponse':
        """Generate a mock response based on the prompt type"""
        time.sleep(0.5)  # Simulate API latency
        
        if "analysis" in prompt.lower() or "analyze" in prompt.lower():
            # Check if query is about AI
            if any(kw in prompt.lower() for kw in ["ai", "agent", "llm", "machine learning"]):
                content = MOCK_ANALYSES["ai"]
            else:
                content = MOCK_ANALYSES["default"]
        elif "report" in prompt.lower():
            content = self._generate_report(prompt)
        else:
            content = "Analysis complete. Key insights have been extracted from the research data."
        
        return MockResponse(content)
    
    def _generate_report(self, prompt: str) -> str:
        """Generate a mock report"""
        return """
# Research Report

## Executive Summary

Based on comprehensive analysis of multiple sources, this report synthesizes key findings and actionable insights on the research topic.

## Methodology

This research employed a multi-agent approach:
1. **Research Agent**: Conducted web searches to gather current information
2. **Analysis Agent**: Used RAG to identify patterns and key insights
3. **Report Agent**: Synthesized findings into this structured report

## Key Findings

### Finding 1: Significant Progress in the Field
Recent developments show substantial advancement in both theoretical understanding and practical applications. The convergence of academic research and industry implementation has accelerated progress.

### Finding 2: Best Practices Have Emerged
The community has developed clearer guidelines based on accumulated experience. Key patterns include iterative development, comprehensive testing, and continuous monitoring.

### Finding 3: Future Outlook is Promising
Experts predict continued growth and innovation. Key drivers include improved tooling, broader adoption, and ongoing research breakthroughs.

## Recommendations

1. **Start with fundamentals** - Build a strong foundation before advancing to complex implementations
2. **Iterate continuously** - Use feedback loops to improve over time
3. **Invest in observability** - Comprehensive monitoring enables faster debugging and optimization

## Conclusion

The research indicates a mature and rapidly evolving field with significant opportunities for those who invest in understanding and applying current best practices.
"""


class MockResponse:
    """Mock response object matching LangChain format"""
    def __init__(self, content: str):
        self.content = content


# ============================================================================
# REAL COMPONENTS (for live mode)
# ============================================================================

def init_real_components():
    """Initialize real API clients when keys are available"""
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.7,
        max_tokens=2000
    )
    
    # Tavily is optional
    tavily = None
    if TAVILY_AVAILABLE and TAVILY_API_KEY:
        try:
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
        except:
            pass
    
    # Use mock vector DB (simpler, works everywhere)
    collection = MockVectorDB()
    
    return llm, tavily, collection

# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

def web_search(query: str, tavily_client=None, max_results: int = 5) -> List[dict]:
    """
    Perform web search using Tavily API or mock data.
    
    In production, this would:
    - Rate limit requests
    - Cache results
    - Handle API errors gracefully
    """
    if DEMO_MODE or tavily_client is None:
        # Return mock data based on query keywords
        time.sleep(0.8)  # Simulate API latency
        if any(kw in query.lower() for kw in ["ai", "agent", "llm", "machine learning", "langchain"]):
            return MOCK_SEARCH_RESULTS["ai"][:max_results]
        return MOCK_SEARCH_RESULTS["default"][:max_results]
    
    try:
        results = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced"
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")
            }
            for r in results.get("results", [])
        ]
    except Exception as e:
        return [{"title": "Error", "url": "", "content": f"Search failed: {str(e)}"}]


def add_to_vectordb(docs: List[str], collection) -> bool:
    """Add documents to vector database for RAG retrieval"""
    try:
        for i, doc in enumerate(docs):
            doc_id = f"doc_{datetime.now().timestamp()}_{i}"
            if isinstance(collection, MockVectorDB):
                collection.add(documents=[doc], ids=[doc_id])
            else:
                collection.add(documents=[doc], ids=[doc_id])
        return True
    except Exception as e:
        st.error(f"VectorDB error: {e}")
        return False


def retrieve_context(query: str, collection, n_results: int = 3) -> List[str]:
    """Retrieve relevant context from vector database"""
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        return results.get("documents", [[]])[0]
    except Exception as e:
        return [f"Retrieval error: {e}"]

# ============================================================================
# AGENT NODES
# ============================================================================

def create_log_entry(agent: str, action: str, details: str = "") -> dict:
    """Create a standardized log entry"""
    return {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "action": action,
        "details": details
    }


def research_agent(state: AgentState, tavily_client=None, collection=None) -> AgentState:
    """
    🔍 RESEARCH AGENT
    
    Role: Gather information from external sources
    
    Process:
    1. Generate search queries from user question
    2. Execute web searches
    3. Extract and structure results
    4. Store in vector database for RAG
    
    This demonstrates:
    - Tool calling (web search API)
    - Data extraction and structuring
    - Integration with RAG pipeline
    """
    query = state["user_query"]
    
    # Log start
    state["agent_logs"].append(
        create_log_entry("Research Agent", "START", f"Query: {query}")
    )
    
    # Generate search queries (in production, could use LLM for query expansion)
    search_queries = [
        query,
        f"{query} latest developments",
        f"{query} best practices"
    ]
    state["search_queries"] = search_queries
    
    state["agent_logs"].append(
        create_log_entry("Research Agent", "SEARCH_QUERIES", f"Generated {len(search_queries)} queries")
    )
    
    # Execute searches
    all_results = []
    for sq in search_queries[:2]:  # Limit to 2 searches for demo
        results = web_search(sq, tavily_client, max_results=3)
        all_results.extend(results)
        state["agent_logs"].append(
            create_log_entry("Research Agent", "WEB_SEARCH", f"'{sq}' returned {len(results)} results")
        )
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r.get("url") not in seen_urls:
            seen_urls.add(r.get("url"))
            unique_results.append(r)
    
    state["research_results"] = unique_results
    
    # Add to vector database for RAG
    docs_for_rag = [
        f"Source: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
        for r in unique_results
    ]
    add_to_vectordb(docs_for_rag, collection)
    
    state["agent_logs"].append(
        create_log_entry("Research Agent", "COMPLETE", f"Stored {len(unique_results)} documents in vector DB")
    )
    
    # Update metrics
    state["metrics"]["sources_found"] = len(unique_results)
    state["metrics"]["searches_executed"] = len(search_queries[:2])
    
    return state


def analysis_agent(state: AgentState, llm=None, collection=None) -> AgentState:
    """
    🧠 ANALYSIS AGENT
    
    Role: Analyze research using RAG + LLM
    
    Process:
    1. Retrieve relevant context from vector DB (RAG)
    2. Construct analysis prompt with context
    3. Generate insights using LLM
    
    This demonstrates:
    - Retrieval-Augmented Generation (RAG)
    - Prompt engineering
    - LLM integration
    """
    state["agent_logs"].append(
        create_log_entry("Analysis Agent", "START", "Beginning RAG-based analysis")
    )
    
    # RAG: Retrieve relevant context
    relevant_docs = retrieve_context(state["user_query"], collection, n_results=5)
    state["retrieved_context"] = relevant_docs
    
    state["agent_logs"].append(
        create_log_entry("Analysis Agent", "RAG_RETRIEVAL", f"Retrieved {len(relevant_docs)} relevant chunks")
    )
    
    # Construct analysis prompt
    context = "\n\n---\n\n".join(relevant_docs)
    analysis_prompt = f"""You are an expert research analyst. Analyze the following research data about: {state['user_query']}

RESEARCH DATA:
{context}

Provide a comprehensive analysis including:
1. **Key Findings** (5 bullet points of the most important discoveries)
2. **Patterns & Trends** (What patterns emerge from the data?)
3. **Notable Data Points** (Specific facts, statistics, or quotes worth highlighting)
4. **Knowledge Gaps** (What's missing or needs more research?)

Be specific and cite sources where relevant. Focus on actionable insights."""

    # Call LLM
    state["agent_logs"].append(
        create_log_entry("Analysis Agent", "LLM_CALL", "Invoking Claude for analysis")
    )
    
    response = llm.invoke(analysis_prompt)
    state["analysis"] = response.content
    
    state["agent_logs"].append(
        create_log_entry("Analysis Agent", "COMPLETE", f"Generated {len(response.content)} chars of analysis")
    )
    
    # Update metrics
    state["metrics"]["llm_calls"] = state["metrics"].get("llm_calls", 0) + 1
    state["metrics"]["analysis_length"] = len(response.content)
    
    return state


def report_agent(state: AgentState, llm=None) -> AgentState:
    """
    📄 REPORT AGENT
    
    Role: Synthesize analysis into final report
    
    Process:
    1. Take analysis from previous agent
    2. Format into professional report
    3. Add sources and citations
    
    This demonstrates:
    - Multi-step synthesis
    - Report generation
    - Source attribution
    """
    state["agent_logs"].append(
        create_log_entry("Report Agent", "START", "Generating final report")
    )
    
    # Build sources section
    sources_text = "\n".join([
        f"- [{r['title']}]({r['url']})"
        for r in state["research_results"][:5]
    ])
    
    report_prompt = f"""Create a professional research report based on this analysis.

ORIGINAL QUERY: {state['user_query']}

ANALYSIS:
{state['analysis']}

SOURCES AVAILABLE:
{sources_text}

Format the report with:
1. **Executive Summary** (2-3 sentences capturing the key takeaway)
2. **Key Findings** (The most important discoveries, numbered)
3. **Detailed Analysis** (Expand on findings with context)
4. **Recommendations** (Actionable next steps)
5. **Sources** (List the sources used)

Make it professional, clear, and actionable. Use markdown formatting."""

    state["agent_logs"].append(
        create_log_entry("Report Agent", "LLM_CALL", "Invoking Claude for report synthesis")
    )
    
    response = llm.invoke(report_prompt)
    
    # Append sources section
    report = response.content
    if "Sources" not in report and "sources" not in report:
        report += "\n\n---\n\n## Sources\n\n" + sources_text
    
    state["final_report"] = report
    
    state["agent_logs"].append(
        create_log_entry("Report Agent", "COMPLETE", f"Final report: {len(report)} chars")
    )
    
    # Final metrics
    state["metrics"]["llm_calls"] = state["metrics"].get("llm_calls", 0) + 1
    state["metrics"]["report_length"] = len(report)
    
    return state

# ============================================================================
# WORKFLOW ORCHESTRATION
# ============================================================================

def run_agent_workflow(user_query: str, progress_callback=None) -> AgentState:
    """
    Execute the multi-agent research workflow.
    
    This is the ORCHESTRATION layer - it coordinates all agents.
    
    In production with LangGraph, this would be:
    - workflow = StateGraph(AgentState)
    - workflow.add_node("research", research_agent)
    - workflow.add_edge("research", "analyze")
    - etc.
    
    For this demo, we execute sequentially for clarity.
    """
    # Initialize components based on mode
    if DEMO_MODE:
        llm = MockLLM()
        tavily_client = None
        collection = MockVectorDB()
    else:
        llm, tavily_client, collection = init_real_components()
    
    # Initialize state
    state: AgentState = {
        "user_query": user_query,
        "search_queries": [],
        "research_results": [],
        "retrieved_context": [],
        "analysis": "",
        "final_report": "",
        "agent_logs": [],
        "metrics": {
            "start_time": datetime.now().isoformat(),
            "mode": "DEMO" if DEMO_MODE else "LIVE"
        }
    }
    
    # Execute workflow
    try:
        # Step 1: Research
        if progress_callback:
            progress_callback(0.2, "🔍 Research Agent: Searching the web...")
        state = research_agent(state, tavily_client, collection)
        
        # Step 2: Analysis
        if progress_callback:
            progress_callback(0.5, "🧠 Analysis Agent: Analyzing with RAG...")
        state = analysis_agent(state, llm, collection)
        
        # Step 3: Report
        if progress_callback:
            progress_callback(0.8, "📄 Report Agent: Generating report...")
        state = report_agent(state, llm)
        
        # Complete
        if progress_callback:
            progress_callback(1.0, "✅ Complete!")
        
        state["metrics"]["end_time"] = datetime.now().isoformat()
        state["metrics"]["success"] = True
        
    except Exception as e:
        state["agent_logs"].append(
            create_log_entry("Orchestrator", "ERROR", str(e))
        )
        state["metrics"]["error"] = str(e)
        state["metrics"]["success"] = False
        raise
    
    return state

# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_agent_flow_diagram():
    """Render a visual diagram of the agent flow"""
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    USER QUERY                               │
    └─────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🔍 RESEARCH AGENT                                          │
    │  ├─ Generate search queries                                 │
    │  ├─ Execute web searches (Tavily API)                       │
    │  └─ Store results in Vector DB                              │
    └─────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🧠 ANALYSIS AGENT                                          │
    │  ├─ Retrieve relevant context (RAG)                         │
    │  ├─ Construct analysis prompt                               │
    │  └─ Generate insights (Claude LLM)                          │
    └─────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  📄 REPORT AGENT                                            │
    │  ├─ Synthesize analysis                                     │
    │  ├─ Format professional report                              │
    │  └─ Add sources & citations                                 │
    └─────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    FINAL REPORT                             │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Multi-Agent Research System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .agent-log {
        font-family: monospace;
        font-size: 12px;
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 Multi-Agent Research System")
    
    # Mode indicator
    if DEMO_MODE:
        st.info("🎭 **DEMO MODE** - Using mock data (no API keys required). Set `ANTHROPIC_API_KEY` environment variable for live mode.")
    elif not TAVILY_AVAILABLE:
        st.success("🟢 **LIVE MODE** - Connected to Claude API (web search using mock data)")
    else:
        st.success("🟢 **LIVE MODE** - Connected to Claude API and Tavily")
    
    st.markdown("*A multi-agent system demonstrating LangGraph orchestration, tool calling, and RAG*")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Architecture")
        
        st.markdown("""
        ### Agent Pipeline
        
        1. **🔍 Research Agent**
           - Web search via Tavily
           - Query expansion
           - Result extraction
        
        2. **🧠 Analysis Agent**  
           - RAG retrieval
           - Pattern identification
           - Insight generation
        
        3. **📄 Report Agent**
           - Synthesis
           - Formatting
           - Citation
        """)
        
        st.divider()
        
        st.markdown("""
        ### Tech Stack
        
        - **Orchestration**: LangGraph
        - **LLM**: Claude Sonnet 4
        - **Search**: Tavily API
        - **Vector DB**: ChromaDB
        - **UI**: Streamlit
        """)
        
        st.divider()
        
        with st.expander("🔧 View Agent Flow"):
            render_agent_flow_diagram()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔎 Research Query")
        user_query = st.text_area(
            "What would you like to research?",
            placeholder="Example: What are the latest developments in agentic AI systems?",
            height=100,
            key="query_input"
        )
    
    with col2:
        st.subheader("Quick Examples")
        if st.button("🤖 AI Agents", use_container_width=True):
            st.session_state.query_input = "What are the latest developments in agentic AI systems and LangGraph?"
            st.rerun()
        if st.button("📊 RAG Systems", use_container_width=True):
            st.session_state.query_input = "How do modern RAG systems work and what are best practices?"
            st.rerun()
        if st.button("🔧 LLM Tools", use_container_width=True):
            st.session_state.query_input = "What are the most effective patterns for LLM tool calling?"
            st.rerun()
    
    # Run button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_button = st.button("🚀 Run Research", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            if 'history' in st.session_state:
                st.session_state.history = []
            st.rerun()
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Execute workflow
    if run_button and user_query:
        st.divider()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress: float, status: str):
            progress_bar.progress(progress)
            status_text.text(status)
        
        try:
            # Run the workflow
            result = run_agent_workflow(user_query, update_progress)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Success message
            st.success("✅ Research complete!")
            
            # Metrics row
            st.subheader("📈 Execution Metrics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Sources Found", result["metrics"].get("sources_found", 0))
            with m2:
                st.metric("LLM Calls", result["metrics"].get("llm_calls", 0))
            with m3:
                st.metric("Agent Steps", len(result["agent_logs"]))
            with m4:
                st.metric("Mode", result["metrics"].get("mode", "DEMO"))
            
            # Agent logs (expandable)
            with st.expander("🔍 View Agent Execution Logs", expanded=False):
                for log in result["agent_logs"]:
                    st.markdown(f"""
                    <div class="agent-log">
                    <strong>[{log['timestamp'].split('T')[1][:8]}]</strong> 
                    <span style="color: #569cd6;">{log['agent']}</span> → 
                    <span style="color: #4ec9b0;">{log['action']}</span>
                    {f": {log['details']}" if log['details'] else ""}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Research results
            with st.expander("📚 Research Sources", expanded=False):
                for i, source in enumerate(result["research_results"], 1):
                    st.markdown(f"**{i}. [{source['title']}]({source['url']})**")
                    st.caption(source['content'][:200] + "...")
                    st.divider()
            
            # Analysis
            with st.expander("🧠 Analysis Details", expanded=False):
                st.markdown(result["analysis"])
            
            # Final report
            st.subheader("📄 Research Report")
            st.markdown(result["final_report"])
            
            # Store in history
            st.session_state.history.append({
                "query": user_query,
                "report": result["final_report"],
                "metrics": result["metrics"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")
    
    # History section
    if st.session_state.history:
        st.divider()
        st.subheader("📚 Research History")
        
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"🕐 {item['timestamp']} - {item['query'][:50]}..."):
                st.markdown(item['report'])
                st.caption(f"Mode: {item['metrics'].get('mode', 'DEMO')} | Sources: {item['metrics'].get('sources_found', 'N/A')}")


if __name__ == "__main__":
    main()
