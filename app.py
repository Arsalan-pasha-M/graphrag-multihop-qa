import streamlit as st
from mhqa.baseline_rag import BaselineRAG
from mhqa.graph_rag import GraphRAG
from mhqa.vector_store import VectorStore
from mhqa.graph_store import GraphStore

# Page Setup
st.set_page_config(
    page_title="GraphRAG vs RAG",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for polished container styling
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .stCard {
        border-radius: 8px;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ GraphRAG vs Traditional RAG")
st.caption("Multi-Hop Question Answering Benchmark")

@st.cache_resource
def systems():
    vs = VectorStore()
    return BaselineRAG(vs), GraphRAG(vs, GraphStore())

# Input Section with Sample Questions
q = st.text_input(
    "Ask a multi-hop question:",
    placeholder="e.g., In which county is the 8th Military Police Brigade of the US Army's barracks located?"
)

run_btn = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

if run_btn and q:
    base, graph = systems()
    c1, c2 = st.columns(2)
    
    for col, sysm in ((c1, base), (c2, graph)):
        with col:
            with st.spinner(f"Running {sysm.name}..."):
                o = sysm.run(q)
            
            # System Header
            st.markdown(f"### 🔹 `{sysm.name}`")
            
            # Answer Box
            answer_text = o["answer"] if o["answer"] and o["answer"].lower() != "unknown" else "⚠️ Answer Not Found"
            st.info(f"**Answer:** {answer_text}")
            
            # Key Performance Metrics
            m1, m2 = st.columns(2)
            m1.metric("Latency", f"{o['latency']:.2f} s")
            m2.metric("Context Size", f"{o['context_words']} words")
            
            st.divider()
            
            # Reasoning Path
            with st.expander("🧠 Reasoning Path", expanded=True):
                for i, step in enumerate(o["reasoning_path"]):
                    st.markdown(f"**{i+1}.** {step}")
            
            # Citations & Retrieved Documents
            with st.expander("📚 Sources & Citations"):
                if o.get("citations"):
                    st.markdown("**Citations:**")
                    if isinstance(o["citations"], dict):
                        for k, v in o["citations"].items():
                            st.caption(f"• **[{k}]**: {v}")
                    else:
                        st.json(o["citations"])
                        
                if o.get("titles"):
                    st.markdown("**Retrieved Documents:**")
                    for title in o["titles"]:
                        st.markdown(f"- `{title}`")

            # GraphViz Knowledge Graph Visualization (Styled for Dark Mode)
            if o.get("facts"):
                st.markdown("🕸️ **Knowledge Subgraph**")
                dot = """
                digraph G {
                    bgcolor="transparent";
                    rankdir=LR;
                    node [shape=box, style="filled,rounded", fillcolor="#1e222a", fontcolor="#ffffff", color="#4b5563", fontname="sans-serif"];
                    edge [fontcolor="#9ca3af", color="#6b7280", fontsize=10, fontname="sans-serif"];
                """ + "\n".join(
                    f'"{f["head"]}" -> "{f["tail"]}" [label=" {f["rel"]} "];' for f in o["facts"][:25]
                ) + "\n}"
                
                st.graphviz_chart(dot, use_container_width=True)