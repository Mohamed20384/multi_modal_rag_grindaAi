import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# import sys
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Import your existing functions
from rag import query_rag, evaluate_batch, qa_pairs

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="RAG QA Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom Dark Theme CSS
# =========================
st.markdown("""
    <style>
    body, .stApp {
        background-color: #1e1e2f;
        color: #f0f0f0;
    }
    .main-header {
        font-size: 3rem;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        border-bottom: 2px solid #9b59b6;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #2c2f48;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        margin-bottom: 1rem;
        color: #f0f0f0;
    }
    .answer-box {
        background-color: #2c2f48;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
        color: #f0f0f0;
    }
    .snippet-box {
        background-color: #3a3d5c;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #9b59b6;
        color: #f0f0f0;
    }
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    .evaluation-table {
        font-size: 0.9rem;
        background-color: #2c2f48;
        color: #f0f0f0;
    }
    .stExpander {
        background-color: #2c2f48 !important;
        color: #f0f0f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown('<h1 class="main-header">🤖 RAG QA Assistant</h1>', unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2721/2721267.png", width=100)
    st.title("⚡ Navigation")
    menu = st.radio("Choose an option", ["Ask a Question", "Evaluate RAG with RAGAS", "About"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_details = st.checkbox("Show detailed information", value=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.info("This app uses RAG (Retrieval Augmented Generation) to answer questions based on your documents.")

# =========================
# Ask a Question
# =========================
if menu == "Ask a Question":
    st.markdown('<h2 class="sub-header">💬 Ask a Question</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_q = st.text_area(
            "Enter your question:",
            placeholder="Type your question here...",
            height=100
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.button("🚀 Get Answer", type="primary", use_container_width=True)
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.rerun()
    
    if submit_btn and user_q.strip():
        with st.spinner("🔍 Querying RAG pipeline..."):
            result = query_rag(user_q)
        
        st.markdown("### 🧠 Answer")
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence Score", f"{result['confidence']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Documents Used", len(result["doc_ids"]))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Snippets Retrieved", len(result["used_snippets"]))
            st.markdown('</div>', unsafe_allow_html=True)
        
        if show_details:
            with st.expander("📄 Document IDs used", expanded=False):
                for doc_id in result["doc_ids"]:
                    st.code(doc_id, language=None)
            with st.expander("✂️ Retrieved snippets", expanded=False):
                for i, s in enumerate(result["used_snippets"]):
                    st.markdown(f'**Snippet {i+1}** (Relevance: {s["score"]:.4f})')
                    st.markdown(f'<div class="snippet-box">{s["snippet"]}</div>', unsafe_allow_html=True)

# =========================
# Evaluation
# =========================
elif menu == "Evaluate RAG with RAGAS":
    st.markdown('<h2 class="sub-header">📊 RAGAS Evaluation</h2>', unsafe_allow_html=True)
    st.info("Evaluate the RAG system on predefined QA pairs using RAGAS metrics.")
    
    if st.button("⚡ Run Evaluation", type="primary"):
        with st.spinner("⏳ Running evaluation on QA pairs..."):
            df_results = evaluate_batch(qa_pairs)
        
        st.success("✅ Evaluation complete!")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ragas_eval_results_{timestamp}.csv"
        df_results.to_csv(filename, index=False)
        
        metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        stats = df_results[metrics].describe()
        
        # Metric cards
        cols = st.columns(len(metrics))
        for i, metric in enumerate(metrics):
            with cols[i]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(metric.replace("_", " ").title(),
                          f"{stats[metric]['mean']:.3f}",
                          delta=f"±{stats[metric]['std']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### 📈 Metrics Visualization")
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Box(y=df_results[metric], name=metric.replace('_', ' ').title()))
        fig.update_layout(
            template="plotly_dark",
            title="Distribution of Evaluation Metrics",
            yaxis_title="Score"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        avg_metrics = df_results[metrics].mean()
        fig2 = px.bar(
            x=metrics,
            y=avg_metrics.values,
            labels={"x": "Metric", "y": "Average Score"},
            title="Average Scores by Metric",
            color=metrics
        )
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### 📋 Detailed Results")
        st.dataframe(df_results, use_container_width=True)
        
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=df_results.to_csv(index=False),
            file_name=filename,
            mime="text/csv"
        )

# =========================
# About Page
# =========================
elif menu == "About":
    st.markdown('<h2 class="sub-header">ℹ️ About RAG QA Assistant</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="answer-box">
    This app implements a **Retrieval Augmented Generation (RAG)** system with:
    - Interactive question answering
    - RAGAS evaluation metrics
    - Dark dashboard theme
    - Visual analytics and export
    </div>
    """, unsafe_allow_html=True)
