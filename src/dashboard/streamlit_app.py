import sys
import os
import hashlib
import logging
import pandas as pd
import streamlit as st

# Configure Root Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.database.sqlite_manager import SQLiteManager
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.ingestion.document_ingestion_pipeline import DocumentIngestionPipeline
from src.evaluation.benchmarking import RetrievalBenchmarker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Streamlit Page Config & Custom Styling (Executive UI)
# ---------------------------------------------------------

st.set_page_config(
    page_title="Enterprise RAG Platform | Automated Evaluation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional Dark Theme CSS
st.markdown("""
<style>
    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }

    /* Executive Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Status Badge Tags */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-blue { background-color: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-purple { background-color: rgba(192, 132, 252, 0.15); color: #C084FC; border: 1px solid rgba(192, 132, 252, 0.3); }
    .badge-green { background-color: rgba(74, 222, 128, 0.15); color: #4ADE80; border: 1px solid rgba(74, 222, 128, 0.3); }
    .badge-amber { background-color: rgba(251, 191, 36, 0.15); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .badge-red { background-color: rgba(248, 113, 113, 0.15); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.3); }

    /* Latency & Metric Cards */
    .metric-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Rationale Box */
    .rationale-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #818CF8;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 16px 0;
        color: #E2E8F0;
        font-size: 0.95rem;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Service Caching & Initialization
# ---------------------------------------------------------

@st.cache_resource
def load_pipeline() -> RAGPipeline:
    logger.info("Initializing RAGPipeline...")
    return RAGPipeline()

@st.cache_resource
def load_evaluator() -> RagasEvaluator:
    logger.info("Initializing RagasEvaluator...")
    return RagasEvaluator()

try:
    pipeline = load_pipeline()
    evaluator = load_evaluator()
    db = SQLiteManager()
    ingestion = DocumentIngestionPipeline()
except Exception as init_err:
    logger.error(f"Failed to initialize core RAG services: {init_err}")
    st.error(f"System Initialization Error: {init_err}")
    st.stop()

# Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()
if "evaluation_history" not in st.session_state:
    st.session_state.evaluation_history = []

# ---------------------------------------------------------
# Header Hero Component
# ---------------------------------------------------------

st.markdown("""
<div class="hero-container">
    <div style="margin-bottom: 8px;">
        <span class="badge badge-blue">ENTERPRISE RAG</span>
        <span class="badge badge-purple">LLM-AS-JUDGE</span>
        <span class="badge badge-green">HYBRID SEARCH</span>
    </div>
    <div class="hero-title">Production-Grade RAG Platform</div>
    <div class="hero-subtitle">Automated Hallucination Detection, Multi-Query Expansion, Context Compression & Retrieval Benchmarking</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Component
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚡ System Observability")
    st.caption("Active Pipeline Settings & Health")
    
    st.markdown("---")
    st.subheader("📊 Session Activity")
    
    st.metric("Queries Processed", len(st.session_state.history))
    
    if st.session_state.evaluation_history:
        latest = st.session_state.evaluation_history[-1]
        st.markdown("**Latest Query Metrics**")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.metric("Faithfulness", f"{latest['faithfulness']:.1f}%")
        with sc2:
            st.metric("Judge Score", f"{latest['judge_score']:.1f}%")
    
    st.markdown("---")
    st.subheader("📜 Recent Query History")
    if not st.session_state.history:
        st.info("No queries executed in this session.")
    else:
        for idx, q_text in enumerate(st.session_state.history[:5], 1):
            st.write(f"**{idx}.** {q_text[:45]}...")

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Query RAG",
    "📄 Document Management",
    "📈 Evaluation Dashboard",
    "📊 Retrieval Benchmarking"
])

# =========================================================
# TAB 1: QUERY RAG & INFERENCE
# =========================================================
with tab1:
    st.subheader("Ask Questions Across Indexed Knowledge Base")

    query_input = st.text_area(
        "Enter Query Prompt",
        placeholder="e.g. What are the key compliance requirements under SEBI board regulations?",
        height=90,
        help="Input your search question. The system will rewrite the query, expand multi-queries, and perform hybrid retrieval."
    )

    search_type_selection = st.selectbox(
        "Retrieval Strategy",
        ["Hybrid (FAISS Vector + BM25)", "Vector Only (FAISS)", "BM25 Only (Lexical Keyword)"],
        help="Select retrieval mode: Hybrid uses Reciprocal Rank Fusion (RRF)."
    )
    # top_k_slider = st.slider("Top Chunks (K)", min_value=1, max_value=20, value=8)
    top_k_slider = 8

    # Strategy Mapping
    if "Vector Only" in search_type_selection:
        search_mode = "vector"
    elif "BM25 Only" in search_type_selection:
        search_mode = "bm25"
    else:
        search_mode = "hybrid"

    if st.button("🚀 Run RAG Inference", use_container_width=True):
        if not query_input.strip():
            st.warning("Please enter a non-empty question.")
            st.stop()

        try:
            with st.spinner(f"🔄 Executing RAG Pipeline [{search_mode.upper()}] — Rewrite → Expansion → Retrieval → Rerank → Compression → Generation → LLM Judge"):
                result = pipeline.run(
                    query=query_input,
                    search_type=search_mode,
                    top_k=top_k_slider
                )

                st.session_state.history.insert(0, query_input)
                st.session_state.history = st.session_state.history[:20]

            answer = result["answer"]
            sources = result["sources"]
            chunks = result["chunks"]
            confidence = result["confidence"]
            evaluation = result["evaluation"]
            metrics = result.get("metrics", {})

            # Persist Session Metrics
            eval_record = evaluation.copy()
            eval_record["total_latency_ms"] = metrics.get("total_latency_ms", 0)
            st.session_state.evaluation_history.append(eval_record)

            # ---------------------------------------------
            # Answer Output Component
            # ---------------------------------------------
            st.markdown("### 💡 Synthesized Answer")
            st.markdown(f"> {answer}")

            # ---------------------------------------------
            # Latency Metrics Component
            # ---------------------------------------------
            st.markdown("### ⏱️ Execution Latency Breakdown")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{metrics.get("total_latency_ms", 0):.0f} ms</div><div class="metric-label">Total Latency</div></div>', unsafe_allow_html=True)
            with mc2:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{metrics.get("retrieval_ms", 0):.0f} ms</div><div class="metric-label">Retrieval</div></div>', unsafe_allow_html=True)
            with mc3:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{metrics.get("reranking_ms", 0):.0f} ms</div><div class="metric-label">Re-Ranking</div></div>', unsafe_allow_html=True)
            with mc4:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{metrics.get("generation_ms", 0):.0f} ms</div><div class="metric-label">LLM Gen</div></div>', unsafe_allow_html=True)
            with mc5:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{metrics.get("evaluation_ms", 0):.0f} ms</div><div class="metric-label">Evaluation</div></div>', unsafe_allow_html=True)

            st.write("")

            # ---------------------------------------------
            # LLM-as-Judge & Hallucination Component
            # ---------------------------------------------
            st.markdown("### 🔬 Automated Evaluation & Hallucination Detection")

            faithfulness = evaluation["faithfulness"]
            relevancy = evaluation["answer_relevancy"]
            judge_score = evaluation["judge_score"]

            if faithfulness >= 75:
                status_markup = '<span class="badge badge-green">✅ WELL GROUNDED</span>'
            elif faithfulness >= 50:
                status_markup = '<span class="badge badge-amber">⚠️ PARTIALLY SUPPORTED</span>'
            else:
                status_markup = '<span class="badge badge-red">🔴 POTENTIAL HALLUCINATION</span>'

            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                st.metric("Faithfulness Score", f"{faithfulness:.1f}%")
                st.markdown(status_markup, unsafe_allow_html=True)
            with ec2:
                st.metric("Answer Relevancy", f"{relevancy:.1f}%")
            with ec3:
                st.metric("LLM Judge Score", f"{judge_score:.1f}%")
            with ec4:
                st.metric("Confidence Score", f"{confidence:.1f}%")

            if evaluation.get("reasoning"):
                st.markdown(f'<div class="rationale-box"><strong>🧠 LLM Judge Assessment Rationale:</strong><br>{evaluation["reasoning"]}</div>', unsafe_allow_html=True)

            # ---------------------------------------------
            # Sources Component
            # ---------------------------------------------
            st.markdown("### 📚 Retrieved Sources & Context Passages")
            if not sources:
                st.info("No explicit high-confidence source citations generated.")
            else:
                for idx, src_name in enumerate(sources):
                    with st.expander(f"📄 Citation {idx + 1}: {src_name}", expanded=(idx == 0)):
                        if idx < len(chunks):
                            chunk_obj = chunks[idx]
                            c_text = chunk_obj.get("text", str(chunk_obj)) if isinstance(chunk_obj, dict) else str(chunk_obj)
                            st.write(c_text)

        except Exception as query_err:
            logger.error(f"Error executing RAG query: {query_err}", exc_info=True)
            st.error(f"Inference Failure: {query_err}")

# =========================================================
# TAB 2: DOCUMENT MANAGEMENT & INGESTION
# =========================================================
with tab2:
    st.subheader("Document Ingestion & Index Management")
    st.write("Upload PDF documents to parse text, chunk content, generate embeddings, and update FAISS vector indices.")

    uploaded_pdf = st.file_uploader("Upload Knowledge PDF Document", type=["pdf"])

    if uploaded_pdf:
        try:
            pdf_bytes = uploaded_pdf.getbuffer()
            file_hash = hashlib.sha256(pdf_bytes).hexdigest()
            upload_sig = f"{uploaded_pdf.name}:{file_hash}"

            save_dir = "data/uploads"
            os.makedirs(save_dir, exist_ok=True)
            saved_file_path = os.path.join(save_dir, uploaded_pdf.name)

            with open(saved_file_path, "wb") as f:
                f.write(pdf_bytes)

            if upload_sig not in st.session_state.processed_uploads:
                with st.spinner(f"🔄 Ingesting '{uploaded_pdf.name}': Extracting PDF → Chunking → Generating Embeddings → Updating FAISS"):
                    ingestion.ingest(pdf_path=saved_file_path)
                st.session_state.processed_uploads.add(upload_sig)

            st.success(f"✅ Document Successfully Ingested: **{uploaded_pdf.name}**")
            
            st.markdown("""
            - ✓ PDF Loaded & Parsed
            - ✓ Text Chunks Created
            - ✓ Sentence-Transformer Embeddings (`all-MiniLM-L6-v2`) Generated
            - ✓ Saved to SQLite Chunk Store & FAISS Vector Index Updated
            """)
        except Exception as ingest_err:
            logger.error(f"Document Ingestion failed: {ingest_err}")
            st.error(f"Ingestion Error: {ingest_err}")

# =========================================================
# TAB 3: EVALUATION & OBSERVABILITY DASHBOARD
# =========================================================
with tab3:
    st.subheader("Analytics & Observability Dashboard")

    if st.session_state.evaluation_history:
        evals = st.session_state.evaluation_history

        faith_list = [e["faithfulness"] for e in evals]
        rel_list = [e["answer_relevancy"] for e in evals]
        judge_list = [e["judge_score"] for e in evals]
        lat_list = [e.get("total_latency_ms", 0) for e in evals]

        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            st.metric("Mean Faithfulness", f"{sum(faith_list)/len(faith_list):.1f}%")
        with ac2:
            st.metric("Mean Relevancy", f"{sum(rel_list)/len(rel_list):.1f}%")
        with ac3:
            st.metric("Mean LLM Judge Score", f"{sum(judge_list)/len(judge_list):.1f}%")
        with ac4:
            st.metric("Mean Total Latency", f"{sum(lat_list)/len(lat_list):.0f} ms")

        st.markdown("---")
        st.subheader("Metric Trends Across Queries")

        eval_df = pd.DataFrame({
            "Query Index": range(1, len(evals) + 1),
            "Faithfulness (%)": faith_list,
            "Relevancy (%)": rel_list,
            "Judge Score (%)": judge_list,
            "Total Latency (ms)": lat_list
        })

        st.line_chart(eval_df.set_index("Query Index")[["Faithfulness (%)", "Relevancy (%)", "Judge Score (%)"]])
        st.caption("Execution Latency (ms) Per Query")
        st.bar_chart(eval_df.set_index("Query Index")[["Total Latency (ms)"]])

        st.markdown("---")
        st.subheader("Detailed Evaluation Ledger")
        st.dataframe(eval_df, use_container_width=True)
    else:
        st.info("No query evaluation logs available in current session.")

# =========================================================
# TAB 4: RETRIEVAL BENCHMARKING
# =========================================================
with tab4:
    st.subheader("Retrieval Engine Benchmarking Suite")
    st.write("Compare **Vector (FAISS)**, **BM25 (Okapi)**, and **Hybrid (RRF)** search performance on Precision@K, Recall@K, MRR, and Latency.")

    bc1, bc2 = st.columns([2, 1])
    with bc1:
        b_k = st.slider("Benchmark Top K", min_value=1, max_value=10, value=5, key="b_top_k_slider")
    with bc2:
        st.write("")
        run_bench = st.button("⚡ Execute Benchmarking Test", use_container_width=True)

    if run_bench:
        with st.spinner("🔄 Benchmarking retrieval strategies across dataset..."):
            try:
                benchmarker = RetrievalBenchmarker(retriever=pipeline.retriever)
                bench_df = benchmarker.run_benchmark(top_k=b_k)

                st.success("✅ Benchmarking Complete!")
                st.subheader("📊 Comparative Benchmark Results")
                st.dataframe(bench_df, use_container_width=True)

                st.subheader("📈 Performance Visualization")
                vc1, vc2 = st.columns(2)
                with vc1:
                    st.markdown("**Precision@K & Recall@K (%)**")
                    st.bar_chart(bench_df.set_index("Strategy")[["Precision@K", "Recall@K"]])
                with vc2:
                    st.markdown("**Average Latency (ms)**")
                    st.bar_chart(bench_df.set_index("Strategy")[["Avg Latency (ms)"]])

            except Exception as bench_err:
                logger.error(f"Benchmarking error: {bench_err}")
                st.error(f"Benchmarking Error: {bench_err}")
