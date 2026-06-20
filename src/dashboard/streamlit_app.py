import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.append(PROJECT_ROOT)

import streamlit as st

from src.database.sqlite_manager import SQLiteManager
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.ingestion.document_ingestion_pipeline import (
    DocumentIngestionPipeline
)

# ----------------------------
# Streamlit Config
# ----------------------------

st.set_page_config(
    page_title="Production RAG Platform",
    layout="wide"
)

st.title("Production Grade RAG Platform")

# ----------------------------
# Services
# ----------------------------

@st.cache_resource
def load_pipeline():
    return RAGPipeline()


@st.cache_resource
def load_evaluator():
    return RagasEvaluator()


pipeline = load_pipeline()
evaluator = load_evaluator()

db = SQLiteManager()

ingestion = DocumentIngestionPipeline()

# ----------------------------
# Session State
# ----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:

    st.header("Query History")

    if not st.session_state.history:
        st.info("No queries yet")
    else:
        for q in st.session_state.history:
            st.write(f"• {q}")

# ----------------------------
# Tabs
# ----------------------------

tab1, tab2 = st.tabs(
    [
        "Ask Question",
        "Category Management"
    ]
)

# ==================================================
# TAB 2 - CATEGORY MANAGEMENT
# ==================================================

with tab2:

    st.subheader("Create Category")

    new_category = st.text_input(
        "Category Name"
    )

    if st.button("Create Category"):

        if new_category.strip():

            db.create_category(
                new_category.strip()
            )

            st.success(
                "Category Created Successfully"
            )

        else:

            st.warning(
                "Enter Category Name"
            )

    st.subheader(
        "Existing Categories"
    )

    categories = db.get_categories()

    if categories:

        for category in categories:

            st.write(
                f"• {category}"
            )

    else:

        st.info(
            "No Categories Found"
        )

# ==================================================
# TAB 1 - ASK QUESTION
# ==================================================

with tab1:

    categories = db.get_categories()

    if not categories:

        st.warning(
            "No categories found. Create one first."
        )

        st.stop()

    selected_category = st.selectbox(
        "Select Category",
        categories
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    # ------------------------------------------
    # PDF Upload & Ingestion
    # ------------------------------------------

    if uploaded_file:

        upload_dir = "data/uploads"

        os.makedirs(
            upload_dir,
            exist_ok=True
        )

        file_path = os.path.join(
            upload_dir,
            uploaded_file.name
        )

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        with st.spinner(
            "Processing document..."
        ):

            ingestion.ingest(
                pdf_path=file_path,
                category=selected_category
            )

        st.success(
            f"""
Uploaded '{uploaded_file.name}'

✓ PDF Loaded

✓ Chunks Generated

✓ Embeddings Created

✓ Saved To Database

✓ FAISS Index Updated

Document is now ready for querying.
"""
        )

    # ------------------------------------------
    # Query Section
    # ------------------------------------------

    query = st.text_input(
        "Ask Question"
    )

    if st.button(
        "Generate Answer"
    ):

        if not query.strip():

            st.warning(
                "Enter a question"
            )

            st.stop()

        with st.spinner(
            "Generating..."
        ):

            result = pipeline.run(
                query=query,
                category=selected_category
            )

            st.session_state.history.insert(
                0,
                query
            )

            st.session_state.history = (
                st.session_state.history[:10]
            )

        answer = result["answer"]
        sources = result["sources"]
        chunks = result["chunks"]
        confidence = result["confidence"]
        evaluation = result["evaluation"]

        # --------------------
        # Answer
        # --------------------

        st.subheader("Answer")

        st.write(answer)

        # --------------------
        # Sources + Chunks
        # --------------------

        st.subheader("Sources")

        for i, source in enumerate(
            sources
        ):

            st.write(
                f"• {source}"
            )

            if i < len(chunks):

                with st.expander(
                    f"View Chunk {i + 1}"
                ):

                    chunk = chunks[i]

                    if isinstance(
                        chunk,
                        dict
                    ):

                        st.write(
                            chunk.get(
                                "text",
                                "No content found"
                            )
                        )

                    else:

                        st.write(chunk)

        # --------------------
        # Evaluation
        # --------------------

        st.subheader(
            "Evaluation"
        )

        if evaluation[
            "faithfulness"
        ] < 50:

            st.error(
                "Potential Hallucination Detected"
            )

        elif evaluation[
            "faithfulness"
        ] < 75:

            st.warning(
                "Answer Partially Supported By Context"
            )

        else:

            st.success(
                "Answer Well Grounded In Context"
            )

        # --------------------
        # Metrics
        # --------------------

        judge_score = round(
            (
                evaluation["faithfulness"]
                +
                evaluation["answer_relevancy"]
            ) / 2,
            2
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        c2.metric(
            "Faithfulness",
            f"{evaluation['faithfulness']:.2f}%"
        )

        c3.metric(
            "Relevancy",
            f"{evaluation['answer_relevancy']:.2f}%"
        )

        c4.metric(
            "Judge",
            f"{judge_score:.2f}%"
        )
