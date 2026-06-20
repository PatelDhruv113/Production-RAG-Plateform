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
import pandas as pd

from src.database.sqlite_manager import SQLiteManager
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.ragas_evaluator import RagasEvaluator


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


@st.cache_resource
def load_evaluator():
    return RagasEvaluator()


pipeline = load_pipeline()
evaluator = load_evaluator()
db = SQLiteManager()

st.set_page_config(
    page_title="Production RAG Platform",
    layout="wide"
)

st.title("Production Grade RAG Platform")

if "history" not in st.session_state:
    st.session_state.history = []
with st.sidebar:

    st.header(
        "Query History"
    )

    if len(
        st.session_state.history
    ) == 0:

        st.info(
            "No queries yet"
        )

    else:

        for q in st.session_state.history:

            st.write(
                f"• {q}"
            )

query = st.text_input(
    "Ask Question"
)

if st.button("Generate Answer"):

    if not query.strip():
        st.warning("Enter a question")
        st.stop()

    with st.spinner("Generating..."):

        result = pipeline.run(query)
        
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

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for i, source in enumerate(sources):

        st.write(f"• {source}")

        if i < len(chunks):

            with st.expander(
                f"View Chunk {i + 1}"
            ):

                chunk = chunks[i]

                if isinstance(chunk, dict):

                    st.write(
                        chunk.get(
                            "text",
                            "No content found"
                        )
                    )

                else:

                    st.write(chunk)

    st.subheader("Evaluation")

    if evaluation["faithfulness"] < 50:

        st.error(
            "⚠ Potential Hallucination Detected"
        )

    elif evaluation["faithfulness"] < 75:

        st.warning(
            "⚠ Answer Partially Supported By Retrieved Context"
        )

    else:

        st.success(
            "✓ Answer Well Grounded In Retrieved Context"
        )

    try:

        judge_score = round(
            (
                evaluation["faithfulness"]
                + evaluation["answer_relevancy"]
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

    except Exception as e:

        st.error(
            f"Evaluation Error: {e}"
        )