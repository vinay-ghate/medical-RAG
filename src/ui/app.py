"""Streamlit Chat UI for Medical Q&A Assistant."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.config import settings
from src.generation.generator import MedicalRAGPipeline
from src.safety.guardrails import STANDARD_DISCLAIMER

st.set_page_config(
    page_title="Medical Q&A Assistant",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource(show_spinner="Initializing Medical RAG Pipeline...")
def load_pipeline():
    """Cache and initialize pipeline components."""
    return MedicalRAGPipeline()


def main():
    st.title("🩺 Medical Q&A Assistant")
    st.caption("Grounded clinical answers backed by official FDA labels and clinical practice guidelines.")

    # Sidebar settings & info
    with st.sidebar:
        st.header("⚙️ System Configuration")
        st.markdown(f"**LLM Provider:** `{settings.llm_provider.upper()}`")
        st.markdown(f"**Model:** `{settings.groq_model}`")
        st.markdown(f"**Embeddings:** `{settings.embedding_model.split('/')[-1]}`")
        st.markdown(f"**Reranker:** `{settings.reranker_model}`")
        st.markdown(f"**Search:** Hybrid (BM25 + Dense) + Rerank")

        # Allow dynamic API key entry if not in environment
        groq_key_input = st.text_input(
            "Groq API Key",
            value=settings.groq_api_key or os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Free key from https://console.groq.com",
        )
        if groq_key_input:
            settings.groq_api_key = groq_key_input
            os.environ["GROQ_API_KEY"] = groq_key_input

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("### ⚠️ Notice")
        st.info("This system answers solely from verified medical references. No diagnosis or dosing.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your grounded medical reference assistant. Ask me questions about drug indications, contraindications, boxed warnings, or guideline recommendations.",
                "citations": [],
            }
        ]

    # Render chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📚 View Cited Sources ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.markdown(
                            f"**[{c['index']}] {c['drug']} — {c['section']}**  \n"
                            f"*Source:* [{c['source']}]({c['source']}) | *Date:* {c['date']} | *Rerank Score:* `{c.get('rerank_score', 'N/A')}`  \n"
                            f"> {c['excerpt']}"
                        )

    # User input
    if prompt := st.chat_input("Ask a clinical or drug question (e.g., 'What are the contraindications of Metformin?')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        pipeline = load_pipeline()

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and verifying citations..."):
                try:
                    response = pipeline.run(prompt)
                    st.markdown(response.answer)

                    if response.citations:
                        with st.expander(f"📚 View Cited Sources ({len(response.citations)})"):
                            for c in response.citations:
                                st.markdown(
                                    f"**[{c['index']}] {c['drug']} — {c['section']}**  \n"
                                    f"*Source:* [{c['source']}]({c['source']}) | *Date:* {c['date']} | *Rerank Score:* `{c.get('rerank_score', 'N/A')}`  \n"
                                    f"> {c['excerpt']}"
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "citations": response.citations,
                        }
                    )
                except Exception as e:
                    error_msg = f"An error occurred while generating the answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "citations": []}
                    )


if __name__ == "__main__":
    main()
