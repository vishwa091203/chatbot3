import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

# -----------------------------
# PAGE SETUP (must be first Streamlit call)
# -----------------------------
st.set_page_config(page_title="Workout & Diet Chatbot", page_icon="💪")
st.title("Workout & Diet Chatbot")
st.caption("Ask questions based on your workout and diet PDFs")

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()

secret_key = None
if "GROQ_API_KEY" in st.secrets:
    secret_key = st.secrets["GROQ_API_KEY"]

groq_api_key = os.getenv("GROQ_API_KEY") or secret_key

if not groq_api_key:
    st.error(
        "Missing GROQ_API_KEY. Add it to your environment (.env) or Streamlit secrets."
    )
    st.stop()

client = Groq(api_key=groq_api_key)


# -----------------------------
# HELPERS
# -----------------------------
def split_text(text: str, chunk_size: int = 1200) -> list[str]:
    """Split text into fixed-size chunks."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def tokenize(text: str) -> set[str]:
    """Simple tokenization for lightweight keyword matching."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {w for w in cleaned.split() if len(w) > 2}


def rank_chunks(query: str, chunks: list[str], top_k: int = 4) -> list[str]:
    """Rank chunks by overlap with query tokens."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return chunks[:top_k]

    scored: list[tuple[int, int, str]] = []
    for idx, chunk in enumerate(chunks):
        chunk_tokens = tokenize(chunk)
        overlap = len(query_tokens & chunk_tokens)
        scored.append((overlap, -idx, chunk))

    scored.sort(reverse=True)

    best = [chunk for score, _, chunk in scored if score > 0][:top_k]
    return best if best else chunks[:top_k]


@st.cache_resource
def load_pdfs() -> list[str]:
    """Load PDF content and return chunks for retrieval."""
    pdf_files = ["workoutinformation.pdf", "dietinformation.pdf"]
    all_text_parts: list[str] = []

    for file_name in pdf_files:
        if not os.path.exists(file_name):
            st.error(f"Missing file: {file_name}")
            st.stop()

        try:
            reader = PdfReader(file_name)
        except Exception as exc:
            st.error(f"Could not open {file_name}: {exc}")
            st.stop()

        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                all_text_parts.append(page_text)

    full_text = "\n".join(all_text_parts).strip()
    if not full_text:
        st.error("No readable text found in the provided PDFs.")
        st.stop()

    chunks = split_text(full_text)
    if not chunks:
        st.error("Unable to prepare context chunks from PDFs.")
        st.stop()

    return chunks


# -----------------------------
# DATA
# -----------------------------
chunks = load_pdfs()

# -----------------------------
# CHAT STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask a workout or diet question...")

if prompt:
    st.session_state.chat_history.append(("user", prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context_chunks = rank_chunks(prompt, chunks, top_k=4)
            context_text = "\n\n".join(context_chunks)[:5000]

            try:
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a fitness and nutrition assistant. "
                                "Answer only using the provided context. "
                                "If the answer is not present, clearly say you cannot find it in the PDFs."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context_text}\n\nQuestion:\n{prompt}",
                        },
                    ],
                )
                answer = completion.choices[0].message.content or "I couldn't generate a response."
            except Exception as exc:
                answer = f"Error while generating response: {exc}"

            st.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))
