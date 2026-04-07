from dotenv import load_dotenv
import os
import streamlit as st
from pypdf import PdfReader
from groq import Groq

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found")
    st.stop()

client = Groq(api_key=groq_api_key)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Workout & Diet Chatbot")
st.title("Workout & Diet Chatbot")
st.caption("Ask questions from your PDFs")

# -----------------------------
# LOAD PDFs (NO LANGCHAIN)
# -----------------------------
@st.cache_resource
def load_pdfs():
    pdf_files = ["workoutinformation.pdf", "dietinformation.pdf"]

    all_text = ""

    for file in pdf_files:
        if not os.path.exists(file):
            st.error(f"Missing file: {file}")
            st.stop()

        try:
            reader = PdfReader(file)
            for page in reader.pages:
                all_text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
            st.stop()

    return all_text

# -----------------------------
# SIMPLE CHUNKING
# -----------------------------
def split_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# -----------------------------
# LOAD DATA
# -----------------------------
full_text = load_pdfs()
chunks = split_text(full_text)

# -----------------------------
# CHAT UI
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Simple retrieval
            relevant_chunks = [c for c in chunks if user_input.lower() in c.lower()][:4]

            if not relevant_chunks:
                context = chunks[:4]
            else:
                context = relevant_chunks

            context_text = "\n\n".join(context)[:3000]

            try:
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful fitness and diet assistant. Answer only from the given context."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context_text}\n\nQuestion:\n{user_input}"
                        }
                    ]
                )

                answer = response.choices[0].message.content

            except Exception as e:
                answer = f"Error: {str(e)}"

            st.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))
