import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from openai import OpenAI
# NEW LANGCHAIN IMPORTS (IMPORTANT)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# =====================================
# CONFIGURATION
# =====================================

st.set_page_config(page_title="AutoSphere Motors AI", layout="wide")
st.title("🚗 AutoSphere Motors AI Assistant")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Please add it to your .env file.")
    st.stop()

LLM_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"

client = OpenAI(api_key=OPENAI_API_KEY)

# =====================================
# BUILD OR LOAD VECTOR STORE
# =====================================

@st.cache_resource
def load_vectorstore():

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    if not os.path.exists("vectorstore"):

        if not os.path.exists("autosphere_policy.docx"):
            st.error("❌ autosphere_policy.docx file not found in project folder.")
            st.stop()

        loader = Docx2txtLoader("autosphere_policy.docx")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        docs = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("vectorstore")

    else:
        vectorstore = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

    return vectorstore


vectorstore = load_vectorstore()

# =====================================
# INTENT CLASSIFIER
# =====================================

def classify_intent(user_message):

    prompt = f"""
    Classify the intent strictly into one of:
    - car_suggestion
    - service_booking
    - test_drive
    - spare_parts_price
    - complaint
    - branch_hours
    - vat_query
    - general_question

    Message: {user_message}

    Return ONLY the intent name.
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip().lower()

# =====================================
# BOOKING VALIDATION RULES
# =====================================

def validate_service_booking(service_type, selected_date):

    now = datetime.now()

    # Must be 24 hours in advance
    if selected_date < now + timedelta(hours=24):
        return False, "⚠ Service must be booked at least 24 hours in advance."

    # Weekend restriction (Saturday/Sunday logic)
    if selected_date.weekday() in [5, 6]:
        if service_type.lower() not in ["oil change", "quick service"]:
            return False, "⚠ Weekend allows only Quick Service."

    return True, "✅ Booking allowed."

# =====================================
# LLM RESPONSE GENERATOR
# =====================================

SYSTEM_PROMPT = """
You are AutoSphere Motors AI Assistant.

STRICT RULES:
- Never invent prices.
- Always follow company policies.
- Use VAT rules correctly.
- If unsure, say you will check with branch.
- For booking, validate rules before confirming.
"""

def generate_response(user_message, context=""):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context + "\n\nUser: " + user_message}
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content

# =====================================
# STREAMLIT CHAT UI
# =====================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask about cars, booking, pricing...")

if user_input:

    try:
        intent = classify_intent(user_input)
    except Exception as e:
        st.error(f"Intent classification error: {e}")
        st.stop()

    context = ""

    # Use RAG only for knowledge-based queries
    if intent in ["general_question", "spare_parts_price", "branch_hours", "vat_query"]:
        docs = vectorstore.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in docs])

    try:
        response = generate_response(user_input, context)
    except Exception as e:
        st.error(f"LLM generation error: {e}")
        st.stop()

    st.session_state.chat_history.append(("User", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "User":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)